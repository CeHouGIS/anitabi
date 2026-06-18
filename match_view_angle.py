"""
在 GSV 等矩形全景图中，找到与动漫帧视角最接近的方向。

方法：
  1. 等矩形全景 → 每隔 YAW_STEP° 做透视重投影（rectilinear projection）
  2. 动漫帧 + 每张截图 → Canny 边缘图（消除风格差异）
  3. 归一化互相关（NCC）打分，取 Top-K 角度
  4. 保存对比图：动漫帧 | 最佳街景截图

依赖：Pillow, NumPy, SciPy, OpenCV, scikit-image
"""
import math, itertools
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
import cv2


# ── 参数 ──────────────────────────────────────────────────
YAW_STEP   = 5        # 偏航扫描步长（度），越小越精细
PITCH_LIST = [-5, 0, 5]  # 俯仰角候选（度）
FOV        = 90       # 透视截图水平视场角（度）
OUT_W, OUT_H = 640, 360  # 透视截图尺寸

TOP_K  = 3            # 保存 Top-K 个最佳角度
SAVE_DIR = Path('outputs/view_match')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SITES = [
    ('S_27364_0002', '冰菓·本町商店街'),
    ('S_27364_0004', '冰菓·鍛冶橋交差点'),
    ('S_27364_0026', '冰菓·宮川朝市'),
    ('S_10440_0030', '未闻花名·旧秩父橋'),
    ('S_10440_0051', '未闻花名·秩父神社'),
    ('S_207195_0069', '摇曳露营·ローソン南アルプス'),
    ('S_207195_0075', '摇曳露营·夜叉神峠登山口'),
]

PANO_DIR  = Path('docs/img/pano')
ANIME_DIR = Path('docs/img/anime')


# ── 透视重投影 ────────────────────────────────────────────
def equirect_to_persp(pano_arr, yaw_deg, pitch_deg=0,
                      fov_deg=FOV, out_w=OUT_W, out_h=OUT_H):
    """从等矩形全景中截取一张透视图。"""
    h, w = pano_arr.shape[:2]
    yaw   = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    f     = (out_w / 2) / math.tan(math.radians(fov_deg) / 2)

    # 输出像素坐标网格（相机坐标系）
    xs = np.arange(out_w) - out_w / 2
    ys = np.arange(out_h) - out_h / 2
    xv, yv = np.meshgrid(xs, ys)
    zv = np.full_like(xv, f, dtype=np.float32)
    norm = np.sqrt(xv**2 + yv**2 + zv**2)
    xn, yn, zn = xv/norm, -yv/norm, zv/norm  # y轴向上

    # 绕 X 轴旋转（pitch）
    cp, sp = math.cos(pitch), math.sin(pitch)
    yn2 = yn*cp - zn*sp
    zn2 = yn*sp + zn*cp

    # 绕 Y 轴旋转（yaw）
    cy, sy = math.cos(yaw), math.sin(yaw)
    xn2 =  xn*cy + zn2*sy
    zn3 = -xn*sy + zn2*cy
    yn3 = yn2

    # 方向向量 → 球面坐标
    lon = np.arctan2(xn2, zn3)          # [-π, π]
    lat = np.arcsin(np.clip(yn3, -1, 1))  # [-π/2, π/2]

    # 球面坐标 → 等矩形像素坐标
    px = (lon / (2*math.pi) + 0.5) * w
    py = (0.5 - lat / math.pi) * h

    # 双线性插值采样
    coords = [py.flatten(), px.flatten()]
    channels = [
        map_coordinates(pano_arr[:, :, c], coords, order=1, mode='wrap')
        .reshape(out_h, out_w).astype(np.uint8)
        for c in range(pano_arr.shape[2])
    ]
    return np.stack(channels, axis=2)


# ── 边缘相似度 ────────────────────────────────────────────
def edge_map(img_arr):
    gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    # 自适应阈值：基于图像中位梯度
    med = np.median(gray)
    lo, hi = max(0, int(med * 0.5)), min(255, int(med * 1.5))
    return cv2.Canny(gray, lo, hi).astype(np.float32) / 255.0


def ncc(a, b):
    """归一化互相关（−1→1，越高越相似）。"""
    a = a - a.mean(); b = b - b.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum()) + 1e-8
    return float((a * b).sum() / denom)


def color_hist_sim(a_arr, b_arr, bins=32):
    """HSV 色调直方图相似度（0→1）。"""
    def hsv_h(arr):
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        h = cv2.calcHist([hsv], [0], None, [bins], [0, 180]).flatten()
        h = h / (h.sum() + 1e-8)
        return h
    ha, hb = hsv_h(a_arr), hsv_h(b_arr)
    return float(cv2.compareHist(ha.reshape(-1,1).astype(np.float32),
                                 hb.reshape(-1,1).astype(np.float32),
                                 cv2.HISTCMP_CORREL))


def combined_score(anime_arr, gsv_arr, w_edge=0.7, w_color=0.3):
    ae = edge_map(anime_arr)
    ge = edge_map(gsv_arr)
    edge_s  = ncc(ae, ge)
    color_s = color_hist_sim(anime_arr, gsv_arr)
    return w_edge * edge_s + w_color * color_s, edge_s, color_s


# ── 保存对比图 ────────────────────────────────────────────
def save_comparison(site_id, label, anime_arr, results):
    """保存：动漫帧 + Top-K 街景截图横排。"""
    pad = 8
    h = OUT_H
    anime_rsz = np.array(Image.fromarray(anime_arr).resize((OUT_W, h)))

    strips = [anime_rsz]
    for rank, (score, yaw, pitch, gsv_arr, edge_s, color_s) in enumerate(results):
        # 在截图上叠加角度标注
        overlay = gsv_arr.copy()
        cv2.putText(overlay, f"Yaw={yaw}\xb0  Pitch={pitch}\xb0",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,100), 1, cv2.LINE_AA)
        cv2.putText(overlay, f"Score={score:.3f}  edge={edge_s:.3f}  color={color_s:.3f}",
                    (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,230,255), 1, cv2.LINE_AA)
        strips.append(overlay)

    # 水平拼接（带白色分隔线）
    n = len(strips)
    total_w = n * OUT_W + (n - 1) * pad
    canvas = np.full((h, total_w, 3), 40, dtype=np.uint8)
    for i, s in enumerate(strips):
        x = i * (OUT_W + pad)
        canvas[:, x:x+OUT_W] = s

    # 顶部标签
    header = np.full((30, total_w, 3), 25, dtype=np.uint8)
    cv2.putText(header, f"[动漫帧]  {'→  [街景 Top'+str(len(results))+']':>30}  {label}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 210, 255), 1, cv2.LINE_AA)
    out = np.vstack([header, canvas])

    out_path = SAVE_DIR / f"{site_id}_match.jpg"
    Image.fromarray(out).save(str(out_path), quality=90)
    return out_path


# ── 主流程 ────────────────────────────────────────────────
def main():
    yaws = list(range(0, 360, YAW_STEP))
    combos = list(itertools.product(yaws, PITCH_LIST))
    print(f"扫描角度组合数：{len(combos)} 个（yaw×pitch = {len(yaws)}×{len(PITCH_LIST)}）\n")

    summary = []
    for site_id, label in SITES:
        pano_path  = PANO_DIR  / f"{site_id}.jpg"
        anime_path = ANIME_DIR / f"{site_id}.jpg"
        if not pano_path.exists() or not anime_path.exists():
            print(f"[跳过] {label}: 文件缺失")
            continue

        print(f"{'─'*60}")
        print(f"处理：{label}")

        pano_arr  = np.array(Image.open(pano_path).convert('RGB'))
        anime_arr = np.array(Image.open(anime_path).convert('RGB').resize((OUT_W, OUT_H)))

        # 扫描所有角度
        scores = []
        for yaw, pitch in combos:
            gsv_crop = equirect_to_persp(pano_arr, yaw, pitch)
            s, es, cs = combined_score(anime_arr, gsv_crop)
            scores.append((s, yaw, pitch, gsv_crop, es, cs))

        scores.sort(reverse=True)
        top = scores[:TOP_K]

        print(f"  Top-{TOP_K} 匹配角度：")
        for rank, (s, yaw, pitch, _, es, cs) in enumerate(top, 1):
            print(f"    #{rank}  Yaw={yaw:3d}°  Pitch={pitch:+d}°  "
                  f"综合={s:.4f}  边缘={es:.4f}  色调={cs:.4f}")

        out_path = save_comparison(site_id, label, anime_arr, top)
        print(f"  → 对比图已保存：{out_path}")
        summary.append((label, top[0][1], top[0][2], top[0][0]))

    print(f"\n{'='*60}")
    print("汇总（最佳匹配角度）：")
    print(f"  {'圣地':<28} {'Yaw':>6} {'Pitch':>6} {'综合分':>8}")
    print(f"  {'-'*52}")
    for label, yaw, pitch, s in summary:
        print(f"  {label:<28} {yaw:>5}°  {pitch:>+5}°  {s:>8.4f}")
    print()
    print(f"对比图保存在：{SAVE_DIR}/")


if __name__ == '__main__':
    main()
