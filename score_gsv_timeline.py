"""
用训练好的 PerceptionNet 对所有历史街景全景打分。

每张等矩形全景 → 4 个透视截图（yaw=0/90/180/270°）→ 推理 → 平均
输出：data/processed/gsv_perception_scores.csv
"""
import json, math, itertools, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
import torchvision.models as models
import torchvision.transforms as T
from scipy.ndimage import map_coordinates

warnings.filterwarnings('ignore')

HISTORY_DIR = Path('data/raw/streetview/history')
MODEL_PATH  = Path('models/perception_net.pth')
STATS_PATH  = Path('models/score_stats.json')
OUT_CSV     = Path('data/processed/gsv_perception_scores.csv')
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

ATTRS = ['safety', 'lively', 'boring', 'wealthy', 'depressing', 'beautiful']
YAWS  = [0, 90, 180, 270]   # 4 个方向均值
FOV   = 90
OUT_W, OUT_H = 224, 224

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]


# ── 模型 ──────────────────────────────────────────────────
class PerceptionNet(nn.Module):
    def __init__(self, num_attrs=6):
        super().__init__()
        backbone = models.resnet50(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_attrs),
        )
    def forward(self, x):
        return self.head(self.features(x))


def load_model():
    m = PerceptionNet().to(DEVICE)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    m.eval()
    return m


# ── 透视重投影 ─────────────────────────────────────────────
def equirect_to_persp(pano_arr, yaw_deg, pitch_deg=0, fov_deg=FOV,
                      out_w=OUT_W, out_h=OUT_H):
    h, w = pano_arr.shape[:2]
    yaw   = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    f     = (out_w / 2) / math.tan(math.radians(fov_deg) / 2)

    xs = np.arange(out_w) - out_w / 2
    ys = np.arange(out_h) - out_h / 2
    xv, yv = np.meshgrid(xs, ys)
    zv = np.full_like(xv, f, dtype=np.float32)
    norm = np.sqrt(xv**2 + yv**2 + zv**2)
    xn, yn, zn = xv/norm, -yv/norm, zv/norm

    cp, sp = math.cos(pitch), math.sin(pitch)
    yn2 = yn*cp - zn*sp; zn2 = yn*sp + zn*cp

    cy, sy = math.cos(yaw), math.sin(yaw)
    xn2 =  xn*cy + zn2*sy
    zn3 = -xn*sy + zn2*cy
    yn3 = yn2

    lon = np.arctan2(xn2, zn3)
    lat = np.arcsin(np.clip(yn3, -1, 1))
    px  = (lon / (2*math.pi) + 0.5) * w
    py  = (0.5 - lat / math.pi) * h

    coords = [py.flatten(), px.flatten()]
    channels = [
        map_coordinates(pano_arr[:,:,c], coords, order=1, mode='wrap')
        .reshape(out_h, out_w).astype(np.uint8)
        for c in range(3)
    ]
    return np.stack(channels, axis=2)


# ── 预处理 ────────────────────────────────────────────────
transform = T.Compose([
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float32),
    T.Normalize(IMG_MEAN, IMG_STD),
])


def score_pano(model, pano_path):
    """对一张全景图取4个方向截图，推理后返回平均分（0-10 归一化空间）。"""
    pano_arr = np.array(Image.open(pano_path).convert('RGB'))
    crops = []
    for yaw in YAWS:
        crop = equirect_to_persp(pano_arr, yaw)
        img  = Image.fromarray(crop)
        t    = transform(img).unsqueeze(0).to(DEVICE)
        crops.append(t)

    batch = torch.cat(crops, dim=0)   # [4, 3, H, W]
    with torch.no_grad():
        preds = model(batch)           # [4, 6]
    return preds.mean(dim=0).cpu().numpy()   # [6]


# ── 主流程 ────────────────────────────────────────────────
def main():
    model = load_model()
    print(f"模型加载完成  device={DEVICE}\n")

    with open(STATS_PATH) as f:
        stats = json.load(f)

    records = []
    site_dirs = sorted(HISTORY_DIR.iterdir())

    for site_dir in site_dirs:
        if not site_dir.is_dir():
            continue
        site_id = site_dir.name
        pano_files = sorted(site_dir.glob('*.jpg'))
        print(f"{'─'*55}")
        print(f"{site_id}  ({len(pano_files)} 个版本)")

        for pano_path in pano_files:
            date_str = pano_path.stem   # YYYY-MM
            raw_scores = score_pano(model, pano_path)

            # 反归一化回 TrueSkill 量纲（可选，保留归一化空间更直观）
            row = {'site_id': site_id, 'date': date_str}
            for i, attr in enumerate(ATTRS):
                row[attr] = round(float(raw_scores[i]), 4)
            records.append(row)

            score_str = "  ".join(f"{a}={row[a]:.2f}" for a in ['lively','beautiful','safety'])
            print(f"  [{date_str}]  {score_str}")

    df = pd.DataFrame(records)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n{'='*55}")
    print(f"评分完成，共 {len(df)} 条记录")
    print(f"保存 → {OUT_CSV}")

    # ── 汇总：各圣地 lively 分时间趋势 ──────────────────
    print("\n各圣地 lively（活力）分时间序列：")
    for sid, grp in df.groupby('site_id'):
        vals = "  ".join(f"{r.date}:{r.lively:.1f}" for _, r in grp.iterrows())
        print(f"  {sid:<20} {vals}")


if __name__ == '__main__':
    main()
