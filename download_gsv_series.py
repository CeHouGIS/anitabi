"""
Phase 3 · 下载历史 GSV 全景（zoom=2，2048×1024）并即时评分

输入：data/raw/sites_eligible.csv（fetch_gsv_meta.py 输出）

流程（每个圣地 × 每个季度快照）：
  1. 检查全景文件是否已存在（断点续跑）
  2. 下载 8 张瓦片（zoom=2，4×2 格局）→ 拼接 2048×1024 全景
  3. 保存 JPEG（Strategy B，约 1-2 MB/张）
  4. 对当前全景取 4 方向透视截图 → 推理 → 追加评分

输出：
  data/raw/streetview/series/{site_id}/{YYYY-MM}.jpg  ← 全景图
  data/processed/panel_scores.csv                     ← 长面板评分表
"""

import csv, json, re, time, math, itertools, logging
from pathlib import Path
from io import BytesIO

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from scipy.ndimage import map_coordinates
import requests

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# ── 路径 ──────────────────────────────────────────────────
SITES_ELIGIBLE = Path('data/raw/sites_eligible.csv')
SERIES_DIR     = Path('data/raw/streetview/series')
MODEL_PATH     = Path('models/perception_net.pth')
PANEL_CSV      = Path('data/processed/panel_scores.csv')

# ── 配置 ──────────────────────────────────────────────────
ZOOM        = 2          # 2048×1024，8 张瓦片，约 1-2 MB/全景
TILE_W      = 512
TILE_H      = 512
TILE_COLS   = 2 ** ZOOM          # 4
TILE_ROWS   = 2 ** (ZOOM - 1)   # 2
PANO_W      = TILE_COLS * TILE_W  # 2048
PANO_H      = TILE_ROWS * TILE_H  # 1024

TILE_DELAY  = 0.06   # 瓦片下载间隔（秒）
SITE_DELAY  = 1.0    # 每个圣地处理完后等待（秒）
JPEG_Q      = 85

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ATTRS  = ['safety', 'lively', 'boring', 'wealthy', 'depressing', 'beautiful']
YAWS   = [0, 90, 180, 270]
FOV    = 90
CROP_W = CROP_H = 224
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

_TILE_URL = (
    "https://streetviewpixels-pa.googleapis.com/v1/tile"
    "?cb_client=maps_sv.tactile&panoid={panoid}"
    "&x={x}&y={y}&zoom={zoom}&nbt=1&fover=2"
)
_TILE_ALT = (
    "http://cbk0.google.com/cbk?output=tile"
    "&panoid={panoid}&zoom={zoom}&x={x}&y={y}"
)

PANEL_FIELDS = ['site_id', 'bangumi_id', 'title_cn', 'air_date',
                'date', 'panoid'] + ATTRS


# ── 模型定义 ──────────────────────────────────────────────
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
    if not MODEL_PATH.exists():
        log.warning(f'模型文件不存在：{MODEL_PATH}，跳过评分步骤')
        return None
    m = PerceptionNet().to(DEVICE)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    m.eval()
    log.info(f'模型加载完成  device={DEVICE}')
    return m


transform = T.Compose([
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float32),
    T.Normalize(IMG_MEAN, IMG_STD),
])


# ── 透视重投影 ─────────────────────────────────────────────
def equirect_to_persp(pano_arr, yaw_deg, pitch_deg=0):
    h, w = pano_arr.shape[:2]
    yaw   = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    f     = (CROP_W / 2) / math.tan(math.radians(FOV) / 2)

    xs = np.arange(CROP_W) - CROP_W / 2
    ys = np.arange(CROP_H) - CROP_H / 2
    xv, yv = np.meshgrid(xs, ys)
    zv   = np.full_like(xv, f, dtype=np.float32)
    norm = np.sqrt(xv**2 + yv**2 + zv**2)
    xn, yn, zn = xv/norm, -yv/norm, zv/norm

    cp, sp = math.cos(pitch), math.sin(pitch)
    yn2 = yn*cp - zn*sp
    zn2 = yn*sp + zn*cp

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
        map_coordinates(pano_arr[:, :, c], coords, order=1, mode='wrap')
        .reshape(CROP_H, CROP_W).astype(np.uint8)
        for c in range(3)
    ]
    return np.stack(channels, axis=2)


def score_pano(model, pano_path):
    """4 方向推理取均值，返回 6 维评分 ndarray。"""
    arr = np.array(Image.open(pano_path).convert('RGB'))
    crops = []
    for yaw in YAWS:
        crop = equirect_to_persp(arr, yaw)
        t    = transform(Image.fromarray(crop)).unsqueeze(0).to(DEVICE)
        crops.append(t)
    batch = torch.cat(crops, dim=0)
    with torch.no_grad():
        preds = model(batch)
    return preds.mean(dim=0).cpu().numpy()


# ── 瓦片下载 ──────────────────────────────────────────────
def download_tile(session, panoid, x, y, retries=3):
    for url_tmpl in [_TILE_URL, _TILE_ALT]:
        url = url_tmpl.format(panoid=panoid, x=x, y=y, zoom=ZOOM)
        for _ in range(retries):
            try:
                r = session.get(url, timeout=15)
                if r.status_code == 200 and len(r.content) > 500:
                    return Image.open(BytesIO(r.content)).convert('RGB')
            except Exception:
                time.sleep(0.5)
    return None


def build_panorama(session, panoid):
    """拼接 zoom=2 全景（2048×1024）。返回 (Image, ok_tiles)。"""
    pano = Image.new('RGB', (PANO_W, PANO_H))
    ok   = 0
    for x, y in itertools.product(range(TILE_COLS), range(TILE_ROWS)):
        tile = download_tile(session, panoid, x, y)
        if tile:
            pano.paste(tile, (x * TILE_W, y * TILE_H))
            ok += 1
        time.sleep(TILE_DELAY)
    return pano, ok


# ── 主流程 ────────────────────────────────────────────────
def main():
    SERIES_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_CSV.parent.mkdir(parents=True, exist_ok=True)

    # ── 读取合格圣地 ────────────────────────────────────────
    if not SITES_ELIGIBLE.exists():
        log.error(f'{SITES_ELIGIBLE} 不存在，请先运行 fetch_gsv_meta.py')
        return

    with open(SITES_ELIGIBLE, encoding='utf-8-sig') as f:
        sites = list(csv.DictReader(f))
    log.info(f'合格圣地：{len(sites)} 个')

    # ── 断点续跑：读已完成的 (site_id, date) ────────────────
    done = set()
    panel_mode = 'a' if PANEL_CSV.exists() else 'w'
    if PANEL_CSV.exists():
        with open(PANEL_CSV, encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                done.add((row['site_id'], row['date']))
        log.info(f'断点续跑：已评分 {len(done)} 条')

    # ── 加载模型 ────────────────────────────────────────────
    model = load_model()

    session = requests.Session()
    session.headers['User-Agent'] = 'Mozilla/5.0'

    panel_f = open(PANEL_CSV, panel_mode, newline='', encoding='utf-8-sig')
    panel_w = csv.DictWriter(panel_f, fieldnames=PANEL_FIELDS)
    if panel_mode == 'w':
        panel_w.writeheader()

    total_dl = total_score = 0

    try:
        for si, site in enumerate(sites):
            site_id    = site['site_id']
            bgm_id     = site['bangumi_id']
            title_cn   = site['title_cn']
            air_date   = site['air_date']
            timeline   = json.loads(site['timeline_json'])

            site_dir = SERIES_DIR / site_id
            site_dir.mkdir(exist_ok=True)

            log.info(f'[{si+1}/{len(sites)}] {title_cn} · {site["name"]}  '
                     f'({len(timeline)} 个快照)')

            for snap in timeline:
                yr, mo  = snap['year'], snap['month']
                date_str = f'{yr}-{mo:02d}'
                panoid   = snap['panoid']
                out_path = site_dir / f'{date_str}.jpg'

                key = (site_id, date_str)

                # ── 下载 ──────────────────────────────────
                if not out_path.exists():
                    pano, ok = build_panorama(session, panoid)
                    if ok < 4:   # 至少 4/8 张瓦片才算有效
                        log.warning(f'  [{date_str}] 瓦片不足 ({ok}/8)，跳过')
                        continue
                    pano.save(str(out_path), 'JPEG', quality=JPEG_Q)
                    sz = out_path.stat().st_size / 1024 / 1024
                    log.info(f'  [{date_str}] 下载完成 {sz:.1f}MB  panoid={panoid[:12]}…')
                    total_dl += 1
                else:
                    log.debug(f'  [{date_str}] 已存在，跳过下载')

                # ── 评分 ──────────────────────────────────
                if key in done or model is None:
                    continue

                try:
                    scores = score_pano(model, out_path)
                    row = {
                        'site_id':   site_id,
                        'bangumi_id': bgm_id,
                        'title_cn':   title_cn,
                        'air_date':   air_date,
                        'date':       date_str,
                        'panoid':     panoid,
                    }
                    for i, attr in enumerate(ATTRS):
                        row[attr] = round(float(scores[i]), 4)
                    panel_w.writerow(row)
                    panel_f.flush()
                    done.add(key)
                    total_score += 1
                    score_str = '  '.join(
                        f'{a}={row[a]:.2f}'
                        for a in ['lively', 'beautiful', 'safety']
                    )
                    log.info(f'  [{date_str}] 评分：{score_str}')
                except Exception as e:
                    log.warning(f'  [{date_str}] 评分失败: {e}')

            time.sleep(SITE_DELAY)

    finally:
        panel_f.close()

    log.info(f'\n{"="*60}')
    log.info(f'本次新下载：{total_dl} 张全景')
    log.info(f'本次新评分：{total_score} 条')
    log.info(f'面板数据 → {PANEL_CSV}  (累计 {len(done)} 条)')
    log.info(f'全景目录 → {SERIES_DIR}/')


if __name__ == '__main__':
    main()
