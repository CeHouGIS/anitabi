"""
爬取各圣地所有可用历史街景全景图。
若同一季度内有多张，只保留最新一张（季度去重）。
保存路径：data/raw/streetview/history/{site_id}/{YYYY_MM}.jpg
同时输出元数据：data/processed/gsv_timeline.csv
"""
import re, time, datetime, logging, itertools, csv
from pathlib import Path
from io import BytesIO

import requests
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

SAVE_DIR = Path('data/raw/streetview/history')
META_CSV = Path('data/processed/gsv_timeline.csv')
ZOOM     = 3
TILE_W   = 512
TILE_H   = 512

SAMPLE_SITES = [
    ('S_27364_0002', '冰菓·本町商店街',         36.1454, 137.2568, 2012, 4),
    ('S_27364_0004', '冰菓·鍛冶橋交差点',       36.1429, 137.2575, 2012, 4),
    ('S_27364_0026', '冰菓·宮川朝市',           36.1450, 137.2579, 2012, 4),
    ('S_10440_0030', '未闻花名·旧秩父橋',       36.0187, 139.0863, 2011, 4),
    ('S_10440_0051', '未闻花名·秩父神社',       35.9966, 139.0841, 2011, 4),
    ('S_207195_0069','摇曳露营·ローソン南アルプス',35.6610,138.4404, 2018, 1),
    ('S_207195_0075','摇曳露营·夜叉神峠登山口', 35.6356, 138.3454, 2018, 1),
]

_PANOID_URL = (
    "https://maps.googleapis.com/maps/api/js/GeoPhotoService.SingleImageSearch"
    "?pb=!1m5!1sapiv3!5sUS!11m2!1m1!1b0!2m4!1m2!3d{lat}!4d{lon}"
    "!2d50!3m10!2m2!1sen!2sGB!9m1!1e2!11m4!1m3!1e2!2b1!3e2"
    "!4m10!1e1!1e2!1e3!1e4!1e8!1e6!5m1!1e2!6m1!1e2&callback=_xdc_._v2mub5"
)
_TILE_URL = (
    "https://streetviewpixels-pa.googleapis.com/v1/tile"
    "?cb_client=maps_sv.tactile&panoid={panoid}&x={x}&y={y}&zoom={zoom}&nbt=1&fover=2"
)
_TILE_URL_ALT = "http://cbk0.google.com/cbk?output=tile&panoid={panoid}&zoom={zoom}&x={x}&y={y}"


def quarter(year, month):
    """(year, month) → 季度标识符，用于去重。"""
    return (year, (month - 1) // 3)


def get_all_panoids(session, lat, lon):
    """返回该坐标所有历史全景，经季度去重后的列表（每季度保留最新）。"""
    url = _PANOID_URL.format(lat=lat, lon=lon)
    r = session.get(url, timeout=12)
    text = r.text

    pans = re.findall(
        r'\[[0-9]+,"(.+?)"\].+?\[\[null,null,(-?[0-9]+\.[0-9]+),(-?[0-9]+\.[0-9]+)',
        text
    )
    if not pans:
        return []
    pans = [{'panoid': p[0], 'lat': float(p[1]), 'lon': float(p[2])} for p in pans]
    if len(pans) > 1 and pans[0]['panoid'] == pans[1]['panoid']:
        pans = pans[1:]

    dates = re.findall(r'([0-9]{0,3})?,?\[(20[0-9]{2}),([0-9]{1,2})\]', text)
    dates = [d for d in dates if 1 <= int(d[2]) <= 12]
    for i, (_, year, month) in enumerate(dates):
        if i < len(pans):
            pans[i]['year']  = int(year)
            pans[i]['month'] = int(month)

    dated = [p for p in pans if 'year' in p]
    if not dated:
        # 无日期信息，只能用第一个
        return [pans[0]] if pans else []

    # 季度去重：每季度保留最新（year+month 最大）
    by_quarter = {}
    for p in dated:
        q = quarter(p['year'], p['month'])
        if q not in by_quarter:
            by_quarter[q] = p
        else:
            cur = by_quarter[q]
            if (p['year'], p['month']) > (cur['year'], cur['month']):
                by_quarter[q] = p

    selected = sorted(by_quarter.values(), key=lambda p: (p['year'], p['month']))
    return selected


def download_tile(session, panoid, x, y, zoom, retries=3):
    for url_tmpl in [_TILE_URL, _TILE_URL_ALT]:
        url = url_tmpl.format(panoid=panoid, x=x, y=y, zoom=zoom)
        for _ in range(retries):
            try:
                r = session.get(url, timeout=15)
                if r.status_code == 200 and len(r.content) > 1000:
                    return Image.open(BytesIO(r.content)).convert('RGB')
            except Exception:
                time.sleep(1)
    return None


def build_panorama(session, panoid):
    cols = 2 ** ZOOM
    rows = 2 ** (ZOOM - 1)
    pano = Image.new('RGB', (cols * TILE_W, rows * TILE_H))
    ok = 0
    for x, y in itertools.product(range(cols), range(rows)):
        tile = download_tile(session, panoid, x, y, ZOOM)
        if tile:
            pano.paste(tile, (x * TILE_W, y * TILE_H))
            ok += 1
        time.sleep(0.05)
    log.info(f"      瓦片: {ok}/{cols*rows}")
    return pano, ok


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    META_CSV.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers['User-Agent'] = 'Mozilla/5.0'

    all_meta = []
    total_downloaded = 0

    for site_id, label, lat, lon, bc_y, bc_m in SAMPLE_SITES:
        log.info(f"{'─'*60}")
        log.info(f"{label}  播出: {bc_y}-{bc_m:02d}")

        site_dir = SAVE_DIR / site_id
        site_dir.mkdir(exist_ok=True)

        versions = get_all_panoids(session, lat, lon)
        if not versions:
            log.warning("  未找到任何全景版本")
            continue

        log.info(f"  季度去重后共 {len(versions)} 个版本：" +
                 ", ".join(f"{p['year']}-{p['month']:02d}" for p in versions))

        for p in versions:
            yr, mo = p['year'], p['month']
            date_str = f"{yr}-{mo:02d}"
            out_path = site_dir / f"{date_str}.jpg"

            if out_path.exists():
                log.info(f"  [{date_str}] 已存在，跳过")
                all_meta.append({
                    'site_id': site_id, 'label': label,
                    'year': yr, 'month': mo, 'date': date_str,
                    'panoid': p['panoid'],
                    'gsv_lat': p['lat'], 'gsv_lon': p['lon'],
                    'broadcast_year': bc_y, 'broadcast_month': bc_m,
                    'months_from_broadcast': (yr - bc_y)*12 + (mo - bc_m),
                    'file': str(out_path),
                })
                continue

            log.info(f"  [{date_str}] 下载中…  panoid={p['panoid']}")
            pano, tiles_ok = build_panorama(session, p['panoid'])
            pano.save(str(out_path), 'JPEG', quality=85)
            sz_mb = out_path.stat().st_size / 1024 / 1024
            log.info(f"      → {pano.width}×{pano.height}px  {sz_mb:.1f}MB  {out_path.name}")

            all_meta.append({
                'site_id': site_id, 'label': label,
                'year': yr, 'month': mo, 'date': date_str,
                'panoid': p['panoid'],
                'gsv_lat': p['lat'], 'gsv_lon': p['lon'],
                'broadcast_year': bc_y, 'broadcast_month': bc_m,
                'months_from_broadcast': (yr - bc_y)*12 + (mo - bc_m),
                'file': str(out_path),
            })
            total_downloaded += 1
            time.sleep(1.5)

    # ── 写元数据 CSV ──────────────────────────────────────
    if all_meta:
        keys = list(all_meta[0].keys())
        with open(META_CSV, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_meta)
        log.info(f"\n元数据保存 → {META_CSV}  ({len(all_meta)} 条记录)")

    # ── 汇总表 ───────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"{'圣地':<26} {'版本数':>5}  时间线")
    print("=" * 70)
    from itertools import groupby
    for site_id, label, *_ in SAMPLE_SITES:
        rows = [m for m in all_meta if m['site_id'] == site_id]
        dates = [m['date'] for m in rows]
        print(f"{label:<26} {len(dates):>5}  {'  →  '.join(dates)}")
    print("=" * 70)
    print(f"\n本次新下载：{total_downloaded} 张")
    print(f"历史全景目录：{SAVE_DIR}/")
    print(f"元数据表：    {META_CSV}")


if __name__ == '__main__':
    main()
