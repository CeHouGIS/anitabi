"""
下载各圣地当前最新的 GSV 全景图（After 快照）。
与 docs/img/pano/ 里播出年月版本（Before）配对，用于 DiD 分析。
保存至 docs/img/pano_now/{site_id}.jpg
"""
import re, time, datetime, logging, itertools
from pathlib import Path
from io import BytesIO

import requests
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

DOCS_DIR = Path('docs/img/pano_now')
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


def get_newest_panoid(session, lat, lon):
    """取该坐标最新的全景，返回 (panoid, date_str, gsv_lat, gsv_lon)。"""
    url = _PANOID_URL.format(lat=lat, lon=lon)
    try:
        r = session.get(url, timeout=12)
        text = r.text
        pans = re.findall(
            r'\[[0-9]+,"(.+?)"\].+?\[\[null,null,(-?[0-9]+\.[0-9]+),(-?[0-9]+\.[0-9]+)',
            text
        )
        if not pans:
            return None, None, None, None
        pans = [{'panoid': p[0], 'lat': float(p[1]), 'lon': float(p[2])} for p in pans]
        if len(pans) > 1 and pans[0]['panoid'] == pans[1]['panoid']:
            pans = pans[1:]

        dates = re.findall(r'([0-9]{0,3})?,?\[(20[0-9]{2}),([0-9]{1,2})\]', text)
        dates = [d for d in dates if 1 <= int(d[2]) <= 12]

        all_versions = []
        for i, (_, year, month) in enumerate(dates):
            if i < len(pans):
                pans[i]['year']  = int(year)
                pans[i]['month'] = int(month)
        dated = [p for p in pans if 'year' in p]

        log.info(f"  可用版本：" + ", ".join(f"{p['year']}-{p['month']:02d}" for p in dated))

        if dated:
            # 取最新
            dated.sort(key=lambda p: (p['year'], p['month']), reverse=True)
            best = dated[0]
            date_str = f"{best['year']}-{best['month']:02d}"
            return best['panoid'], date_str, best['lat'], best['lon']
        return pans[0]['panoid'], None, pans[0]['lat'], pans[0]['lon']
    except Exception as e:
        log.warning(f"  panoid error: {e}")
        return None, None, None, None


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
    log.info(f"    瓦片: {ok}/{cols*rows}")
    return pano


def main():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    before_dir = Path('docs/img/pano')

    session = requests.Session()
    session.headers['User-Agent'] = 'Mozilla/5.0'

    results = []

    for site_id, label, lat, lon, bc_y, bc_m in SAMPLE_SITES:
        log.info(f"{'─'*58}")
        log.info(f"{label}  播出: {bc_y}-{bc_m:02d}")

        out_path = DOCS_DIR / f"{site_id}.jpg"

        panoid, date_str, gsv_lat, gsv_lon = get_newest_panoid(session, lat, lon)
        if not panoid:
            log.warning("  未找到 panoid，跳过")
            results.append((site_id, label, bc_y, bc_m, None, None))
            continue

        log.info(f"  → 选取最新版本: {date_str}  panoid={panoid}")

        pano = build_panorama(session, panoid)
        pano.save(str(out_path), 'JPEG', quality=88)
        log.info(f"  → 保存 {pano.width}×{pano.height}px  {out_path}")
        results.append((site_id, label, bc_y, bc_m, date_str, out_path))
        time.sleep(1.5)

    # ── 汇总 ──────────────────────────────────────────────
    print()
    print("=" * 75)
    print(f"{'圣地':<26} {'播出':<9} {'Before版本':<12} {'After(当前)':>12}  {'年差':>5}")
    print("=" * 75)
    before_dates = {
        'S_27364_0002': '2012-04', 'S_27364_0004': '2012-03',
        'S_27364_0026': '2012-04', 'S_10440_0030': '2011-05',
        'S_10440_0051': '2011-05', 'S_207195_0069': '2018-08',
        'S_207195_0075': '2023-04',
    }
    for sid, label, bc_y, bc_m, after_date, _ in results:
        before = before_dates.get(sid, '—')
        if after_date and before != '—':
            by = int(before[:4]); ay = int(after_date[:4])
            gap = f"{ay-by}年"
        else:
            gap = '—'
        print(f"{label:<26} {bc_y}-{bc_m:02d}  {before:<12} {after_date or '失败':>12}  {gap:>5}")
    print("=" * 75)
    print(f"\nAfter 全景保存在：{DOCS_DIR}/")
    print(f"Before 全景在：  {before_dir}/")


if __name__ == '__main__':
    main()
