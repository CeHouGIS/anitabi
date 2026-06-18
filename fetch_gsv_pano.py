"""
下载 GSV 全景图（等矩形投影）。
zoom=3 → 8×4 瓦片 → 4096×2048 px，可直接喂给 Pannellum。

选取距动漫播出时间最近的全景版本，而非最新版本。
"""
import re, time, datetime, logging, itertools
from pathlib import Path
from io import BytesIO

import requests
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

SAVE_DIR  = Path('data/raw/streetview')
DOCS_DIR  = Path('docs/img/pano')
ZOOM      = 3        # 8×4 tiles → 4096×2048 px
TILE_W    = 512
TILE_H    = 512

# (site_id, label, lat, lon, target_year, target_month)
# target = 动漫首播年月，选取 GSV 中最接近该时间的全景
SAMPLE_SITES = [
    # 冰菓 — 2012-04 首播
    ('S_27364_0002', '冰菓·本町商店街',         36.1454, 137.2568, 2012, 4),
    ('S_27364_0004', '冰菓·鍛冶橋交差点',       36.1429, 137.2575, 2012, 4),
    ('S_27364_0026', '冰菓·宮川朝市',           36.1450, 137.2579, 2012, 4),
    # 未闻花名 — 2011-04 首播
    ('S_10440_0030', '未闻花名·旧秩父橋',       36.0187, 139.0863, 2011, 4),
    ('S_10440_0051', '未闻花名·秩父神社',       35.9966, 139.0841, 2011, 4),
    # 摇曳露营△ — 2018-01 首播
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


def get_panoid(session, lat, lon, target_year=None, target_month=None):
    """选取距 target_year/month 最近的全景 panoid；若无目标日期则取最新。"""
    url = _PANOID_URL.format(lat=lat, lon=lon)
    try:
        r = session.get(url, timeout=12)
        text = r.text
        pans = re.findall(
            r'\[[0-9]+,"(.+?)"\].+?\[\[null,null,(-?[0-9]+\.[0-9]+),(-?[0-9]+\.[0-9]+)',
            text
        )
        if not pans:
            return None, None
        pans = [{'panoid': p[0], 'lat': float(p[1]), 'lon': float(p[2])} for p in pans]
        if len(pans) > 1 and pans[0]['panoid'] == pans[1]['panoid']:
            pans = pans[1:]

        dates = re.findall(r'([0-9]{0,3})?,?\[(20[0-9]{2}),([0-9]{1,2})\]', text)
        dates = [d for d in dates if 1 <= int(d[2]) <= 12]
        if dates:
            for i, (_, year, month) in enumerate(dates):
                if i < len(pans):
                    pans[i]['year'] = int(year)
                    pans[i]['month'] = int(month)
            dated = [p for p in pans if 'year' in p]
            if dated:
                log.info(f"  可用全景版本：" +
                         ", ".join(f"{p['year']}-{p['month']:02d}" for p in dated))
                if target_year and target_month:
                    target_dt = datetime.datetime(target_year, target_month, 1)
                    dated.sort(key=lambda p: abs(
                        (datetime.datetime(p['year'], p['month'], 1) - target_dt).days
                    ))
                    chosen = dated[0]
                    log.info(f"  目标: {target_year}-{target_month:02d}  "
                             f"→ 选取: {chosen['year']}-{chosen['month']:02d}  panoid={chosen['panoid']}")
                    return chosen['panoid'], f"{chosen['year']}-{chosen['month']:02d}"
                else:
                    dated.sort(key=lambda p: datetime.datetime(p['year'], p['month'], 1))
                    chosen = dated[-1]
                    return chosen['panoid'], f"{chosen['year']}-{chosen['month']:02d}"
        log.warning("  未能解析全景日期，使用第一个候选")
        return pans[0]['panoid'], None
    except Exception as e:
        log.warning(f"  panoid error: {e}")
        return None, None


def download_tile(session, panoid, x, y, zoom, retries=3):
    for url_tmpl in [_TILE_URL, _TILE_URL_ALT]:
        url = url_tmpl.format(panoid=panoid, x=x, y=y, zoom=zoom)
        for attempt in range(retries):
            try:
                r = session.get(url, timeout=15)
                if r.status_code == 200 and len(r.content) > 1000:
                    return Image.open(BytesIO(r.content)).convert('RGB')
            except Exception:
                time.sleep(1)
    return None


def build_panorama(session, panoid, zoom=ZOOM):
    cols = 2 ** zoom
    rows = 2 ** (zoom - 1)
    pano = Image.new('RGB', (cols * TILE_W, rows * TILE_H))
    total = cols * rows
    ok = 0
    for x, y in itertools.product(range(cols), range(rows)):
        tile = download_tile(session, panoid, x, y, zoom)
        if tile:
            pano.paste(tile, (x * TILE_W, y * TILE_H))
            ok += 1
        time.sleep(0.05)
    log.info(f"    瓦片: {ok}/{total}")
    return pano


def main():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers['User-Agent'] = 'Mozilla/5.0'

    results = {}

    for row in SAMPLE_SITES:
        site_id, label, lat, lon = row[0], row[1], row[2], row[3]
        target_year  = row[4] if len(row) > 4 else None
        target_month = row[5] if len(row) > 5 else None

        log.info(f"{'─'*58}")
        log.info(f"{label}  ({lat:.4f}, {lon:.4f})  目标年月: {target_year}-{target_month:02d}" if target_year else
                 f"{label}  ({lat:.4f}, {lon:.4f})")

        out_path = DOCS_DIR / f"{site_id}.jpg"

        panoid, chosen_date = get_panoid(session, lat, lon, target_year, target_month)
        if not panoid:
            log.warning("  未找到 panoid，跳过")
            continue

        pano = build_panorama(session, panoid, zoom=ZOOM)
        pano.save(str(out_path), 'JPEG', quality=88)
        date_str = f"  [{chosen_date}]" if chosen_date else ""
        log.info(f"  → 保存 {pano.width}×{pano.height}px{date_str}  {out_path}")
        results[site_id] = f"img/pano/{site_id}.jpg"
        time.sleep(1.5)

    print("\n// GSV 全景路径映射（粘贴到 index.html）:")
    print("const PANO_IMGS = {")
    for sid, path in results.items():
        print(f'  "{sid}": "{path}",')
    print("};")


if __name__ == '__main__':
    main()
