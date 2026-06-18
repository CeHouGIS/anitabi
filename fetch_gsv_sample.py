"""
快速下载样本街景图片 — 每部动画选2个代表性圣地，各下载0°/90°两个方向。
"""
import sys, time, re, datetime, logging
from pathlib import Path
from io import BytesIO

import requests
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

SAVE_DIR = Path('data/raw/streetview')

# 代表性圣地：(site_id, 地点名, lat, lon)
SAMPLE_SITES = [
    # 冰菓 — 高山市
    ('S_27364_0002', '冰菓·本町商店街',         36.1454, 137.2568),
    ('S_27364_0004', '冰菓·鍛冶橋',             36.1429, 137.2575),
    ('S_27364_0026', '冰菓·宮川朝市',           36.1450, 137.2579),
    # 未闻花名 — 秩父市
    ('S_10440_0030', '未闻花名·旧秩父橋',       36.0187, 139.0863),
    ('S_10440_0051', '未闻花名·秩父神社',       35.9966, 139.0841),
    # 摇曳露营△ — 山梨/长野
    ('S_207195_0069', '摇曳露营·ローソン南アルプス', 35.6610, 138.4404),
    ('S_207195_0075', '摇曳露营·夜叉神峠登山口', 35.6356, 138.3454),
]

HEADINGS = [0, 90, 180, 270]
W, H = 640, 360

_PANOID_URL = (
    "https://maps.googleapis.com/maps/api/js/GeoPhotoService.SingleImageSearch"
    "?pb=!1m5!1sapiv3!5sUS!11m2!1m1!1b0!2m4!1m2!3d{lat}!4d{lon}"
    "!2d50!3m10!2m2!1sen!2sGB!9m1!1e2!11m4!1m3!1e2!2b1!3e2"
    "!4m10!1e1!1e2!1e3!1e4!1e8!1e6!5m1!1e2!6m1!1e2&callback=_xdc_._v2mub5"
)
_THUMB_URL = (
    "https://geo0.ggpht.com/cbk"
    "?cb_client=maps_sv.tactile&authuser=0&hl=en&output=thumbnail&nbt"
    "&w={w}&h={h}&yaw={heading}&panoid={panoid}&thumbfov=90&pitch=0"
)


def get_panoid(session, lat, lon):
    url = _PANOID_URL.format(lat=lat, lon=lon)
    try:
        r = session.get(url, timeout=12)
        if r.status_code != 200:
            return None
        text = r.text
        pans = re.findall(
            r'\[[0-9]+,"(.+?)"\].+?\[\[null,null,(-?[0-9]+\.[0-9]+),(-?[0-9]+\.[0-9]+)',
            text
        )
        if not pans:
            return None
        pans = [{'panoid': p[0], 'lat': float(p[1]), 'lon': float(p[2])} for p in pans]
        if len(pans) > 1 and pans[0] == pans[1]:
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
                dated.sort(key=lambda p: datetime.datetime(p['year'], p['month'], 1))
                return dated[-1]['panoid']
        return pans[0]['panoid']
    except Exception as e:
        log.warning(f"  panoid error: {e}")
        return None


def download_heading(session, panoid, heading, save_path):
    url = _THUMB_URL.format(panoid=panoid, heading=heading, w=W, h=H)
    try:
        r = session.get(url, timeout=15)
        if r.status_code == 200 and len(r.content) > 5000:
            Image.open(BytesIO(r.content)).save(str(save_path), 'JPEG')
            return True
    except Exception as e:
        log.debug(f"  thumb error: {e}")
    return False


def main():
    session = requests.Session()
    session.headers['User-Agent'] = 'Mozilla/5.0'

    for site_id, label, lat, lon in SAMPLE_SITES:
        log.info(f"{'─'*50}")
        log.info(f"{label}  ({lat:.4f}, {lon:.4f})")
        panoid = get_panoid(session, lat, lon)
        if not panoid:
            log.warning(f"  → 未找到 panoid，跳过")
            continue
        log.info(f"  panoid: {panoid}")

        site_dir = SAVE_DIR / site_id
        site_dir.mkdir(parents=True, exist_ok=True)

        ok = 0
        for h in HEADINGS:
            fp = site_dir / f"{h:03d}.jpg"
            if download_heading(session, panoid, h, fp):
                ok += 1
            time.sleep(0.4)
        log.info(f"  → {ok}/{len(HEADINGS)} 张下载成功  ({site_dir})")
        time.sleep(1.0)

    log.info("完成")


if __name__ == '__main__':
    main()
