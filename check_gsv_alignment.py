"""
核查 GSV 全景与动漫圣地标注的空间/时间一致性。
输出每个圣地的：
  - anitabi 标注坐标
  - GSV 实际拍摄坐标
  - 空间偏差（米）
  - GSV 拍摄年月 vs 动漫首播年月（月差）
"""
import re, math, datetime, logging
from pathlib import Path
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# (site_id, label, anitabi_lat, anitabi_lon, broadcast_year, broadcast_month)
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


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def query_gsv(session, lat, lon, target_year, target_month):
    url = _PANOID_URL.format(lat=lat, lon=lon)
    r = session.get(url, timeout=12)
    text = r.text

    pans = re.findall(
        r'\[[0-9]+,"(.+?)"\].+?\[\[null,null,(-?[0-9]+\.[0-9]+),(-?[0-9]+\.[0-9]+)',
        text
    )
    if not pans:
        return None
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
            target_dt = datetime.datetime(target_year, target_month, 1)
            dated.sort(key=lambda p: abs(
                (datetime.datetime(p['year'], p['month'], 1) - target_dt).days
            ))
            return dated[0]   # closest to broadcast date
    return pans[0]


def month_diff(y1, m1, y2, m2):
    return (y2 - y1) * 12 + (m2 - m1)


def main():
    session = requests.Session()
    session.headers['User-Agent'] = 'Mozilla/5.0'

    rows = []
    for sid, label, a_lat, a_lon, bc_y, bc_m in SAMPLE_SITES:
        log.info(f"查询 {label}")
        info = query_gsv(session, a_lat, a_lon, bc_y, bc_m)
        if not info:
            log.warning(f"  未找到 GSV 数据")
            continue

        gsv_lat = info['lat']
        gsv_lon = info['lon']
        dist_m  = haversine_m(a_lat, a_lon, gsv_lat, gsv_lon)
        gsv_y   = info.get('year')
        gsv_m   = info.get('month')
        if gsv_y:
            mdiff = month_diff(bc_y, bc_m, gsv_y, gsv_m)
            time_str = f"{gsv_y}-{gsv_m:02d}  (Δ{mdiff:+d}月)"
        else:
            time_str = "日期未知"

        rows.append((sid, label, a_lat, a_lon, gsv_lat, gsv_lon,
                     dist_m, f"{bc_y}-{bc_m:02d}", time_str, info['panoid']))

    # ── 打印表格 ────────────────────────────────────────────
    print()
    print("=" * 110)
    print(f"{'圣地':<24} {'标注坐标':>22} {'GSV坐标':>22} {'空间偏差':>9} {'播出年月':>9} {'GSV年月(月差)':>20}")
    print("=" * 110)
    for r in rows:
        sid, label, a_lat, a_lon, g_lat, g_lon, dist_m, bc_str, time_str, panoid = r
        coord_a = f"({a_lat:.4f},{a_lon:.4f})"
        coord_g = f"({g_lat:.4f},{g_lon:.4f})"
        dist_fmt = f"{dist_m:.0f}m" if dist_m < 1000 else f"{dist_m/1000:.2f}km"
        flag = "⚠" if dist_m > 200 else "✓"
        print(f"{label:<24} {coord_a:>22} {coord_g:>22} {dist_fmt:>8} {flag}  {bc_str:>9}  {time_str}")
    print("=" * 110)
    print()
    print("说明：")
    print("  空间偏差 < 50m  → GSV 拍摄点与圣地标注高度吻合")
    print("  空间偏差 50-200m → 拍摄点在圣地附近（道路宽度/停车场偏移）")
    print("  空间偏差 > 200m → ⚠ 需人工复核标注精度")
    print("  月差 = GSV拍摄年月 − 动漫首播年月（负=播出前，正=播出后）")

if __name__ == '__main__':
    main()
