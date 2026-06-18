"""
Phase 2 · 圣地 GSV 时序元数据抓取 + 动漫帧图下载

输入：data/raw/sites_raw.csv（discover_anime.py 输出，含全年代）

流程：
  1. 跳过 GSV_FROM_YEAR 之前的圣地（只对2016+做DiD分析）
  2. 对每个圣地坐标调用 GeoPhotoService → 获取所有历史 panoid 及日期
  3. 季度去重（每季度只保留最新一张）
  4. DiD 可用性过滤：播出前 ≥1 张 AND 播出后 ≥1 张
  5. 下载动漫帧缩略图 → docs/img/anime_full/{site_id}.jpg
  6. 断点续跑（已处理的 site_id 跳过）

输出：
  data/raw/sites_eligible.csv  ← 合格圣地 + 完整 panoid 时序（JSON 列）
  docs/img/anime_full/         ← 动漫帧图（仅2016+）
"""

import csv, re, time, json, logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image
from io import BytesIO

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# ── 路径 ──────────────────────────────────────────────────
SITES_RAW  = Path('data/raw/sites_raw.csv')
SITES_OUT  = Path('data/raw/sites_eligible.csv')
ANIME_IMG  = Path('docs/img/anime_full')

# ── 配置 ──────────────────────────────────────────────────
GSV_DELAY       = 0.4    # GeoPhotoService 请求间隔（秒）
IMG_DOWNLOAD_W  = 6      # 图片下载并发线程数
IMG_SIZE        = (640, 360)   # 动漫帧保存尺寸
GSV_FROM_YEAR   = 2016   # 只对此年及之后播出的动漫做 GSV 时序查询

OUT_FIELDS = [
    'site_id', 'bangumi_id', 'title_cn', 'air_date',
    'point_id', 'name', 'lat', 'lon',
    'n_snapshots', 'n_pre', 'n_post',
    'earliest', 'latest',
    'timeline_json',   # [{panoid, year, month, lat, lon}, ...]
    'anime_img_local',
]

# ── GeoPhotoService ────────────────────────────────────────
_GEO_URL = (
    "https://maps.googleapis.com/maps/api/js/GeoPhotoService.SingleImageSearch"
    "?pb=!1m5!1sapiv3!5sUS!11m2!1m1!1b0!2m4!1m2!3d{lat}!4d{lon}"
    "!2d50!3m10!2m2!1sen!2sGB!9m1!1e2!11m4!1m3!1e2!2b1!3e2"
    "!4m10!1e1!1e2!1e3!1e4!1e8!1e6!5m1!1e2!6m1!1e2&callback=_xdc_._v2mub5"
)


def quarter(year, month):
    return (year, (month - 1) // 3)


def get_gsv_timeline(session, lat, lon):
    """
    返回季度去重后的 panoid 列表，每项含 year/month/panoid/lat/lon。
    """
    url = _GEO_URL.format(lat=lat, lon=lon)
    try:
        r = session.get(url, timeout=12)
        text = r.text
    except Exception as e:
        log.debug(f'GeoPhotoService 失败 ({lat},{lon}): {e}')
        return []

    # 解析 panoid + 坐标
    pans = re.findall(
        r'\[[0-9]+,"(.+?)"\].+?\[\[null,null,(-?[0-9]+\.[0-9]+),(-?[0-9]+\.[0-9]+)',
        text,
    )
    if not pans:
        return []
    pans = [{'panoid': p[0], 'lat': float(p[1]), 'lon': float(p[2])}
            for p in pans]
    if len(pans) > 1 and pans[0]['panoid'] == pans[1]['panoid']:
        pans = pans[1:]

    # 解析日期
    dates = re.findall(r'([0-9]{0,3})?,?\[(20[0-9]{2}),([0-9]{1,2})\]', text)
    dates = [d for d in dates if 1 <= int(d[2]) <= 12]
    for i, (_, year, month) in enumerate(dates):
        if i < len(pans):
            pans[i]['year']  = int(year)
            pans[i]['month'] = int(month)

    dated = [p for p in pans if 'year' in p]
    if not dated:
        return []

    # 季度去重：每季度保留最新
    by_q = {}
    for p in dated:
        q = quarter(p['year'], p['month'])
        if q not in by_q:
            by_q[q] = p
        else:
            cur = by_q[q]
            if (p['year'], p['month']) > (cur['year'], cur['month']):
                by_q[q] = p

    return sorted(by_q.values(), key=lambda p: (p['year'], p['month']))


def download_anime_img(session, image_url, out_path):
    """下载动漫帧图（取高清版，去掉 ?plan= 参数）。"""
    if out_path.exists():
        return True
    clean_url = image_url.split('?')[0]
    try:
        r = session.get(clean_url, timeout=15)
        if r.status_code != 200 or len(r.content) < 500:
            return False
        img = Image.open(BytesIO(r.content)).convert('RGB')
        img = img.resize(IMG_SIZE, Image.LANCZOS)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(out_path), 'JPEG', quality=88)
        return True
    except Exception as e:
        log.debug(f'图片下载失败 {clean_url}: {e}')
        return False


def main():
    ANIME_IMG.mkdir(parents=True, exist_ok=True)
    SITES_OUT.parent.mkdir(parents=True, exist_ok=True)

    # ── 读取输入 ────────────────────────────────────────────
    if not SITES_RAW.exists():
        log.error(f'{SITES_RAW} 不存在，请先运行 discover_anime.py')
        return

    with open(SITES_RAW, encoding='utf-8-sig') as f:
        all_sites = list(csv.DictReader(f))
    log.info(f'读取 {len(all_sites)} 个圣地坐标')

    # ── 断点续跑：跳过已处理的 site_id ─────────────────────
    done_ids = set()
    if SITES_OUT.exists():
        with open(SITES_OUT, encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                done_ids.add(row['site_id'])
        log.info(f'断点续跑：已处理 {len(done_ids)} 个')

    pending = [s for s in all_sites if s['site_id'] not in done_ids]
    log.info(f'待处理 {len(pending)} 个')

    # ── Phase A：GSV 元数据查询 ─────────────────────────────
    session = requests.Session()
    session.headers['User-Agent'] = 'Mozilla/5.0'

    out_mode = 'a' if SITES_OUT.exists() else 'w'
    out_f    = open(SITES_OUT, out_mode, newline='', encoding='utf-8-sig')
    out_w    = csv.DictWriter(out_f, fieldnames=OUT_FIELDS)
    if out_mode == 'w':
        out_w.writeheader()

    # 收集需要下载图片的 (site_id, image_url, out_path)
    img_tasks = []
    n_eligible = len(done_ids)

    try:
        for i, site in enumerate(pending):
            site_id  = site['site_id']
            air_date = site['air_date']      # 'YYYY-MM-DD'
            lat      = float(site['lat'])
            lon      = float(site['lon'])
            image_url = site.get('image_url', '')

            # 播出年月
            try:
                bc_y = int(air_date[:4])
                bc_m = int(air_date[5:7])
            except Exception:
                log.debug(f'{site_id}: 无法解析播出日期 {air_date}，跳过')
                continue

            # 只对 GSV_FROM_YEAR 及之后的动漫做 GSV 时序分析
            if bc_y < GSV_FROM_YEAR:
                done_ids.add(site_id)   # 标记已处理（跳过），避免重复日志
                continue

            if (i + 1) % 100 == 0:
                log.info(f'进度 {i+1}/{len(pending)}  合格率 '
                         f'{n_eligible}/{i+1}')

            timeline = get_gsv_timeline(session, lat, lon)
            time.sleep(GSV_DELAY)

            if not timeline:
                continue

            # 统计播出前/后
            pre  = [p for p in timeline
                    if (p['year'], p['month']) < (bc_y, bc_m)]
            post = [p for p in timeline
                    if (p['year'], p['month']) >= (bc_y, bc_m)]

            # DiD 条件：播出前后各至少 1 张
            if not pre or not post:
                continue

            earliest = f"{timeline[0]['year']}-{timeline[0]['month']:02d}"
            latest   = f"{timeline[-1]['year']}-{timeline[-1]['month']:02d}"

            # 确定动漫帧图本地路径
            img_local = ''
            if image_url:
                img_path  = ANIME_IMG / f'{site_id}.jpg'
                img_local = str(img_path)
                img_tasks.append((site_id, image_url, img_path))

            out_w.writerow({
                'site_id':       site_id,
                'bangumi_id':    site['bangumi_id'],
                'title_cn':      site['title_cn'],
                'air_date':      air_date,
                'point_id':      site['point_id'],
                'name':          site['name'],
                'lat':           lat,
                'lon':           lon,
                'n_snapshots':   len(timeline),
                'n_pre':         len(pre),
                'n_post':        len(post),
                'earliest':      earliest,
                'latest':        latest,
                'timeline_json': json.dumps(timeline, ensure_ascii=False),
                'anime_img_local': img_local,
            })
            out_f.flush()
            n_eligible += 1

    finally:
        out_f.close()

    log.info(f'\nGSV 元数据完成：{n_eligible} 个合格圣地（含已有记录）')

    # ── Phase B：并发下载动漫帧图 ─────────────────────────
    log.info(f'\n开始下载 {len(img_tasks)} 张动漫帧图（{IMG_DOWNLOAD_W} 线程）…')

    img_session = requests.Session()
    img_session.headers['User-Agent'] = 'Mozilla/5.0'

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=IMG_DOWNLOAD_W) as pool:
        futures = {
            pool.submit(download_anime_img, img_session, url, path): sid
            for sid, url, path in img_tasks
            if not path.exists()   # 跳过已下载
        }
        for fut in as_completed(futures):
            sid = futures[fut]
            try:
                if fut.result():
                    ok += 1
                else:
                    fail += 1
            except Exception:
                fail += 1
            if (ok + fail) % 100 == 0:
                log.info(f'  图片进度 {ok+fail}/{len(futures)}  '
                         f'成功 {ok}  失败 {fail}')

    log.info(f'\n{"="*60}')
    log.info(f'合格圣地 → {SITES_OUT}  ({n_eligible} 条)')
    log.info(f'动漫帧图 → {ANIME_IMG}/  (成功 {ok}, 失败 {fail})')


if __name__ == '__main__':
    main()
