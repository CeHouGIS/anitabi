"""
Phase 1 · 发现 1990-2024 年在 anitabi 有数据的日本动漫（全年代普查）

流程：
  Bangumi v0 Search API (按季度分页，避免 1000 条上限)
  → 逐 ID 检查 anitabi /lite
  → 过滤：日本圣地数 ≥ MIN_JAPAN_SITES
  → 断点续跑（已记录的 bangumi_id 跳过）

输出：
  data/raw/anime_list.csv   ← 合格动漫元数据（全年代）
  data/raw/sites_raw.csv    ← 所有圣地坐标（全年代，不含图片）

注：fetch_gsv_meta.py 只对 2016+ 动漫做 GSV 时序分析（GSV_FROM_YEAR 配置）
"""

import csv, time, logging
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# ── 配置 ────────────────────────────────────────────────
YEARS           = range(1990, 2025)   # 全年代普查；2016+ 圣地将进入 GSV 管线
MIN_JAPAN_SITES = 5
JAPAN_LAT       = (24.0, 46.0)
JAPAN_LON       = (122.0, 154.0)
# anitabi 有 Cloudflare 保护，必须串行 + 长间隔，禁止并发
ANITABI_DELAY   = 6.0    # anitabi 请求间隔（秒）——低于此值易触发 CF 封禁
BGM_DELAY       = 0.3    # Bangumi 请求间隔（秒）

ANIME_CSV = Path('data/raw/anime_list.csv')
SITES_CSV = Path('data/raw/sites_raw.csv')

ANIME_FIELDS = ['bangumi_id', 'title_cn', 'title_jp',
                'air_date', 'platform', 'n_japan_sites']
SITES_FIELDS = ['site_id', 'bangumi_id', 'title_cn', 'air_date',
                'point_id', 'name', 'lat', 'lon', 'image_url', 'ep']


# ── 工具函数 ─────────────────────────────────────────────
def in_japan(lat, lon):
    return JAPAN_LAT[0] <= lat <= JAPAN_LAT[1] and \
           JAPAN_LON[0] <= lon <= JAPAN_LON[1]


def get_bangumi_quarter(session, year, quarter):
    """返回指定季度的所有动漫 subject 列表（已分页合并）。
    先尝试加 tag:'日本' 过滤（只获取日本动漫），若结果为空再不加 tag 重试。
    """
    q_start = (quarter - 1) * 3 + 1
    q_end   = q_start + 3
    date_from = f'{year}-{q_start:02d}-01'
    date_to   = f'{year}-{q_end:02d}-01' if q_end <= 12 \
                else f'{year+1}-01-01'

    results, offset, limit = [], 0, 50
    while True:
        try:
            r = session.post(
                'https://api.bgm.tv/v0/search/subjects',
                json={
                    'keyword': '',
                    'filter': {
                        'type': [2],
                        'tag':  ['日本'],          # 只要日本来源
                        'air_date': [f'>={date_from}', f'<{date_to}'],
                    },
                    'sort': 'rank',
                    'limit': limit,
                    'offset': offset,
                },
                timeout=15,
            )
            if r.status_code != 200:
                log.warning(f'Bangumi {year}Q{quarter} HTTP {r.status_code}')
                break
            data   = r.json()
            batch  = data.get('data') or []
            results.extend(batch)
            total  = data.get('total', 0)
            if offset + limit >= total or not batch:
                break
            offset += limit
            time.sleep(BGM_DELAY)
        except Exception as e:
            log.error(f'Bangumi {year}Q{quarter} error: {e}')
            break

    return results


def check_anitabi(bgm_id):
    """
    调用 anitabi /lite（不复用 session，避免 Cloudflare 连接跟踪）。
    返回 (n_japan_sites, japan_points_list)。
    403 响应时指数退避重试，最多 3 次。
    """
    import requests as _req
    backoff = 60
    for attempt in range(3):
        time.sleep(ANITABI_DELAY)
        try:
            r = _req.get(
                'https://api.anitabi.cn/bangumi/%d/lite' % bgm_id,
                headers={
                    'User-Agent': (
                        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                        'AppleWebKit/537.36 (KHTML, like Gecko) '
                        'Chrome/124.0.0.0 Safari/537.36'
                    ),
                    'Accept':          'application/json, */*',
                    'Accept-Language': 'zh-CN,zh;q=0.9,ja;q=0.8',
                    'Referer':         'https://anitabi.cn/',
                },
                timeout=12,
            )
            if r.status_code == 200:
                break
            if r.status_code == 403:
                log.warning('  anitabi 403 (CF限速)，等待 %ds…' % backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 300)
            else:
                return 0, []
        except Exception as e:
            log.debug('anitabi %d: %s' % (bgm_id, e))
            return 0, []
    else:
        log.warning('  anitabi %d: 3 次重试均失败，跳过' % bgm_id)
        return 0, []

    try:
        data = r.json()

        points = data.get('litePoints') or []
        japan = []
        for p in points:
            geo = p.get('geo') or []
            if len(geo) < 2:
                continue
            lat, lon = float(geo[0]), float(geo[1])
            if not in_japan(lat, lon):
                continue
            japan.append({
                'point_id':  p.get('id', ''),
                'name':      p.get('cn') or p.get('name', ''),
                'lat':       lat,
                'lon':       lon,
                'image_url': p.get('image', ''),
                'ep':        p.get('ep', ''),
            })
        return len(japan), japan

    except Exception as e:
        log.debug('anitabi %d: %s' % (bgm_id, e))
        return 0, []


# ── 主流程 ────────────────────────────────────────────────
def main():
    ANIME_CSV.parent.mkdir(parents=True, exist_ok=True)

    # ── 断点续跑：读取已完成的 bangumi_id ──────────────────
    done_ids = set()
    if ANIME_CSV.exists():
        with open(ANIME_CSV, encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                done_ids.add(int(row['bangumi_id']))
        log.info(f'断点续跑：已记录 {len(done_ids)} 部动漫')

    # ── 打开输出文件（追加模式）────────────────────────────
    anime_mode = 'a' if ANIME_CSV.exists() else 'w'
    sites_mode = 'a' if SITES_CSV.exists() else 'w'

    anime_f = open(ANIME_CSV, anime_mode, newline='', encoding='utf-8-sig')
    sites_f = open(SITES_CSV, sites_mode, newline='', encoding='utf-8-sig')

    anime_w = csv.DictWriter(anime_f, fieldnames=ANIME_FIELDS)
    sites_w = csv.DictWriter(sites_f, fieldnames=SITES_FIELDS)

    if anime_mode == 'w':
        anime_w.writeheader()
    if sites_mode == 'w':
        sites_w.writeheader()

    bgm_session = requests.Session()
    bgm_session.headers['User-Agent'] = 'anitabi-research/1.0 (ce.hou.gis@gmail.com)'
    bgm_session.headers['Accept']     = 'application/json'
    # anitabi 使用不复用 session 的独立函数，见 check_anitabi()

    total_anime, total_sites = len(done_ids), 0
    if SITES_CSV.exists():
        with open(SITES_CSV, encoding='utf-8-sig') as f:
            total_sites = sum(1 for _ in csv.DictReader(f))

    try:
        for year in YEARS:
            for quarter in range(1, 5):
                log.info(f'── {year} Q{quarter} ──')
                subjects = get_bangumi_quarter(bgm_session, year, quarter)
                log.info(f'  Bangumi 返回 {len(subjects)} 部')

                # 过滤已处理 + 去重
                new_subjects = [
                    s for s in subjects
                    if s['id'] not in done_ids
                ]
                # 去重（同一季度可能重复）
                seen = set()
                uniq = []
                for s in new_subjects:
                    if s['id'] not in seen:
                        seen.add(s['id'])
                        uniq.append(s)
                new_subjects = uniq

                if not new_subjects:
                    log.info('  全部已处理，跳过')
                    time.sleep(BGM_DELAY)
                    continue

                log.info('  待检查 %d 部（串行，间隔 %.0fs）' % (len(new_subjects), ANITABI_DELAY))

                # 串行检查 anitabi（CF限速，不可并发）
                for subj in new_subjects:
                    bgm_id   = subj['id']
                    title_cn = subj.get('name_cn') or ''
                    title_jp = subj.get('name', '')
                    air_date = subj.get('date', '')
                    platform = subj.get('platform', '')
                    done_ids.add(bgm_id)

                    try:
                        n, points = check_anitabi(bgm_id)
                    except Exception as e:
                        log.debug('  %d 异常: %s' % (bgm_id, e))
                        n, points = 0, []

                    if n < MIN_JAPAN_SITES:
                        continue

                    # 写动漫元数据
                    anime_w.writerow({
                        'bangumi_id':    bgm_id,
                        'title_cn':      title_cn,
                        'title_jp':      title_jp,
                        'air_date':      air_date,
                        'platform':      platform,
                        'n_japan_sites': n,
                    })
                    anime_f.flush()
                    total_anime += 1

                    # 写圣地坐标
                    for pt in points:
                        sites_w.writerow({
                            'site_id':    '%d_%s' % (bgm_id, pt['point_id']),
                            'bangumi_id': bgm_id,
                            'title_cn':   title_cn,
                            'air_date':   air_date,
                            'point_id':   pt['point_id'],
                            'name':       pt['name'],
                            'lat':        pt['lat'],
                            'lon':        pt['lon'],
                            'image_url':  pt['image_url'],
                            'ep':         pt['ep'],
                        })
                        total_sites += 1
                    sites_f.flush()

                    log.info('  ✓ [%d] %s  %d 个日本圣地  播出:%s'
                             % (bgm_id, title_cn or title_jp, n, air_date))

                time.sleep(BGM_DELAY)

    finally:
        anime_f.close()
        sites_f.close()

    log.info(f'\n{"="*60}')
    log.info(f'完成：{total_anime} 部动漫  {total_sites} 个圣地坐标')
    log.info(f'  动漫列表 → {ANIME_CSV}')
    log.info(f'  圣地坐标 → {SITES_CSV}')


if __name__ == '__main__':
    main()
