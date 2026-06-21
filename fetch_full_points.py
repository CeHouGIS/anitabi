"""
补全圣地点位：对 anime_list.csv 里每部动漫调用 /bangumi/{id}/points/detail
（经 r.jina.ai 中继 + 真 curl，绕过校园网 IP 被 anitabi Cloudflare 封禁），
取全量带坐标点位，重建 sites_raw.csv（替代 /lite 只取 top-10 的不完整数据）。

输入：data/raw/anime_list.csv（discover_anime.py 输出，550 部）
输出：data/raw/sites_raw.csv（全量点位，覆盖原 top-10 版本）
     data/raw/points_checked.txt（已抓全量点位的 bgm_id，断点续跑）
"""

import csv, os, json as _json, subprocess, time, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

ANIME_CSV   = Path('data/raw/anime_list.csv')
SITES_CSV   = Path('data/raw/sites_raw.csv')
CHECKED_TXT = Path('data/raw/points_checked.txt')

JINA_PREFIX = 'https://r.jina.ai/'
JINA_KEY    = os.environ.get('JINA_API_KEY', '').strip()
DELAY       = float(os.environ.get('ANITABI_DELAY', '2.5' if JINA_KEY else '6.0'))
TRIES       = 4

JAPAN_LAT = (24.0, 46.0)
JAPAN_LON = (122.0, 154.0)
SITES_FIELDS = ['site_id', 'bangumi_id', 'title_cn', 'air_date',
                'point_id', 'name', 'lat', 'lon', 'image_url', 'ep']


def in_japan(lat, lon):
    return JAPAN_LAT[0] <= lat <= JAPAN_LAT[1] and JAPAN_LON[0] <= lon <= JAPAN_LON[1]


def _jina_points(bgm_id):
    """经 Jina 取 /points/detail，返回 (kind, data)。
    kind: 'json'(list) / 'blocked'(换IP重试) / 'ratelimit'(退避) / 'nodata' / 'error'"""
    url = '%shttps://api.anitabi.cn/bangumi/%s/points/detail' % (JINA_PREFIX, bgm_id)
    cmd = ['curl', '-s', '-w', '\n%{http_code}', '--connect-timeout', '15',
           '--max-time', '60', '-H', 'X-Return-Format: text']
    if JINA_KEY:
        cmd += ['-H', 'Authorization: Bearer %s' % JINA_KEY]
    cmd.append(url)
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=70).stdout
    except Exception as e:
        return 'error', str(e)
    nl = out.rfind('\n')
    body, code = (out[:nl], out[nl + 1:].strip()) if nl >= 0 else (out, '')
    b = body.strip()
    if 'you have been blocked' in body or 'Attention Required' in body:
        return 'blocked', body
    if code in ('403', '429', '402'):
        return 'ratelimit', body
    if b.startswith('['):
        try:
            return 'json', _json.loads(b)
        except Exception:
            return 'error', body
    if 'Target URL returned' in body or code in ('404', '410'):
        return 'nodata', body
    return 'error', body


def get_points(bgm_id):
    """返回该作品全部日本点位 list[dict]，失败返回 None。"""
    backoff = 30
    for attempt in range(TRIES):
        time.sleep(DELAY)
        kind, data = _jina_points(bgm_id)
        if kind == 'json':
            jp = []
            for p in data:
                geo = p.get('geo') or []
                if len(geo) < 2:
                    continue
                lat, lon = float(geo[0]), float(geo[1])
                if not in_japan(lat, lon):
                    continue
                jp.append({
                    'point_id':  p.get('id', ''),
                    'name':      p.get('cn') or p.get('name', ''),
                    'lat':       lat, 'lon': lon,
                    'image_url': p.get('image', ''),
                    'ep':        p.get('ep', ''),
                })
            return jp
        if kind == 'nodata':
            return []
        if kind == 'blocked':
            log.warning('  %s: Jina出口IP被拦，换IP重试(%d/%d)' % (bgm_id, attempt + 1, TRIES))
            continue
        if kind == 'ratelimit':
            log.warning('  %s: Jina限速，退避 %ds' % (bgm_id, backoff))
            time.sleep(backoff); backoff = min(backoff * 2, 240)
            continue
        time.sleep(5)
    log.warning('  %s: %d 次均失败，跳过' % (bgm_id, TRIES))
    return None


def main():
    if not ANIME_CSV.exists():
        log.error('%s 不存在' % ANIME_CSV); return
    anime = list(csv.DictReader(open(ANIME_CSV, encoding='utf-8-sig')))
    log.info('动漫 %d 部' % len(anime))

    done = set()
    if CHECKED_TXT.exists():
        for line in open(CHECKED_TXT, encoding='utf-8'):
            line = line.strip()
            if line:
                done.add(line)
        log.info('断点续跑：已抓 %d 部' % len(done))

    sites_mode = 'a' if SITES_CSV.exists() else 'w'
    sites_f = open(SITES_CSV, sites_mode, newline='', encoding='utf-8-sig')
    sites_w = csv.DictWriter(sites_f, fieldnames=SITES_FIELDS)
    if sites_mode == 'w':
        sites_w.writeheader()
    checked_f = open(CHECKED_TXT, 'a', encoding='utf-8')

    n_done = len(done)
    n_sites = 0
    try:
        for i, a in enumerate(anime):
            bgm_id = a['bangumi_id']
            if bgm_id in done:
                continue
            pts = get_points(bgm_id)
            if pts is None:        # 抓取失败，不记 checkpoint，下次重试
                continue
            for pt in pts:
                sites_w.writerow({
                    'site_id':   '%s_%s' % (bgm_id, pt['point_id']),
                    'bangumi_id': bgm_id, 'title_cn': a['title_cn'],
                    'air_date':  a['air_date'], 'point_id': pt['point_id'],
                    'name': pt['name'], 'lat': pt['lat'], 'lon': pt['lon'],
                    'image_url': pt['image_url'], 'ep': pt['ep'],
                })
                n_sites += 1
            sites_f.flush()
            checked_f.write('%s\n' % bgm_id); checked_f.flush()
            done.add(bgm_id); n_done += 1
            if n_done % 25 == 0:
                log.info('  进度 %d/%d  本次新增圣地 %d' % (n_done, len(anime), n_sites))
            if pts:
                log.info('  ✓ [%s] %s  %d 个日本点位' % (bgm_id, a['title_cn'][:16], len(pts)))
    finally:
        sites_f.close(); checked_f.close()

    log.info('完成：%d 部已抓，本次新增 %d 个圣地坐标 → %s' % (n_done, n_sites, SITES_CSV))


if __name__ == '__main__':
    main()
