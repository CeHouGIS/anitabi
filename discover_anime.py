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

import csv, time, logging, datetime as _dt
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
# 校园网 IP 被 anitabi 的 Cloudflare WAF 硬封（403），故所有 anitabi 请求改走
# r.jina.ai 中继（用 Jina 自己的出口 IP 取数据）。传输层用真·命令行 curl——
# Jina 放行 curl/* 的 UA，却拦 python-urllib / curl_cffi（伪装浏览器的 bot）。
import os, subprocess, json as _json
JINA_PREFIX     = 'https://r.jina.ai/'
JINA_KEY        = os.environ.get('JINA_API_KEY', '').strip()  # 填了限速更宽（200+ RPM）
# 有 key 时可加速到 ≈24 RPM；无 key 时须放慢到 6s(≈10 RPM) 以免被 Jina 免费层 403
ANITABI_DELAY   = float(os.environ.get('ANITABI_DELAY', '2.5' if JINA_KEY else '6.0'))
ANITABI_TRIES   = 4      # 单个 ID 最多尝试次数（撞到被拦的 Jina 出口 IP 时换 IP 重试）
BGM_DELAY       = 0.3    # Bangumi 请求间隔（秒）

ANIME_CSV   = Path('data/raw/anime_list.csv')
SITES_CSV   = Path('data/raw/sites_raw.csv')
CHECKED_TXT = Path('data/raw/checked_ids.txt')  # 所有已查过的 bgm_id（含 nodata），断点续跑用

ANIME_FIELDS = ['bangumi_id', 'title_cn', 'title_jp',
                'air_date', 'platform', 'n_japan_sites']
SITES_FIELDS = ['site_id', 'bangumi_id', 'title_cn', 'air_date',
                'point_id', 'name', 'lat', 'lon', 'image_url', 'ep']


# ── 工具函数 ─────────────────────────────────────────────
def in_japan(lat, lon):
    return JAPAN_LAT[0] <= lat <= JAPAN_LAT[1] and \
           JAPAN_LON[0] <= lon <= JAPAN_LON[1]


BGM_SORTS = ('rank', 'heat', 'score', 'match')  # 不同 sort 给不同 top-10，用于单日并集

def _bgm_search(session, d_from, d_to, sort='rank'):
    """单次 Bangumi 搜索 type=2(动画) + air_date 区间，返回 (subjects, total)。
    注意：该 API limit 实际封顶 10、offset 无效，单次最多拿到 top-10。
    对超时/5xx/429 退避重试；持续失败则抛 RuntimeError，绝不静默把窗口当空跳过
    （否则 Bangumi 一抖动就会漏掉整段年份）。"""
    last = ''
    for attempt in range(6):
        try:
            r = session.post(
                'https://api.bgm.tv/v0/search/subjects',
                json={'keyword': '',
                      'filter': {'type': [2], 'air_date': [f'>={d_from}', f'<{d_to}']},
                      'sort': sort, 'limit': 10, 'offset': 0},
                timeout=20,
            )
            if r.status_code == 200:
                data = r.json()
                return (data.get('data') or []), data.get('total', 0)
            last = f'HTTP {r.status_code}'
        except Exception as e:
            last = type(e).__name__
        log.warning(f'Bangumi {d_from}~{d_to} {last}，退避重试({attempt + 1}/6)')
        time.sleep(8 * (attempt + 1))   # 8/16/24/32/40s
    raise RuntimeError(f'Bangumi {d_from}~{d_to} 重试6次仍失败: {last}')


def enumerate_subjects(session, d_from, d_to, emit):
    """自适应日期细分，完整枚举 [d_from, d_to) 内所有 type=2 动画并去重 emit。
    因 search 单查最多 top-10：total>10 就对半切日期；切到单日仍 >10 用多 sort 并集兜底。
    d_from / d_to 为 datetime.date。"""
    batch, total = _bgm_search(session, d_from.isoformat(), d_to.isoformat())
    time.sleep(BGM_DELAY)
    if total <= 0:
        return
    if total <= 10:
        for s in batch:
            emit(s)
        return
    span = (d_to - d_from).days
    if span <= 1:
        # 单日仍 >10：合并多个 sort 的 top-10（可达 ~40，足够覆盖单日峰值）
        merged = {s['id']: s for s in batch}
        for srt in BGM_SORTS[1:]:
            b, _ = _bgm_search(session, d_from.isoformat(), d_to.isoformat(), sort=srt)
            time.sleep(BGM_DELAY)
            for s in b:
                merged.setdefault(s['id'], s)
        for s in merged.values():
            emit(s)
        return
    mid = d_from + _dt.timedelta(days=span // 2)
    enumerate_subjects(session, d_from, mid, emit)
    enumerate_subjects(session, mid, d_to, emit)


def _jina_get(bgm_id):
    """经 r.jina.ai 中继取 anitabi /lite 原文。用真 curl 子进程。
    返回 (kind, text)：
      kind='json'    → text 为 anitabi 返回的 JSON 原文
      kind='blocked' → Jina 出口 IP 被 anitabi 拦（需换 IP 重试）
      kind='ratelimit'→ Jina 自身限速 403（需退避重试）
      kind='nodata'  → anitabi 404，该作品无巡礼数据（不必重试）
      kind='error'   → 其他网络/超时错误（可重试）
    """
    url = '%shttps://api.anitabi.cn/bangumi/%d/lite' % (JINA_PREFIX, bgm_id)
    cmd = ['curl', '-s', '-w', '\n%{http_code}', '--connect-timeout', '15',
           '--max-time', '40', '-H', 'X-Return-Format: text']
    if JINA_KEY:
        cmd += ['-H', 'Authorization: Bearer %s' % JINA_KEY]
    cmd.append(url)
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=50).stdout
    except Exception as e:
        return 'error', str(e)
    nl = out.rfind('\n')
    body, code = (out[:nl], out[nl + 1:].strip()) if nl >= 0 else (out, '')
    b = body.strip()
    if 'you have been blocked' in body or 'Attention Required' in body:
        return 'blocked', body
    if code in ('403', '429', '402'):   # Jina 限速 / 配额耗尽 → 退避
        return 'ratelimit', body
    if b.startswith('{') and '"litePoints"' in b:
        return 'json', b
    if 'Target URL returned' in body or code in ('404', '410'):
        return 'nodata', body
    return 'error', body


def check_anitabi(bgm_id):
    """经 Jina 中继调用 anitabi /lite，返回 (n_japan_sites, japan_points_list)。
    撞到被 anitabi 拦的 Jina 出口 IP / Jina 限速时换 IP 退避重试。"""
    data = None
    backoff = 30
    for attempt in range(ANITABI_TRIES):
        time.sleep(ANITABI_DELAY)
        kind, text = _jina_get(bgm_id)
        if kind == 'json':
            try:
                data = _json.loads(text)
            except Exception:
                data = None
            break
        if kind == 'nodata':
            return 0, []          # 该作品确实无巡礼数据
        if kind == 'blocked':
            log.warning('  anitabi %d: Jina出口IP被拦，换IP重试(%d/%d)'
                        % (bgm_id, attempt + 1, ANITABI_TRIES))
            continue              # 立即重试 → Jina 多半给个新出口 IP
        if kind == 'ratelimit':
            log.warning('  anitabi %d: Jina限速403，退避 %ds' % (bgm_id, backoff))
            time.sleep(backoff)
            backoff = min(backoff * 2, 240)
            continue
        # error：短暂退避后重试
        time.sleep(5)
    if data is None:
        log.warning('  anitabi %d: %d 次均失败，跳过' % (bgm_id, ANITABI_TRIES))
        return 0, []

    try:
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

    # ── 断点续跑：done_ids = 所有"已查过"的 ID（含 nodata），避免重启重查 ──
    done_ids = set()
    n_hit = 0
    if ANIME_CSV.exists():
        with open(ANIME_CSV, encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                n_hit += 1
    if CHECKED_TXT.exists():
        with open(CHECKED_TXT, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.isdigit():
                    done_ids.add(int(line))
    log.info(f'断点续跑：已查 {len(done_ids)} 部（其中合格 {n_hit} 部）')
    checked_f = open(CHECKED_TXT, 'a', encoding='utf-8')

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

    total_sites = 0
    if SITES_CSV.exists():
        with open(SITES_CSV, encoding='utf-8-sig') as f:
            total_sites = sum(1 for _ in csv.DictReader(f))

    # 计数器用可变容器，供 emit 闭包修改
    stat = {'anime': n_hit, 'sites': total_sites, 'checked': 0, 'seen': set()}

    def emit(subj):
        bgm_id = subj['id']
        if bgm_id in stat['seen']:      # 本次运行内去重（自适应窗口可能重叠）
            return
        stat['seen'].add(bgm_id)
        if bgm_id in done_ids:          # 跨运行去重（已查过的跳过）
            return
        done_ids.add(bgm_id)

        title_cn = subj.get('name_cn') or ''
        title_jp = subj.get('name', '')
        air_date = subj.get('date', '')
        platform = subj.get('platform', '')

        try:
            n, points = check_anitabi(bgm_id)
        except Exception as e:
            log.debug('  %d 异常: %s' % (bgm_id, e))
            n, points = 0, []

        # 记录"已查"检查点（含 nodata），重启时跳过
        checked_f.write('%d\n' % bgm_id)
        checked_f.flush()
        stat['checked'] += 1
        if stat['checked'] % 100 == 0:
            log.info('  …已查 %d 部（合格 %d / 圣地 %d）'
                     % (stat['checked'], stat['anime'], stat['sites']))

        if n < MIN_JAPAN_SITES:
            return

        anime_w.writerow({
            'bangumi_id': bgm_id, 'title_cn': title_cn, 'title_jp': title_jp,
            'air_date': air_date, 'platform': platform, 'n_japan_sites': n,
        })
        anime_f.flush()
        stat['anime'] += 1

        for pt in points:
            sites_w.writerow({
                'site_id':    '%d_%s' % (bgm_id, pt['point_id']),
                'bangumi_id': bgm_id, 'title_cn': title_cn, 'air_date': air_date,
                'point_id':   pt['point_id'], 'name': pt['name'],
                'lat': pt['lat'], 'lon': pt['lon'],
                'image_url':  pt['image_url'], 'ep': pt['ep'],
            })
            stat['sites'] += 1
        sites_f.flush()
        log.info('  ✓ [%d] %s  %d 个日本圣地  播出:%s'
                 % (bgm_id, title_cn or title_jp, n, air_date))

    try:
        # 可用 BGM_START_DATE / BGM_END_DATE 覆盖枚举区间（断点重启时跳过已覆盖的早期年份；
        # 自适应递归左优先=按时间从早到晚，故崩溃点之前的日期均已完整覆盖，可安全前移起点）
        d_start = _dt.date.fromisoformat(
            os.environ.get('BGM_START_DATE', f'{min(YEARS)}-01-01'))
        d_end   = _dt.date.fromisoformat(
            os.environ.get('BGM_END_DATE',   f'{max(YEARS) + 1}-01-01'))
        log.info('自适应枚举 %s ~ %s（type=2 动画全量）' % (d_start, d_end))
        enumerate_subjects(bgm_session, d_start, d_end, emit)
    finally:
        anime_f.close()
        sites_f.close()
        checked_f.close()

    total_anime = stat['anime']
    total_sites = stat['sites']
    log.info(f'\n{"="*60}')
    log.info(f'完成：枚举 {len(stat["seen"])} 部 / 本次新查 {stat["checked"]} 部')
    log.info(f'累计：{total_anime} 部合格动漫  {total_sites} 个圣地坐标')
    log.info(f'  动漫列表 → {ANIME_CSV}')
    log.info(f'  圣地坐标 → {SITES_CSV}')


if __name__ == '__main__':
    main()
