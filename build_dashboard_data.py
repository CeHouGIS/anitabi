"""
为 dashboard 生成数据文件（docs/data/*.json）：
  summary.json  总体统计 + 2016前后占比(逐年) + 聚合 DiD event-study 趋势
  sites.json    DiD合格圣地(地图点位) + 每点播出前后感知变化(用于着色)
  scores.json   每个合格圣地的感知时序(lively/beautiful/safety)
输入：data/raw/anime_list.csv, sites_raw.csv, sites_eligible.csv
     data/processed/panel_scores.csv
"""
import csv, json
from collections import defaultdict
from pathlib import Path

OUT = Path('docs/data'); OUT.mkdir(parents=True, exist_ok=True)
ATTRS3 = ['lively', 'beautiful', 'safety']

def rows(p): return list(csv.DictReader(open(p, encoding='utf-8-sig')))

anime  = rows('data/raw/anime_list.csv')
sites  = rows('data/raw/sites_raw.csv')
elig   = rows('data/raw/sites_eligible.csv')
panel  = rows('data/processed/panel_scores.csv')

def ym(s):           # 'YYYY-MM' -> (y,m)
    return int(s[:4]), int(s[5:7])

# ── 每个圣地的感知时序（按日期排序）────────────────────────
by_site = defaultdict(list)
for r in panel:
    by_site[r['site_id']].append(r)
for sid in by_site:
    by_site[sid].sort(key=lambda r: r['date'])

elig_meta = {r['site_id']: r for r in elig}

scores = {}
sites_geo = []
for sid, recs in by_site.items():
    m = elig_meta.get(sid)
    if not m:
        continue
    dates = [r['date'] for r in recs]
    lv = [round(float(r['lively']), 2) for r in recs]
    bt = [round(float(r['beautiful']), 2) for r in recs]
    sf = [round(float(r['safety']), 2) for r in recs]
    bc = m['air_date'][:7]
    scores[sid] = {'d': dates, 'l': lv, 'b': bt, 's': sf,
                   'bc': bc, 't': m['title_cn'], 'nm': m['name']}
    # 播出前/后 lively 均值差（地图着色）
    by, bm = ym(bc)
    pre = [r['lively'] for r in recs if ym(r['date']) < (by, bm)]
    post = [r['lively'] for r in recs if ym(r['date']) >= (by, bm)]
    delta = None
    if pre and post:
        delta = round(sum(map(float, post))/len(post) - sum(map(float, pre))/len(pre), 3)
    sites_geo.append({
        'id': sid, 'lat': round(float(m['lat']), 5), 'lon': round(float(m['lon']), 5),
        'bid': m['bangumi_id'], 't': m['title_cn'], 'nm': m['name'],
        'bc': bc, 'np': len(recs), 'dl': delta,
    })

# ── 2016 前后占比（逐年圣地数）─────────────────────────────
sites_by_year = defaultdict(int)
for s in sites:
    y = s['air_date'][:4]
    if y.isdigit():
        sites_by_year[int(y)] += 1
anime_by_year = defaultdict(int)
for a in anime:
    y = a['air_date'][:4]
    if y.isdigit():
        anime_by_year[int(y)] += 1
years = sorted(set(sites_by_year) | set(anime_by_year))
ratio_by_year = [{'y': y, 'sites': sites_by_year.get(y, 0),
                  'anime': anime_by_year.get(y, 0)} for y in years]
post_sites = sum(v for k, v in sites_by_year.items() if k >= 2016)
post_anime = sum(v for k, v in anime_by_year.items() if k >= 2016)
tot_sites = sum(sites_by_year.values())
tot_anime = len(anime)

# ── 聚合 DiD event-study：按"距播出年数"分桶取均值 ─────────
buckets = defaultdict(lambda: {a: [] for a in ATTRS3})
for r in panel:
    try:
        cy = int(r['date'][:4]); by = int(r['air_date'][:4])
    except Exception:
        continue
    rel = cy - by
    if -6 <= rel <= 9:
        for a in ATTRS3:
            buckets[rel][a].append(float(r[a]))
event = []
for rel in sorted(buckets):
    e = {'rel': rel, 'n': len(buckets[rel]['lively'])}
    for a in ATTRS3:
        v = buckets[rel][a]
        e[a] = round(sum(v)/len(v), 3) if v else None
    event.append(e)

summary = {
    'stats': {'anime': tot_anime, 'sites': tot_sites,
              'eligible': len(scores), 'panel': len(panel)},
    'ratio': {'post_anime': post_anime, 'post_sites': post_sites,
              'tot_anime': tot_anime, 'tot_sites': tot_sites,
              'pct_anime': round(100*post_anime/tot_anime, 1),
              'pct_sites': round(100*post_sites/tot_sites, 1)},
    'ratio_by_year': ratio_by_year,
    'event_study': event,
}

json.dump(summary, open(OUT/'summary.json', 'w'), ensure_ascii=False)
json.dump(sites_geo, open(OUT/'sites.json', 'w'), ensure_ascii=False)
json.dump(scores, open(OUT/'scores.json', 'w'), ensure_ascii=False)

print(f"summary.json  stats={summary['stats']}")
print(f"  占比: 2016+ 作品{summary['ratio']['pct_anime']}% 圣地{summary['ratio']['pct_sites']}%")
print(f"  event_study 桶: {[(e['rel'],e['n']) for e in event]}")
print(f"sites.json    {len(sites_geo)} 个合格圣地  (有DiD差值: {sum(1 for s in sites_geo if s['dl'] is not None)})")
print(f"scores.json   {len(scores)} 个圣地时序")
for f in ['summary.json','sites.json','scores.json']:
    print(f"  {f}: {(OUT/f).stat().st_size/1024:.0f} KB")
