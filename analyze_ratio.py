"""
统计分析：2016年前后圣地巡礼占比

输入：
  data/raw/anime_list.csv   ← discover_anime.py 输出（全年代）
  data/raw/sites_raw.csv    ← discover_anime.py 输出（全年代）

输出：控制台打印分析结果 + data/processed/ratio_summary.csv
"""

import csv
from collections import defaultdict
from pathlib import Path

ANIME_CSV = Path('data/raw/anime_list.csv')
SITES_CSV = Path('data/raw/sites_raw.csv')
OUT_CSV   = Path('data/processed/ratio_summary.csv')
CUTOFF    = 2016

def load_csv(path):
    if not path.exists():
        print(f'文件不存在：{path}')
        return []
    with open(path, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def main():
    anime = load_csv(ANIME_CSV)
    sites = load_csv(SITES_CSV)

    if not anime or not sites:
        print('数据文件为空或不存在，请先运行 discover_anime.py')
        return

    # ── 按年统计动漫数量 ──────────────────────────────────────
    anime_by_year = defaultdict(int)
    for a in anime:
        y = a['air_date'][:4]
        if y.isdigit():
            anime_by_year[int(y)] += 1

    # ── 按年统计圣地数量 ──────────────────────────────────────
    sites_by_year = defaultdict(int)
    for s in sites:
        y = s['air_date'][:4]
        if y.isdigit():
            sites_by_year[int(y)] += 1

    total_anime = len(anime)
    total_sites = len(sites)
    post_anime  = sum(v for k, v in anime_by_year.items() if k >= CUTOFF)
    post_sites  = sum(v for k, v in sites_by_year.items() if k >= CUTOFF)
    pre_anime   = total_anime - post_anime
    pre_sites   = total_sites - post_sites

    print('=' * 56)
    print('  圣地巡礼 · 全年代统计（anitabi × Bangumi）')
    print('=' * 56)
    print(f'  {"年代":<12} {"动漫数":>8} {"占比":>8} {"圣地数":>8} {"占比":>8}')
    print(f'  {"-"*12} {"-"*8} {"-"*8} {"-"*8} {"-"*8}')
    print(f'  {"< 2016":<12} {pre_anime:>8,} {100*pre_anime/total_anime:>7.1f}% '
          f'{pre_sites:>8,} {100*pre_sites/total_sites:>7.1f}%')
    print(f'  {"≥ 2016":<12} {post_anime:>8,} {100*post_anime/total_anime:>7.1f}% '
          f'{post_sites:>8,} {100*post_sites/total_sites:>7.1f}%')
    print(f'  {"合计":<12} {total_anime:>8,} {"100.0%":>8} {total_sites:>8,} {"100.0%":>8}')
    print('=' * 56)

    # ── 按年份逐行打印 ────────────────────────────────────────
    print('\n  逐年明细：')
    print(f'  {"年份":<6} {"动漫数":>6} {"圣地数":>7}')
    all_years = sorted(set(anime_by_year) | set(sites_by_year))
    for y in all_years:
        a = anime_by_year.get(y, 0)
        s = sites_by_year.get(y, 0)
        marker = ' ◀ 2016+' if y == CUTOFF else ''
        print(f'  {y:<6} {a:>6,} {s:>7,}{marker}')

    # ── 保存汇总 CSV ──────────────────────────────────────────
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['year','n_anime','n_sites'])
        w.writeheader()
        for y in all_years:
            w.writerow({'year': y, 'n_anime': anime_by_year.get(y,0),
                        'n_sites': sites_by_year.get(y,0)})
    print(f'\n  汇总已保存 → {OUT_CSV}')

if __name__ == '__main__':
    main()
