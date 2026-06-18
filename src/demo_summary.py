"""
Quick demo summary: load the processed tables and print comparison stats.
Run after the full pipeline has completed.
"""

import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)


def run_summary():
    processed = ROOT / 'data' / 'processed'

    # ── Load tables ──────────────────────────────────────────────────────
    site_path    = processed / 'anime_site_table.csv'
    comp_path    = processed / 'comparison_site_table.csv'
    spatial_path = processed / 'spatial_context_table.csv'

    for p in [site_path, comp_path, spatial_path]:
        if not p.exists():
            print(f"Missing: {p}")
            return

    site_df    = pd.read_csv(site_path)
    comp_df    = pd.read_csv(comp_path)
    spatial_df = pd.read_csv(spatial_path)

    print("\n" + "="*62)
    print(" DEMO PIPELINE — SUMMARY STATISTICS")
    print("="*62)

    # ── Table counts ─────────────────────────────────────────────────────
    print(f"\n[1] Data counts")
    print(f"    anime pilgrimage sites : {len(site_df)}")
    print(f"    comparison sites       : {len(comp_df)}")
    print(f"    spatial features rows  : {len(spatial_df)}")
    print(f"      → anime (is_anime_site=1): {(spatial_df['is_anime_site']==1).sum()}")
    print(f"      → comparison (=0):         {(spatial_df['is_anime_site']==0).sum()}")

    # ── Anime breakdown ───────────────────────────────────────────────────
    print(f"\n[2] Sites by anime")
    print(site_df.groupby(['anime_id','anime_title'])
                 .size().rename('site_count').to_string())

    # ── Place type distribution ───────────────────────────────────────────
    print(f"\n[3] Place type distribution (anime sites)")
    print(site_df['place_type'].value_counts().to_string())

    # ── Spatial feature comparison ────────────────────────────────────────
    anime_sp = spatial_df[spatial_df['is_anime_site'] == 1]
    comp_sp  = spatial_df[spatial_df['is_anime_site'] == 0]

    if len(anime_sp) and len(comp_sp):
        print(f"\n[4] Spatial features: anime sites vs comparison (mean ± std)")
        print(f"    {'Metric':<40} {'Anime':>10} {'Comparison':>12}")
        print(f"    {'-'*64}")

        metrics = {
            'distance_to_nearest_station_m':  'Dist to station (m)',
            'distance_to_nearest_bus_stop_m': 'Dist to bus stop (m)',
            'poi_total_500m':                  'POI total (500m)',
            'restaurant_density_500m':         'Restaurants (500m)',
            'shrine_density_500m':             'Shrines (500m)',
            'tourism_service_score':           'Tourism service score',
        }
        for col, label in metrics.items():
            if col in spatial_df.columns:
                a = anime_sp[col].dropna()
                c = comp_sp[col].dropna()
                a_mean = f"{a.mean():.1f}" if len(a) else 'n/a'
                c_mean = f"{c.mean():.1f}" if len(c) else 'n/a'
                print(f"    {label:<40} {a_mean:>10} {c_mean:>12}")

    # ── Anime frame images downloaded ─────────────────────────────────────
    frames_base = ROOT / 'data' / 'raw' / 'anime_frames'
    frames = list(frames_base.rglob('*.jpg'))
    print(f"\n[5] Anime frame images downloaded: {len(frames)}")
    for d in sorted(frames_base.iterdir()):
        imgs = list(d.glob('*.jpg'))
        anime_id = d.name
        # site_df stores anime_id as int; compare by casting directory name
        match = site_df[site_df['anime_id'].astype(str) == anime_id]
        title = match['anime_title'].iloc[0] if len(match) else anime_id
        print(f"    [{anime_id}] {title}: {len(imgs)} images")

    # ── Quick data quality notes ──────────────────────────────────────────
    print(f"\n[6] Data quality")
    n_sensitive = int(site_df['private_or_sensitive_location'].sum())
    n_no_frame  = int((site_df['anime_frame_path'] == '').sum())
    print(f"    Sensitive/private locations flagged : {n_sensitive}")
    print(f"    Sites missing anime frame image     : {n_no_frame}")
    print(f"    Coordinate confidence (all 'high')  : {(site_df['coordinate_confidence']=='high').sum()}")

    # ── Tourism transformation table ──────────────────────────────────────
    tourism_path = processed / 'tourism_transformation_table.csv'
    if tourism_path.exists():
        t_df = pd.read_csv(tourism_path)
        print(f"\n[7] Tourism transformation table ({len(t_df)} rows)")
        print(t_df.groupby('anime_title')[
            ['years_since_broadcast', 'bangumi_rating',
             'fan_site_mentions', 'tourism_transformation_score']
        ].mean().round(2).to_string())

    print("\n" + "="*62)
    print(" Output files")
    print("="*62)
    for f in ['data/processed/anime_site_table.csv',
              'data/processed/comparison_site_table.csv',
              'data/processed/spatial_context_table.csv',
              'data/processed/tourism_transformation_table.csv']:
        p = ROOT / f
        sz = p.stat().st_size / 1024 if p.exists() else 0
        mark = '✓' if p.exists() else '✗'
        print(f"  {mark}  {f}  ({sz:.1f} KB)")
    print()


if __name__ == '__main__':
    run_summary()
