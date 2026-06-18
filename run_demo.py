#!/opt/conda/bin/python3.10
"""
Demo pipeline runner — anime pilgrimage spatial research project.

Steps executed:
  1. Fetch anime metadata + pilgrimage points from anitabi API
  2. Download anime frame images
  3. Build data/processed/anime_site_table.csv
  4. Generate comparison sites (OSM) → comparison_site_table.csv
  5. Extract spatial features (OSM) → spatial_context_table.csv
  6. (Optional) Download GSV street-view images
  7. Print summary statistics

Usage:
  cd /workplace/anitabi_project
  /opt/conda/bin/python3.10 run_demo.py [--step N] [--no-gsv]
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import yaml
import pandas as pd

# ---- make src importable regardless of cwd ----
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from src.fetch_anitabi       import AnitabiFetcher
from src.build_site_table    import build_anime_site_table
from src.generate_comparison import generate_comparison_sites
from src.extract_spatial     import extract_spatial_features
from src.build_tourism_table import build_tourism_table
from src.gsv_downloader      import download_gsv_for_sites


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def load_config(path=None):
    path = path or ROOT / 'config' / 'config.yaml'
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s  %(levelname)-7s  %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(ROOT / 'reports' / 'pipeline.log', mode='w'),
        ]
    )


def ensure_dirs(config):
    dirs = [
        config['paths']['raw']['anime_sites'],
        config['paths']['raw']['anime_frames'],
        config['paths']['raw']['streetview'],
        config['paths']['interim'],
        config['paths']['processed'],
        config['paths']['outputs']['figures'],
        config['paths']['outputs']['tables'],
        config['paths']['outputs']['maps'],
        config['paths']['reports'],
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step1_fetch(config):
    """Fetch anime metadata + pilgrimage points from anitabi API."""
    print("\n=== Step 1: Fetch anitabi data ===")
    fetcher = AnitabiFetcher(config)
    raw_points = fetcher.fetch_all()
    print(f"  Raw points collected: {len(raw_points)}")
    # Save intermediate
    interim_path = Path(config['paths']['interim']) / 'raw_points.csv'
    pd.DataFrame(raw_points).to_csv(interim_path, index=False, encoding='utf-8-sig')
    print(f"  Saved interim → {interim_path}")
    return raw_points


def step2_build_site_table(raw_points, config):
    """Clean and normalise into anime_site_table.csv."""
    print("\n=== Step 2: Build anime_site_table ===")
    site_df = build_anime_site_table(raw_points, config)
    print(f"  anime_site_table rows: {len(site_df)}")
    print(f"  Anime IDs: {site_df['anime_id'].unique().tolist()}")
    print(f"  Place type distribution:\n{site_df['place_type'].value_counts().to_string()}")
    return site_df


def step3_comparison(site_df, config):
    """Generate comparison sites."""
    print("\n=== Step 3: Generate comparison sites ===")
    comp_df = generate_comparison_sites(site_df, config)
    print(f"  comparison_site_table rows: {len(comp_df)}")
    print(f"  Groups: {comp_df['comparison_group'].value_counts().to_string()}")
    return comp_df


def step4_spatial(site_df, comp_df, config):
    """Extract spatial features."""
    print("\n=== Step 4: Extract spatial features (OSM) ===")
    spatial_df = extract_spatial_features(site_df, comp_df, config)
    print(f"  spatial_context_table rows: {len(spatial_df)}")
    cols = ['distance_to_nearest_station_m', 'poi_total_500m', 'tourism_service_score']
    for col in cols:
        if col in spatial_df.columns:
            vals = spatial_df[col].dropna()
            if len(vals):
                print(f"  {col}: mean={vals.mean():.1f}, max={vals.max():.1f}")
    return spatial_df


def step5_tourism(site_df, config):
    """Build tourism transformation table."""
    print("\n=== Step 5: Build tourism transformation table ===")
    tourism_df = build_tourism_table(site_df, config)
    print(f"  tourism_transformation_table rows: {len(tourism_df)}")
    cols = ['anime_title', 'bangumi_rating', 'fan_site_mentions', 'tourism_transformation_score']
    print(tourism_df[cols].drop_duplicates('anime_title').to_string(index=False))
    return tourism_df


def step6_gsv(site_df, config):
    """Download GSV street-view images (optional)."""
    print("\n=== Step 6: GSV street-view download ===")
    if not config['gsv']['enabled']:
        print("  Skipped (gsv.enabled=false in config)")
        return site_df
    site_df = download_gsv_for_sites(site_df, config)
    n_downloaded = (site_df['streetview_image_path'] != '').sum()
    print(f"  Downloaded GSV images for {n_downloaded} sites")
    # Update anime_site_table with new paths
    out = Path(config['paths']['processed']) / 'anime_site_table.csv'
    site_df.to_csv(out, index=False, encoding='utf-8-sig')
    return site_df


def print_summary(site_df, comp_df, spatial_df):
    print("\n" + "="*60)
    print("DEMO PIPELINE COMPLETE")
    print("="*60)
    print(f"\n  Anime pilgrimage sites : {len(site_df)}")
    print(f"  Comparison sites       : {len(comp_df)}")
    print(f"  Spatial features       : {len(spatial_df)}")

    print("\n  Output files:")
    for f in [
        'data/processed/anime_site_table.csv',
        'data/processed/comparison_site_table.csv',
        'data/processed/spatial_context_table.csv',
    ]:
        p = ROOT / f
        size = p.stat().st_size if p.exists() else 0
        print(f"    {'✓' if p.exists() else '✗'} {f}  ({size/1024:.1f} KB)")

    if len(spatial_df):
        anime_rows = spatial_df[spatial_df['is_anime_site'] == 1]
        comp_rows  = spatial_df[spatial_df['is_anime_site'] == 0]
        print("\n  Spatial feature comparison (anime vs comparison):")
        for col in ['distance_to_nearest_station_m', 'poi_total_500m']:
            if col in spatial_df.columns:
                a = anime_rows[col].dropna().mean()
                c = comp_rows[col].dropna().mean()
                print(f"    {col}:")
                print(f"      anime sites : {a:.1f}")
                print(f"      comparison  : {c:.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Anime pilgrimage demo pipeline')
    parser.add_argument('--step', type=int, default=0,
                        help='Run from this step (1=fetch, 2=build, 3=comparison, 4=spatial, 5=gsv)')
    parser.add_argument('--gsv', action='store_true', default=False,
                        help='Enable GSV download (overrides config)')
    args = parser.parse_args()

    # Setup
    setup_logging()
    config = load_config()
    ensure_dirs(config)

    if args.gsv:
        config['gsv']['enabled'] = True

    log = logging.getLogger(__name__)
    log.info("Starting anime pilgrimage demo pipeline")
    log.info(f"Anime IDs: {config['demo']['anime_ids']}")
    t0 = time.time()

    # Resolve all output paths relative to project root
    interim_path    = ROOT / config['paths']['interim'] / 'raw_points.csv'
    site_table_path = ROOT / config['paths']['processed'] / 'anime_site_table.csv'
    comp_table_path = ROOT / config['paths']['processed'] / 'comparison_site_table.csv'
    spatial_path    = ROOT / config['paths']['processed'] / 'spatial_context_table.csv'

    if args.step <= 1:
        raw_points = step1_fetch(config)
    else:
        if not interim_path.exists():
            print(f"ERROR: interim file not found: {interim_path}")
            print("Run from step 1 first:  python run_demo.py")
            sys.exit(1)
        raw_points = pd.read_csv(interim_path).to_dict('records')
        print(f"\n=== Step 1: Skipped (loaded {len(raw_points)} rows from {interim_path})")

    if args.step <= 2:
        site_df = step2_build_site_table(raw_points, config)
    else:
        site_df = pd.read_csv(site_table_path)
        print(f"\n=== Step 2: Skipped (loaded {len(site_df)} rows)")

    if args.step <= 3:
        comp_df = step3_comparison(site_df, config)
    else:
        comp_df = pd.read_csv(comp_table_path)
        print(f"\n=== Step 3: Skipped (loaded {len(comp_df)} rows)")

    if args.step <= 4:
        spatial_df = step4_spatial(site_df, comp_df, config)
    else:
        spatial_df = pd.read_csv(spatial_path) if spatial_path.exists() else pd.DataFrame()
        print(f"\n=== Step 4: Skipped")

    tourism_table_path = ROOT / config['paths']['processed'] / 'tourism_transformation_table.csv'

    if args.step <= 5:
        tourism_df = step5_tourism(site_df, config)
    else:
        tourism_df = pd.read_csv(tourism_table_path) if tourism_table_path.exists() else pd.DataFrame()
        print(f"\n=== Step 5: Skipped")

    if args.step <= 6:
        site_df = step6_gsv(site_df, config)

    print_summary(site_df, comp_df, spatial_df)
    log.info(f"Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
