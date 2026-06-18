"""
Build anime_site_table.csv from raw anitabi points.

Assigns site_ids, deduplicates, and normalises schema to match REQUIREMENT.md.
"""

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# Columns in the final table (REQUIREMENT.md §5.1)
SITE_TABLE_COLS = [
    'site_id', 'anime_id', 'anime_title', 'scene_id', 'episode',
    'lat', 'lon', 'location_name', 'city', 'prefecture', 'country',
    'place_type', 'urban_context',
    'anime_frame_path', 'streetview_image_path',
    'source_url', 'source_type', 'coordinate_confidence',
    'real_location_match_type', 'is_publicly_accessible',
    'private_or_sensitive_location', 'notes',
]


def build_anime_site_table(raw_points, config):
    """
    Parameters
    ----------
    raw_points : list[dict]   output of AnitabiFetcher.fetch_all()
    config     : dict         loaded from config.yaml

    Returns
    -------
    pd.DataFrame
    """
    out_path = Path(config['paths']['processed']) / 'anime_site_table.csv'

    if not raw_points:
        log.warning("No raw points supplied – returning empty DataFrame.")
        return pd.DataFrame(columns=SITE_TABLE_COLS)

    df = pd.DataFrame(raw_points)

    # --- deduplicate on (anime_id, lat, lon) rounded to 5 dp ---
    df['lat_r'] = df['lat'].round(5)
    df['lon_r'] = df['lon'].round(5)
    before = len(df)
    df = df.drop_duplicates(subset=['anime_id', 'lat_r', 'lon_r'])
    log.info(f"Deduplicated {before} → {len(df)} rows")
    df = df.drop(columns=['lat_r', 'lon_r'])

    # --- assign site_id ---
    df = df.reset_index(drop=True)
    df['site_id'] = df.apply(
        lambda r: f"S_{r['anime_id']}_{int(r.name):04d}", axis=1
    )

    # --- fill columns not yet populated ---
    df['streetview_image_path'] = ''
    df['urban_context']          = ''   # filled later by spatial extraction
    df['real_location_match_type'] = 'exact'
    df['is_publicly_accessible']   = 1
    df['prefecture']               = df['city'].apply(_city_to_prefecture)

    # --- drop internal helper columns not in final schema ---
    for col in ['image_url_remote', 'image_path_remote', 'anime_title_jp',
                'city_lat', 'city_lon']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # --- reorder to final schema (add missing cols as empty) ---
    for col in SITE_TABLE_COLS:
        if col not in df.columns:
            df[col] = ''
    df = df[SITE_TABLE_COLS]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    log.info(f"Saved anime_site_table → {out_path}  ({len(df)} rows)")

    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Quick lookup for demo anime cities → prefecture
_CITY_PREFECTURE = {
    '高山市': '岐阜県',
    '大洗町': '茨城県',
    '秩父市': '埼玉県',
    '東京':   '東京都',
    '東京都': '東京都',
    '京都':   '京都府',
    '京都市': '京都府',
    '大阪':   '大阪府',
    '鎌倉市': '神奈川県',
    '横浜市': '神奈川県',
    # ゆるキャン△ — anitabi may return simplified Chinese city names
    '山梨県': '山梨県',
    '山梨县': '山梨県',   # simplified Chinese → normalise to Japanese
    '富士河口湖町': '山梨県',
    '身延町': '山梨県',
    '静岡市': '静岡県',
    '浜松市': '静岡県',
}

def _city_to_prefecture(city):
    return _CITY_PREFECTURE.get(city, '')
