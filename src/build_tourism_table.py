"""
Build tourism_transformation_table.csv.

Combines:
  - Bangumi popularity scores (from API)
  - anitabi site count as fan_site_mentions proxy
  - Placeholder binary flags for official integration variables
    (official_map, collaboration_event, etc.) — to be manually coded.

Saves to data/processed/tourism_transformation_table.csv
"""

import logging
import time
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

TOURISM_COLS = [
    'site_id', 'anime_id', 'anime_title', 'broadcast_date',
    'years_since_broadcast', 'anime_popularity_score',
    'mal_members', 'mal_score', 'bangumi_rating',
    'fan_site_mentions',
    'official_map_dummy',
    'social_media_posts_total',
    'online_visibility_score',
    'official_promotion',
    'pilgrimage_map',
    'collaboration_event',
    'character_panel',
    'local_merchandise',
    'stamp_rally',
    'resident_conflict_notice',
    'sensitive_location',
    'tourism_transformation_score',
    'governance_notes',
]

CURRENT_YEAR = 2026


def build_tourism_table(site_df, config):
    """
    Parameters
    ----------
    site_df : pd.DataFrame  anime_site_table
    config  : dict

    Returns
    -------
    pd.DataFrame
    """
    out_path = Path(config['paths']['processed']) / 'tourism_transformation_table.csv'

    # --- Fetch Bangumi popularity for each unique anime ---
    anime_ids = site_df['anime_id'].unique()
    popularity = {}
    for aid in anime_ids:
        info = _fetch_bangumi_info(int(aid), config['api'])
        popularity[str(aid)] = info

    # --- Build per-site rows ---
    records = []
    for _, row in site_df.iterrows():
        aid   = str(row['anime_id'])
        pop   = popularity.get(aid, {})
        # Broadcast date comes from Bangumi API (per-anime), not per-site
        bcast = pop.get('broadcast_date') or str(row.get('broadcast_date', ''))
        year  = _extract_year(bcast)
        yrs   = (CURRENT_YEAR - year) if year else None

        # fan_site_mentions proxy: total anitabi points for this anime
        # Cast both sides to str to avoid int/str type mismatch after CSV round-trip
        total_sites = int((site_df['anime_id'].astype(str) == str(aid)).sum())

        records.append({
            'site_id':               row['site_id'],
            'anime_id':              aid,
            'anime_title':           row['anime_title'],
            'broadcast_date':        bcast,
            'years_since_broadcast': yrs,
            'anime_popularity_score': pop.get('popularity_score'),
            'mal_members':           pop.get('mal_members'),
            'mal_score':             None,
            'bangumi_rating':        pop.get('bangumi_rating'),
            'fan_site_mentions':     total_sites,
            # Binary integration flags — placeholder (0): manually code later
            'official_map_dummy':       0,
            'social_media_posts_total': None,
            'online_visibility_score':  None,
            'official_promotion':       0,
            'pilgrimage_map':           0,
            'collaboration_event':      0,
            'character_panel':          0,
            'local_merchandise':        0,
            'stamp_rally':              0,
            'resident_conflict_notice': int(row.get('private_or_sensitive_location', 0)),
            'sensitive_location':       int(row.get('private_or_sensitive_location', 0)),
            'tourism_transformation_score': None,   # computed after manual coding
            'governance_notes':         '',
        })

    df = pd.DataFrame(records, columns=TOURISM_COLS)

    # compute a preliminary transformation score from available data
    df['tourism_transformation_score'] = _compute_prelim_score(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    log.info(f"Saved tourism_transformation_table → {out_path}  ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# Bangumi API helpers
# ---------------------------------------------------------------------------

def _fetch_bangumi_info(subject_id, api_config):
    """Fetch broadcast date and popularity from Bangumi API.

    Bangumi v0 /subjects/{id} response:
      date: "2012-04-22"
      rating: { score: 8.2, total: 32732, rank: 151 }
    """
    base  = api_config.get('bangumi_base', 'https://api.bgm.tv/v0')
    delay = api_config.get('request_delay', 0.6)

    try:
        url  = f"{base}/subjects/{subject_id}"
        resp = requests.get(url, timeout=10,
                            headers={'User-Agent': 'anime-pilgrimage-research/1.0'})
        time.sleep(delay)
        if resp.status_code == 200:
            d           = resp.json()
            rating_info = d.get('rating') or {}
            score       = rating_info.get('score')
            members     = rating_info.get('total')
            return {
                'broadcast_date':  d.get('date', ''),
                'bangumi_rating':  score,
                'mal_members':     members,
                'popularity_score': _normalise_score(score, members),
            }
    except Exception as e:
        log.debug(f"  Bangumi API failed for {subject_id}: {e}")
    return {}


def _normalise_score(rating, members):
    """Simple 0-10 composite: 70% rating + 30% log10(members) / log10(maxmembers)."""
    if rating is None:
        return None
    score = 0.7 * (rating / 10)
    if members and members > 0:
        import math
        score += 0.3 * (math.log10(members) / math.log10(500_000))
    return round(min(score * 10, 10), 2)


def _extract_year(date_str):
    if not date_str:
        return None
    try:
        return int(str(date_str)[:4])
    except Exception:
        return None


def _compute_prelim_score(df):
    """
    Preliminary tourism transformation score based on:
      - Anime popularity (normalised)
      - Years since broadcast (older = more time to establish)
      - Number of anitabi spots (fan_site_mentions)
    All other binary fields are 0 in the placeholder table.
    """
    pop = df['anime_popularity_score'].fillna(5) / 10   # 0-1
    yrs = df['years_since_broadcast'].fillna(0).clip(0, 15) / 15   # 0-1
    fan = (df['fan_site_mentions'].fillna(0).clip(0, 50) / 50)      # 0-1
    score = (0.4 * pop + 0.3 * yrs + 0.3 * fan) * 10
    return score.round(2)
