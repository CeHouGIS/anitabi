"""
Extract spatial features from OSM for all sites (anime + comparison).

Performance strategy: ONE bulk Overpass query per point (all POI types combined),
then filter locally — reduces API calls from ~8 to ~2 per point.

For each point computes:
  - Distance to nearest railway station / bus stop
  - POI counts in 500m and 1km buffers (restaurant, cafe, convenience, hotel,
    attraction, shrine)
  - Road intersection density in 500m buffer
  - Distance to nearest water body
  - Composite tourism service score

Saves to data/processed/spatial_context_table.csv

IMPORTANT: All metre-based calculations use the local UTM CRS (never EPSG:4326).
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import osmnx as ox
import pyproj

warnings.filterwarnings('ignore')
log = logging.getLogger(__name__)

SPATIAL_COLS = [
    'point_id', 'is_anime_site', 'source_site_id',
    'lat', 'lon', 'city', 'prefecture', 'place_type',
    'distance_to_nearest_station_m',
    'distance_to_nearest_bus_stop_m',
    'restaurant_density_500m',
    'cafe_density_500m',
    'convenience_store_density_500m',
    'hotel_density_1km',
    'tourist_attraction_density_1km',
    'shrine_density_500m',
    'poi_total_500m',
    'poi_total_1km',
    'road_intersection_density_500m',
    'distance_to_water_m',
    'tourism_service_score',
]

# Set by extract_spatial_features() from config
_road_network_enabled = True

# Targeted OSM tags — only the specific values we need.
# Using True (= all values for a key) causes very large responses.
_BULK_TAGS = {
    'amenity':  ['restaurant', 'cafe', 'fast_food', 'bar', 'pub',
                 'place_of_worship', 'school', 'hospital', 'pharmacy',
                 'toilets', 'parking'],
    'shop':     ['convenience', 'supermarket', 'mall'],
    'tourism':  ['hotel', 'motel', 'hostel', 'guest_house',
                 'attraction', 'museum', 'gallery', 'viewpoint', 'information'],
    'railway':  'station',
    'highway':  'bus_stop',
    'natural':  'water',
}


def extract_spatial_features(site_df, comparison_df, config):
    """
    Parameters
    ----------
    site_df       : pd.DataFrame  anime_site_table
    comparison_df : pd.DataFrame  comparison_site_table
    config        : dict

    Returns
    -------
    pd.DataFrame  spatial_context_table
    """
    global _road_network_enabled
    out_path  = Path(config['paths']['processed']) / 'spatial_context_table.csv'
    max_sites = config['spatial'].get('max_sites_for_demo', 20)
    _road_network_enabled = config['spatial'].get('extract_road_network', True)

    # Configure Overpass server and timeout
    overpass_url = config['spatial'].get('overpass_url')
    if overpass_url:
        ox.settings.overpass_url = overpass_url
    # timeout: Overpass server-side query limit (seconds)
    ox.settings.timeout = 60
    # requests_kwargs: HTTP client hard timeout — prevents 3+ min hangs when
    # the Overpass server accepts the connection but stalls on large queries
    ox.settings.requests_kwargs = {'timeout': 90}

    log.info(f"Overpass: {ox.settings.overpass_url}")
    log.info(f"Road network: {'enabled' if _road_network_enabled else 'disabled'}")

    all_rows = _prepare_anime_rows(site_df) + _prepare_comparison_rows(comparison_df)
    all_rows = all_rows[:max_sites]
    log.info(f"Extracting spatial features for {len(all_rows)} points (cap={max_sites})")

    records = []
    for i, row in enumerate(all_rows):
        log.info(f"  [{i+1}/{len(all_rows)}] {row['point_id']} ({row['lat']:.4f}, {row['lon']:.4f})")
        feats = _extract_one_point(row['lat'], row['lon'])
        records.append({**row, **feats})

    df = pd.DataFrame(records)
    df['tourism_service_score'] = _compute_tourism_score(df)

    for col in SPATIAL_COLS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[SPATIAL_COLS]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    log.info(f"Saved spatial_context_table → {out_path}  ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# Main feature extraction (2 API calls per point)
# ---------------------------------------------------------------------------

def _extract_one_point(lat, lon):
    results = {k: np.nan for k in [
        'distance_to_nearest_station_m', 'distance_to_nearest_bus_stop_m',
        'restaurant_density_500m', 'cafe_density_500m',
        'convenience_store_density_500m', 'hotel_density_1km',
        'tourist_attraction_density_1km', 'shrine_density_500m',
        'poi_total_500m', 'poi_total_1km',
        'road_intersection_density_500m', 'distance_to_water_m',
    ]}

    try:
        local_crs = _utm_crs(lon)
        to_local  = pyproj.Transformer.from_crs('EPSG:4326', local_crs, always_xy=True)
        cx, cy    = to_local.transform(lon, lat)
        ctr       = Point(cx, cy)  # centre in projected metres

        # ── API call 1: bulk POI fetch within 1 km ────────────────────────
        all_feats = _fetch_bulk_pois(lat, lon, dist=1000)

        if all_feats is not None and not all_feats.empty:
            pois_proj = all_feats.to_crs(local_crs).copy()
            pois_proj['_dist'] = pois_proj.geometry.centroid.distance(ctr)

            def count_within(mask, r):
                return int((mask & (pois_proj['_dist'] <= r)).sum())

            def min_dist(mask):
                sub = pois_proj.loc[mask, '_dist']
                return round(float(sub.min()), 1) if len(sub) else np.nan

            # Masks
            is_restaurant  = pois_proj.get('amenity', pd.Series(dtype=str)) == 'restaurant'
            is_cafe        = pois_proj.get('amenity', pd.Series(dtype=str)) == 'cafe'
            is_convenience = pois_proj.get('shop', pd.Series(dtype=str)) == 'convenience'
            is_hotel       = pois_proj.get('tourism', pd.Series(dtype=str)).isin(
                                 ['hotel', 'motel', 'hostel', 'guest_house'])
            is_attraction  = pois_proj.get('tourism', pd.Series(dtype=str)).isin(
                                 ['attraction', 'museum', 'gallery', 'viewpoint'])
            is_shrine      = (pois_proj.get('amenity', pd.Series(dtype=str)) == 'place_of_worship') & \
                             (pois_proj.get('religion', pd.Series(dtype=str)).isin(['shinto', 'buddhist']))
            is_station     = pois_proj.get('railway', pd.Series(dtype=str)) == 'station'
            is_bus         = pois_proj.get('highway', pd.Series(dtype=str)) == 'bus_stop'
            is_water       = pois_proj.get('natural', pd.Series(dtype=str)) == 'water'

            results['distance_to_nearest_station_m']  = min_dist(is_station)
            results['distance_to_nearest_bus_stop_m'] = min_dist(is_bus)
            results['distance_to_water_m']            = min_dist(is_water)

            results['restaurant_density_500m']          = count_within(is_restaurant,  500)
            results['cafe_density_500m']                = count_within(is_cafe,         500)
            results['convenience_store_density_500m']   = count_within(is_convenience,  500)
            results['hotel_density_1km']                = count_within(is_hotel,        1000)
            results['tourist_attraction_density_1km']   = count_within(is_attraction,   1000)
            results['shrine_density_500m']              = count_within(is_shrine,        500)

            # Total POI (exclude infrastructure like bus stops)
            infra = is_station | is_bus | is_water
            results['poi_total_500m'] = count_within(~infra, 500)
            results['poi_total_1km']  = count_within(~infra, 1000)

        # ── API call 2 (optional): road network for intersection density ─────
        if _road_network_enabled:
            try:
                G = ox.graph_from_point((lat, lon), dist=600, network_type='walk',
                                        simplify=True)
                nodes, _ = ox.graph_to_gdfs(G)
                nodes_p  = nodes.to_crs(local_crs)
                n_inter  = int((nodes_p.geometry.distance(ctr) <= 500).sum())
                area_km2 = np.pi * 0.5 ** 2
                results['road_intersection_density_500m'] = round(n_inter / area_km2, 2)
            except Exception as e:
                log.debug(f"    Road network failed: {e}")

    except Exception as e:
        log.warning(f"  Feature extraction error ({lat:.4f},{lon:.4f}): {e}")

    return results


# ---------------------------------------------------------------------------
# OSM fetch helper
# ---------------------------------------------------------------------------

def _fetch_bulk_pois(lat, lon, dist=1000):
    """Single Overpass query for all POI types at once."""
    try:
        return ox.features_from_point((lat, lon), tags=_BULK_TAGS, dist=dist)
    except Exception as e:
        log.debug(f"    Bulk POI query failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Row preparation helpers
# ---------------------------------------------------------------------------

def _prepare_anime_rows(site_df):
    return [{
        'point_id':      r['site_id'],
        'is_anime_site': 1,
        'source_site_id': r['site_id'],
        'lat':  float(r['lat']),  'lon':  float(r['lon']),
        'city': r.get('city',''), 'prefecture': r.get('prefecture',''),
        'place_type': r.get('place_type',''),
    } for _, r in site_df.iterrows()]


def _prepare_comparison_rows(comp_df):
    return [{
        'point_id':      r['comparison_id'],
        'is_anime_site': 0,
        'source_site_id': r['matched_site_id'],
        'lat':  float(r['lat']),  'lon':  float(r['lon']),
        'city': r.get('city',''), 'prefecture': r.get('prefecture',''),
        'place_type': r.get('place_type',''),
    } for _, r in comp_df.iterrows()]


# ---------------------------------------------------------------------------
# Coordinate & scoring helpers
# ---------------------------------------------------------------------------

def _utm_crs(lon):
    zone = int((lon + 180) / 6) + 1
    return f"EPSG:{32600 + zone}"


def _compute_tourism_score(df):
    weights = {
        'restaurant_density_500m':        1.5,
        'hotel_density_1km':              2.0,
        'tourist_attraction_density_1km': 2.0,
        'convenience_store_density_500m': 0.5,
        'cafe_density_500m':              1.0,
        'shrine_density_500m':            1.0,
        'poi_total_500m':                 2.0,
    }
    score = pd.Series(0.0, index=df.index)
    total = sum(weights.values())
    for col, w in weights.items():
        if col in df.columns:
            s  = df[col].fillna(0)
            mx = s.max()
            if mx > 0:
                score += (s / mx) * w
    return (score / total * 10).round(2)
