"""
Generate comparison sites for the anime pilgrimage research.

A-type: random same-city points (3 per anime site)
B-type: same-place-type points from OSM  (best-effort, 2 per site)

Saves to data/processed/comparison_site_table.csv
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, MultiPolygon
import osmnx as ox

warnings.filterwarnings('ignore')
log = logging.getLogger(__name__)

# REQUIREMENT.md §5.2
COMPARISON_COLS = [
    'comparison_id', 'matched_site_id', 'comparison_group',
    'lat', 'lon', 'city', 'prefecture', 'country',
    'place_type', 'sampling_method', 'sampling_seed', 'is_anime_site', 'notes',
]

# OSM tags for B-type same-place-type sampling
_PLACE_TYPE_TAGS = {
    'station':        {'railway': 'station'},
    'shrine':         {'amenity': 'place_of_worship', 'religion': 'shinto'},
    'temple':         {'amenity': 'place_of_worship', 'religion': 'buddhist'},
    'school':         {'amenity': 'school'},
    'bridge':         {'man_made': 'bridge'},
    'coastal':        {'natural': 'coastline'},
    'park':           {'leisure': 'park'},
    'shopping_street':{'shop': True},
}


def generate_comparison_sites(site_df, config):
    """
    Parameters
    ----------
    site_df  : pd.DataFrame  (anime_site_table)
    config   : dict

    Returns
    -------
    pd.DataFrame
    """
    out_path = Path(config['paths']['processed']) / 'comparison_site_table.csv'
    seed     = config['comparison']['random_seed']
    n_rand   = config['comparison']['random_per_site']
    n_type   = config['comparison']['place_type_per_site']

    rng = np.random.default_rng(seed)

    records = []

    # Group by city so we only fetch the city boundary once per city
    cities = site_df['city'].unique()
    city_polygons = {}
    for city in cities:
        poly = _get_city_polygon(city)
        if poly is not None:
            city_polygons[city] = poly
            log.info(f"  Boundary found for city: {city}")
        else:
            log.warning(f"  Could not retrieve boundary for: {city}")

    for _, row in site_df.iterrows():
        city  = row['city']
        poly  = city_polygons.get(city)

        # --- A-type: random same-city ---
        if poly is not None:
            rand_pts = _sample_random_in_polygon(poly, n_rand, rng)
        else:
            # fallback: jitter around site location
            rand_pts = _jitter_around(row['lat'], row['lon'], n_rand, rng)

        for i, (rlat, rlon) in enumerate(rand_pts):
            records.append({
                'comparison_id':   f"CMP_A_{row['site_id']}_{i:02d}",
                'matched_site_id': row['site_id'],
                'comparison_group': 'same_city_random',
                'lat':             rlat,
                'lon':             rlon,
                'city':            city,
                'prefecture':      row.get('prefecture', ''),
                'country':         'Japan',
                'place_type':      'random',
                'sampling_method': 'random_in_polygon' if poly else 'jitter',
                'sampling_seed':   seed,
                'is_anime_site':   0,
                'notes':           '',
            })

        # --- B-type: same-place-type from OSM ---
        tags = _PLACE_TYPE_TAGS.get(row['place_type'])
        if tags and poly is not None:
            osm_pts = _sample_same_type_osm(poly, tags, n_type, row['lat'], row['lon'])
            for i, (tlat, tlon) in enumerate(osm_pts):
                records.append({
                    'comparison_id':   f"CMP_B_{row['site_id']}_{i:02d}",
                    'matched_site_id': row['site_id'],
                    'comparison_group': 'same_place_type',
                    'lat':             tlat,
                    'lon':             tlon,
                    'city':            city,
                    'prefecture':      row.get('prefecture', ''),
                    'country':         'Japan',
                    'place_type':      row['place_type'],
                    'sampling_method': 'osm_same_type',
                    'sampling_seed':   seed,
                    'is_anime_site':   0,
                    'notes':           '',
                })

    df = pd.DataFrame(records, columns=COMPARISON_COLS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    log.info(f"Saved comparison_site_table → {out_path}  ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------

def _get_city_polygon(city_name):
    """Fetch city boundary polygon from OSM via Nominatim."""
    queries = [
        f"{city_name}, Japan",
        city_name,
    ]
    for q in queries:
        try:
            gdf = ox.geocode_to_gdf(q)
            geom = gdf.geometry.iloc[0]
            if geom.geom_type in ('Polygon', 'MultiPolygon'):
                return geom
        except Exception:
            continue
    return None


def _sample_random_in_polygon(polygon, n, rng):
    """
    Sample n random (lat, lon) points inside polygon.
    Uses rejection sampling within the bounding box.
    """
    minx, miny, maxx, maxy = polygon.bounds
    pts = []
    max_attempts = n * 100
    attempts = 0
    while len(pts) < n and attempts < max_attempts:
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        p = Point(x, y)
        if polygon.contains(p):
            pts.append((y, x))   # lat, lon
        attempts += 1
    if len(pts) < n:
        log.debug(f"  Only found {len(pts)}/{n} random points in polygon")
    return pts


def _jitter_around(lat, lon, n, rng, radius_deg=0.05):
    """Fallback: random points near the given coordinate."""
    pts = []
    for _ in range(n):
        dlat = rng.uniform(-radius_deg, radius_deg)
        dlon = rng.uniform(-radius_deg, radius_deg)
        pts.append((lat + dlat, lon + dlon))
    return pts


def _sample_same_type_osm(polygon, tags, n, site_lat, site_lon):
    """
    Query OSM features matching `tags` within the city polygon,
    return up to n points that are not identical to the anime site.
    Uses features_from_point at the city centroid (osmnx 2.x compatible).
    """
    try:
        # Use city centroid and a large search radius
        centroid = polygon.centroid
        # Estimate city radius from bounding box diagonal
        minx, miny, maxx, maxy = polygon.bounds
        radius_deg = max(maxx - minx, maxy - miny) / 2.0
        radius_m   = int(radius_deg * 111_000)  # rough degrees→metres
        radius_m   = min(radius_m, 25_000)       # cap at 25 km

        feats = ox.features_from_point(
            (centroid.y, centroid.x),
            tags=tags,
            dist=radius_m
        )
        if feats is None or feats.empty:
            return []

        feats = feats.copy()
        feats['lat'] = feats.geometry.centroid.y
        feats['lon'] = feats.geometry.centroid.x

        # Exclude points too close to the anime site
        feats = feats[
            (abs(feats['lat'] - site_lat) > 0.001) |
            (abs(feats['lon'] - site_lon) > 0.001)
        ]
        # Keep only points inside the city polygon
        mask = feats.apply(
            lambda r: polygon.contains(Point(r['lon'], r['lat'])), axis=1
        )
        feats = feats[mask].dropna(subset=['lat', 'lon'])

        return [(r['lat'], r['lon']) for _, r in feats.head(n).iterrows()]

    except Exception as e:
        log.debug(f"  OSM same-type query failed: {e}")
        return []
