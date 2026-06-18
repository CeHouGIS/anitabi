"""
Fetch anime information and pilgrimage point data from the anitabi API.
Also downloads anime frame images.

Endpoints used:
  GET https://api.anitabi.cn/bangumi/{id}/lite
  GET https://api.anitabi.cn/bangumi/{id}/points/detail?haveImage=true
  GET https://image.anitabi.cn/{image_path}?plan=h360
"""

import os
import json
import time
import logging
from pathlib import Path

import requests
from tqdm import tqdm

log = logging.getLogger(__name__)


class AnitabiFetcher:
    def __init__(self, config):
        self.base      = config['api']['anitabi_base']
        self.img_base  = config['api']['image_base']
        self.delay     = config['api']['request_delay']
        self.retries   = config['api']['max_retries']
        self.timeout   = config['api']['timeout']

        self.raw_dir    = Path(config['paths']['raw']['anime_sites'])
        self.frames_dir = Path(config['paths']['raw']['anime_frames'])
        self.anime_ids  = config['demo']['anime_ids']
        self.max_sites  = config['demo']['max_sites_per_anime']
        self.max_frames = config['demo']['max_frames_per_anime']

        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'anime-pilgrimage-research/1.0'})

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def fetch_all(self):
        """Fetch all demo anime and return a flat list of point dicts."""
        all_points = []
        for aid in self.anime_ids:
            log.info(f"[anitabi] Fetching anime {aid} …")
            points = self._fetch_one_anime(aid)
            if points:
                all_points.extend(points)
                log.info(f"[anitabi] {aid}: {len(points)} points collected")
            else:
                log.warning(f"[anitabi] {aid}: no data returned, skipping")
        log.info(f"[anitabi] Total raw points: {len(all_points)}")
        return all_points

    # ------------------------------------------------------------------
    # Per-anime fetch
    # ------------------------------------------------------------------
    def _fetch_one_anime(self, subject_id):
        # 1) Lite info (title, city, overall counts)
        lite = self._get(f"/bangumi/{subject_id}/lite")
        if lite is None:
            return []

        self._save_json(lite, self.raw_dir / f"{subject_id}_lite.json")

        anime_meta = self._parse_lite(lite, subject_id)

        # 2) Detailed points
        points_raw = self._get(
            f"/bangumi/{subject_id}/points/detail",
            params={"haveImage": "true"}
        )
        if points_raw is None:
            return []

        self._save_json(points_raw, self.raw_dir / f"{subject_id}_points.json")

        # 3) Parse points and cap at max_sites
        points = self._parse_points(points_raw, anime_meta)
        points = points[: self.max_sites]

        # 4) Download anime frame images
        self._download_frames(points, subject_id)

        return points

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
    def _parse_lite(self, data, subject_id):
        """Extract consistent metadata from the lite response.

        Actual response schema (confirmed from API):
          { id, cn, title, city, cover, color, geo:[lat,lon], zoom,
            modified, litePoints:[...] }
        """
        if isinstance(data, list):
            data = data[0] if data else {}

        title_cn = data.get('cn') or data.get('name') or str(subject_id)
        title_jp = data.get('title') or title_cn
        city     = data.get('city') or ''

        # geo is [lat, lon] in anitabi
        geo = data.get('geo') or []
        city_lat = float(geo[0]) if len(geo) >= 2 else None
        city_lon = float(geo[1]) if len(geo) >= 2 else None

        return {
            'anime_id':      str(subject_id),
            'anime_title':   title_cn,
            'anime_title_jp': title_jp,
            'city':          city,
            'city_lat':      city_lat,
            'city_lon':      city_lon,
        }

    def _parse_points(self, data, anime_meta):
        """Parse raw points list into normalized dicts."""
        if not isinstance(data, list):
            # Some responses wrap in a key
            data = data.get('points') or data.get('data') or []

        results = []
        for i, pt in enumerate(data):
            parsed = self._parse_single_point(pt, anime_meta, i)
            if parsed:
                results.append(parsed)
        return results

    def _parse_single_point(self, pt, anime_meta, idx):
        """Normalize a single point record.

        Actual response schema (confirmed from API):
          { id, cn, name, image (full URL), ep, s, geo:[lat,lon],
            origin, originURL }
        """
        if not isinstance(pt, dict):
            return None

        # --- coordinates: geo is [lat, lon] ---
        geo = pt.get('geo') or []
        if len(geo) < 2:
            return None
        lat, lon = float(geo[0]), float(geo[1])

        # Basic Japan sanity check
        if not (24 <= lat <= 46 and 122 <= lon <= 154):
            return None

        # --- image: full URL already, swap plan to h360 for better quality ---
        image_url = pt.get('image') or ''
        image_url_hq = image_url.replace('?plan=h160', '?plan=h360') if image_url else ''

        scene_id = str(pt.get('id', f"{anime_meta['anime_id']}_{idx:04d}"))
        name     = pt.get('name') or pt.get('cn') or ''

        local_frame = ''
        if image_url_hq:
            # Derive a clean filename from scene_id
            local_frame = f"data/raw/anime_frames/{anime_meta['anime_id']}/{scene_id}.jpg"

        return {
            'scene_id':       scene_id,
            'anime_id':       anime_meta['anime_id'],
            'anime_title':    anime_meta['anime_title'],
            'episode':        str(pt.get('ep', '')),
            'lat':            lat,
            'lon':            lon,
            'location_name':  name,
            'city':           anime_meta['city'],
            'prefecture':     '',
            'country':        'Japan',
            'place_type':     _infer_place_type(name),
            'image_url_remote': image_url_hq,   # full URL for download
            'anime_frame_path': local_frame,
            'source_url':     pt.get('originURL') or pt.get('origin') or '',
            'source_type':    'anitabi_fan_site',
            'coordinate_confidence': 'high',
            'private_or_sensitive_location': _is_sensitive(name),
            'notes': '',
        }

    # ------------------------------------------------------------------
    # Image download
    # ------------------------------------------------------------------
    def _download_frames(self, points, subject_id):
        save_dir = self.frames_dir / str(subject_id)
        save_dir.mkdir(parents=True, exist_ok=True)

        to_download = [p for p in points if p.get('image_url_remote')]
        to_download = to_download[: self.max_frames]

        for pt in tqdm(to_download, desc=f"  frames/{subject_id}", leave=False):
            local_path = Path(pt['anime_frame_path'])
            local_path.parent.mkdir(parents=True, exist_ok=True)

            if local_path.exists():
                continue

            # image_url_remote is already a full URL
            url = pt['image_url_remote']
            ok  = self._download_file(url, local_path)
            if not ok:
                pt['anime_frame_path'] = ''

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------
    def _get(self, path, params=None):
        url = self.base + path
        for attempt in range(self.retries):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                time.sleep(self.delay)
                if resp.status_code == 200:
                    return resp.json()
                log.warning(f"  HTTP {resp.status_code} for {url}")
                return None
            except Exception as e:
                log.warning(f"  Attempt {attempt+1} failed: {e}")
                time.sleep(self.delay * 2)
        return None

    def _download_file(self, url, save_path):
        try:
            resp = self.session.get(url, timeout=self.timeout, stream=True)
            time.sleep(0.3)
            if resp.status_code == 200:
                with open(save_path, 'wb') as f:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)
                return True
        except Exception as e:
            log.warning(f"  Download failed {url}: {e}")
        return False

    @staticmethod
    def _save_json(data, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _is_japan(lat, lon):
    return 24 <= lat <= 46 and 122 <= lon <= 154


# Keywords for automatic place-type inference from Japanese location names
_PLACE_TYPE_RULES = [
    ('station',         ['駅']),
    ('shrine',          ['神社', '大社', '稲荷']),
    ('temple',          ['寺', '寺院', '堂']),
    ('school',          ['学校', '高校', '中学', '小学', '大学']),
    ('bridge',          ['橋', '陸橋']),
    ('coastal',         ['海', '浜', '岸', '港', '砂浜']),
    ('river',           ['川', '河', '堤']),
    ('hill_viewpoint',  ['坂', '丘', '山', '展望']),
    ('shopping_street', ['商店街', '商業', '市場']),
    ('park',            ['公園', '広場']),
    ('residential',     ['住宅', '団地', 'アパート', 'マンション']),
    ('intersection',    ['交差点', '角']),
]

def _infer_place_type(name):
    if not name:
        return 'unknown'
    for ptype, keywords in _PLACE_TYPE_RULES:
        for kw in keywords:
            if kw in name:
                return ptype
    return 'other'


_SENSITIVE_KEYWORDS = ['学校', '高校', '中学', '小学', '大学',
                       '住宅', '団地', 'アパート', 'マンション', '幼稚園']

def _is_sensitive(name):
    if not name:
        return 0
    return int(any(kw in name for kw in _SENSITIVE_KEYWORDS))
