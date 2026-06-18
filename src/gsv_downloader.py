"""
Google Street View downloader for specific pilgrimage site coordinates.

Adapted from /workplace/SVI-download-pano/GSVdownload_maoran.py
Original by Fan Zhang; adapted for discrete-point sampling in this project.

Uses the unofficial GSV metadata API (no API key required) and the
thumbnail endpoint for image download.

Usage:
    from src.gsv_downloader import download_gsv_for_sites
    download_gsv_for_sites(site_df, config)
"""

import os
import re
import time
import datetime
import logging
from pathlib import Path

import requests
import pandas as pd
from tqdm import tqdm
from PIL import Image
from io import BytesIO

log = logging.getLogger(__name__)

# Unofficial GSV endpoints (no API key needed)
_PANOID_URL = (
    "https://maps.googleapis.com/maps/api/js/GeoPhotoService.SingleImageSearch"
    "?pb=!1m5!1sapiv3!5sUS!11m2!1m1!1b0!2m4!1m2!3d{lat}!4d{lon}"
    "!2d50!3m10!2m2!1sen!2sGB!9m1!1e2!11m4!1m3!1e2!2b1!3e2"
    "!4m10!1e1!1e2!1e3!1e4!1e8!1e6!5m1!1e2!6m1!1e2&callback=_xdc_._v2mub5"
)
_THUMB_URL = (
    "https://geo0.ggpht.com/cbk"
    "?cb_client=maps_sv.tactile&authuser=0&hl=en&output=thumbnail&nbt"
    "&w={w}&h={h}&yaw={heading}&panoid={panoid}&thumbfov={fov}&pitch={pitch}"
)


def download_gsv_for_sites(site_df, config):
    """
    Download GSV images for anime pilgrimage sites.

    Parameters
    ----------
    site_df : pd.DataFrame   anime_site_table (must have site_id, lat, lon columns)
    config  : dict

    Returns
    -------
    pd.DataFrame   updated site_df with streetview_image_path filled
    """
    if not config['gsv']['enabled']:
        log.info("GSV download disabled in config (gsv.enabled=false). Skipping.")
        return site_df

    save_dir     = Path(config['paths']['raw']['streetview'])
    headings     = config['gsv']['headings']       # [0, 90, 180, 270]
    delay        = config['gsv']['request_delay']
    max_sites    = config['gsv']['max_sites']
    w, h         = config['gsv']['image_size'].split('x')
    w, h         = int(w), int(h)

    subset = site_df.head(max_sites).copy()
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'})

    for idx, row in tqdm(subset.iterrows(), total=len(subset), desc="GSV download"):
        site_id = row['site_id']
        lat, lon = float(row['lat']), float(row['lon'])

        site_dir = save_dir / site_id
        site_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: get panoid
        panoid = _get_latest_panoid(session, lat, lon)
        if panoid is None:
            log.warning(f"  {site_id}: no panoid found at ({lat:.4f},{lon:.4f})")
            continue

        # Step 2: download images for each heading
        downloaded = []
        for heading in headings:
            fpath = site_dir / f"{panoid}_{heading}.jpg"
            if fpath.exists():
                downloaded.append(str(fpath))
                continue
            ok = _download_thumb(session, panoid, heading, w, h, fpath, delay)
            if ok:
                downloaded.append(str(fpath))

        if downloaded:
            # Store the 0° image as the primary reference
            site_df.at[idx, 'streetview_image_path'] = downloaded[0]
            log.info(f"  {site_id}: {len(downloaded)} GSV images saved")
        time.sleep(delay)

    return site_df


# ---------------------------------------------------------------------------
# Panoid retrieval
# ---------------------------------------------------------------------------

def _get_latest_panoid(session, lat, lon, retries=3):
    """Retrieve the latest available panoid near (lat, lon)."""
    url = _PANOID_URL.format(lat=lat, lon=lon)
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=10)
            if resp.status_code != 200:
                return None
            return _parse_latest_panoid(resp.text)
        except Exception as e:
            log.debug(f"  panoid attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return None


def _parse_latest_panoid(text):
    """
    Extract the most recent panoid from the GSV metadata response.
    Returns the panoid string, or None if not found.
    """
    pans = re.findall(
        r'\[[0-9]+,"(.+?)"\].+?\[\[null,null,(-?[0-9]+\.[0-9]+),(-?[0-9]+\.[0-9]+)',
        text
    )
    if not pans:
        return None

    pans = [{'panoid': p[0], 'lat': float(p[1]), 'lon': float(p[2])} for p in pans]

    # Remove duplicate first entry
    if len(pans) > 1 and pans[0] == pans[1]:
        pans = pans[1:]

    # Get dates
    dates = re.findall(r'([0-9]{0,3})?,?\[(20[0-9]{2}),([0-9]{1,2})\]', text)
    dates = [d for d in dates if 1 <= int(d[2]) <= 12]

    if dates and len(pans) >= len(dates):
        # Assign dates to panoramas
        for i, (_, year, month) in enumerate(dates):
            if i < len(pans):
                pans[i]['year']  = int(year)
                pans[i]['month'] = int(month)

    # Pick the latest dated panorama
    dated = [p for p in pans if 'year' in p]
    if dated:
        dated.sort(key=lambda p: datetime.datetime(p['year'], p['month'], 1))
        return dated[-1]['panoid']

    return pans[0]['panoid'] if pans else None


# ---------------------------------------------------------------------------
# Image download
# ---------------------------------------------------------------------------

def _download_thumb(session, panoid, heading, w, h, save_path, delay):
    url = _THUMB_URL.format(
        panoid=panoid, heading=heading,
        fov=90, pitch=0, w=w, h=h
    )
    try:
        resp = session.get(url, timeout=15, stream=True)
        time.sleep(delay)
        if resp.status_code == 200:
            img = Image.open(BytesIO(resp.content))
            img.save(str(save_path), 'JPEG')
            return True
    except Exception as e:
        log.debug(f"  thumb download failed ({panoid}, {heading}°): {e}")
    return False
