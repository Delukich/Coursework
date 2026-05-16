import json
import logging
import os
from functools import lru_cache
from time import sleep
from typing import Tuple

import numpy as np
import requests

from src.config import (
    CACHE_DIR,
    COUNTRY_COORDS_FALLBACK,
    GEOCODING_USER_AGENT,
    OFFLINE,
    normalize_country_name,
)

logger = logging.getLogger(__name__)
GEOCODE_CACHE_FILE = os.path.join(CACHE_DIR, "geocode_cache.json")


def load_geocode_cache() -> dict:
    if os.path.exists(GEOCODE_CACHE_FILE):
        try:
            with open(GEOCODE_CACHE_FILE, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception as exc:
            logger.error(f"Failed to read geocode cache: {exc}")
    return {}


def save_geocode_cache(cache: dict):
    tmp_path = GEOCODE_CACHE_FILE + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as file:
            json.dump(cache, file, ensure_ascii=False, indent=2)
        os.replace(tmp_path, GEOCODE_CACHE_FILE)
    except Exception as exc:
        logger.error(f"Failed to save geocode cache: {exc}")


geocode_cache = load_geocode_cache()


def _fallback_coords(country: str) -> Tuple[float, float]:
    return COUNTRY_COORDS_FALLBACK.get(country, (np.nan, np.nan))


def _coords_match(a: Tuple[float, float], b: Tuple[float, float], tol: float = 1e-4) -> bool:
    if any(np.isnan(x) for x in [a[0], a[1], b[0], b[1]]):
        return False
    return abs(float(a[0]) - float(b[0])) <= tol and abs(float(a[1]) - float(b[1])) <= tol


def _query_nominatim(city: str, country: str) -> Tuple[float, float]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"city": city, "country": country, "format": "json", "limit": 1}
    headers = {"User-Agent": GEOCODING_USER_AGENT}
    try:
        sleep(1)
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as exc:
        logger.error(f"Geocode error for {city}, {country}: {exc}")
    return _fallback_coords(country)


@lru_cache(maxsize=20000)
def geocode_city(city: str, country: str, force_live: bool = False) -> Tuple[float, float]:
    if not city or not country:
        return np.nan, np.nan

    normalized_country = normalize_country_name(country)
    key = f"{city.strip().lower()}|{normalized_country.strip().lower()}"
    fallback_coords = _fallback_coords(normalized_country)

    if key in geocode_cache:
        lat, lon = geocode_cache[key]
        cached_coords = (float(lat), float(lon))
        # In manual mode we may want to replace coarse country-level cache entries
        # with exact city coordinates when network access is available.
        if not force_live or not _coords_match(cached_coords, fallback_coords):
            return cached_coords

    if OFFLINE and not force_live:
        lat, lon = _fallback_coords(normalized_country)
    else:
        lat, lon = _query_nominatim(city, normalized_country)
        if any(np.isnan(x) for x in [lat, lon]):
            lat, lon = fallback_coords

    geocode_cache[key] = (lat, lon)
    save_geocode_cache(geocode_cache)
    return float(lat), float(lon)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    if any(np.isnan(x) for x in [lat1, lon1, lat2, lon2]):
        return np.nan

    radius_km = 6371.0
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return radius_km * c
