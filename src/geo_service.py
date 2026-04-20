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
    NOMINATIM_USER_AGENT,
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


@lru_cache(maxsize=20000)
def geocode_city(city: str, country: str) -> Tuple[float, float]:
    if not city or not country:
        return np.nan, np.nan

    normalized_country = normalize_country_name(country)
    key = f"{city.strip().lower()}|{normalized_country.strip().lower()}"
    if key in geocode_cache:
        lat, lon = geocode_cache[key]
        return float(lat), float(lon)

    if OFFLINE:
        lat, lon = COUNTRY_COORDS_FALLBACK.get(normalized_country, (np.nan, np.nan))
    else:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"city": city, "country": normalized_country, "format": "json", "limit": 1}
        headers = {"User-Agent": NOMINATIM_USER_AGENT}
        try:
            sleep(1)
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data:
                lat = float(data[0]["lat"])
                lon = float(data[0]["lon"])
            else:
                lat, lon = COUNTRY_COORDS_FALLBACK.get(normalized_country, (np.nan, np.nan))
        except Exception as exc:
            logger.error(f"Geocode error for {city}, {normalized_country}: {exc}")
            lat, lon = COUNTRY_COORDS_FALLBACK.get(normalized_country, (np.nan, np.nan))

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
