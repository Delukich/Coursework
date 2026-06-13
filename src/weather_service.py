import json
import os
import logging
import numpy as np
from typing import Dict
import requests
from datetime import date
from src.config import CACHE_DIR, OFFLINE

logger = logging.getLogger(__name__)
WEATHER_CACHE_FILE = os.path.join(CACHE_DIR, "weather_cache.json")


def load_weather_cache() -> dict:
    if os.path.exists(WEATHER_CACHE_FILE):
        try:
            with open(WEATHER_CACHE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Loaded {len(data)} weather records from cache")
                return data
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Weather cache is corrupted ({e}); starting from scratch")
            try:
                os.rename(WEATHER_CACHE_FILE, WEATHER_CACHE_FILE + ".bak")
            except OSError:
                pass
    return {}


def save_weather_cache(cache: dict):
    tmp_path = WEATHER_CACHE_FILE + ".tmp"
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, WEATHER_CACHE_FILE)
    except Exception as e:
        logger.error(f"Failed to save weather cache: {e}")


weather_cache = load_weather_cache()


class WeatherService:

    API_URL = "https://archive-api.open-meteo.com/v1/archive"
    FALLBACK = {'temp': 15.0, 'rain': 0.0}

    def get_real_weather(self, lat: float, lon: float, date_obj: date, force_live: bool = False) -> Dict[str, float]:
        if np.isnan(lat) or np.isnan(lon):
            return self.FALLBACK.copy()

        key = f"{lat:.4f}_{lon:.4f}_{date_obj.isoformat()}"

        if key in weather_cache:
            cached = weather_cache[key]
            if force_live and cached == self.FALLBACK:
                cached = self._fetch_from_api(lat, lon, date_obj)
                if cached == self.FALLBACK:
                    cached = self._estimate_climate(lat, date_obj)
                weather_cache[key] = cached
                save_weather_cache(weather_cache)
            elif OFFLINE and cached == self.FALLBACK:
                cached = self._estimate_climate(lat, date_obj)
                weather_cache[key] = cached
                save_weather_cache(weather_cache)
            return cached

        result = self._get_weather_result(lat, lon, date_obj, force_live=force_live)

        weather_cache[key] = result
        save_weather_cache(weather_cache)
        return result

    def _get_weather_result(self, lat: float, lon: float, date_obj: date, force_live: bool = False) -> Dict[str, float]:
        if OFFLINE and not force_live:
            return self._estimate_climate(lat, date_obj)
        return self._fetch_from_api(lat, lon, date_obj)

    def _estimate_climate(self, lat: float, date_obj: date) -> Dict[str, float]:
        abs_lat = abs(float(lat))
        month = int(date_obj.month)

        mean_temp = max(-2.0, 28.0 - 0.42 * abs_lat)
        temp_amplitude = min(18.0, 2.0 + 0.22 * abs_lat)

        peak_month = 7 if lat >= 0 else 1
        seasonal_phase = np.cos((month - peak_month) * 2.0 * np.pi / 12.0)
        temp_c = mean_temp + temp_amplitude * seasonal_phase

        if abs_lat < 15:
            mean_rain = 5.5
            rain_amplitude = 2.5
        elif abs_lat < 30:
            mean_rain = 3.0
            rain_amplitude = 1.8
        else:
            mean_rain = 1.4
            rain_amplitude = 1.0

        wet_peak_month = 8 if lat >= 0 else 2
        wet_phase = np.cos((month - wet_peak_month) * 2.0 * np.pi / 12.0)
        rain_mm = max(0.0, mean_rain + rain_amplitude * wet_phase)

        return {
            'temp': round(float(temp_c), 1),
            'rain': round(float(rain_mm), 1),
        }

    def _fetch_from_api(self, lat: float, lon: float, date_obj: date) -> Dict[str, float]:
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": date_obj.isoformat(),
            "end_date": date_obj.isoformat(),
            "daily": "temperature_2m_mean,rain_sum",
            "timezone": "auto"
        }
        try:
            resp = requests.get(self.API_URL, params=params, timeout=10)
            resp.raise_for_status()
            daily = resp.json().get("daily", {})
            temp_list = daily.get("temperature_2m_mean", [None])
            rain_list = daily.get("rain_sum", [None])
            temp = temp_list[0] if temp_list and temp_list[0] is not None else np.nan
            rain = rain_list[0] if rain_list and rain_list[0] is not None else np.nan
            return {
                'temp': float(temp) if not np.isnan(temp) else self.FALLBACK['temp'],
                'rain': float(rain) if not np.isnan(rain) else self.FALLBACK['rain'],
            }
        except requests.exceptions.Timeout:
            logger.error(f"Weather request timed out for ({lat:.2f}, {lon:.2f})")
        except requests.exceptions.RequestException as e:
            logger.error(f"Weather request error: {e}")
        except Exception as e:
            logger.error(f"Unexpected weather error: {e}")
        return self._estimate_climate(lat, date_obj)
