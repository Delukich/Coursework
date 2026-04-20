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
                logger.info(f"Завантажено {len(data)} записів погоди з кешу")
                return data
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Пошкоджений кеш погоди ({e}), починаємо з нуля")
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
        logger.error(f"Помилка збереження кешу погоди: {e}")


weather_cache = load_weather_cache()


class WeatherService:

    API_URL = "https://archive-api.open-meteo.com/v1/archive"
    FALLBACK = {'temp': 15.0, 'rain': 0.0}

    def get_real_weather(self, lat: float, lon: float, date_obj: date) -> Dict[str, float]:
        if np.isnan(lat) or np.isnan(lon):
            return self.FALLBACK.copy()

        key = f"{lat:.4f}_{lon:.4f}_{date_obj.isoformat()}"

        if key in weather_cache:
            return weather_cache[key]

        if OFFLINE:
            result = self.FALLBACK.copy()
        else:
            result = self._fetch_from_api(lat, lon, date_obj)

        weather_cache[key] = result
        save_weather_cache(weather_cache)
        return result

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
            logger.error(f"Таймаут запиту погоди для ({lat:.2f}, {lon:.2f})")
        except requests.exceptions.RequestException as e:
            logger.error(f"Помилка запиту погоди: {e}")
        except Exception as e:
            logger.error(f"Несподівана помилка погоди: {e}")
        return self.FALLBACK.copy()