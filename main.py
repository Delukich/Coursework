import os
import traceback
import warnings
from datetime import datetime
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.config import (
    COUNTRY_COORDS_FALLBACK,
    CATEGORICAL_COLUMNS_USED,
    SHIPPING_MODE_COST,
    SHIPPING_MODE_RANK,
    MODEL_PATH,
    FEATURE_META_PATH,
    ENRICHED_OUTPUT_PATH,
    RAW_DATA_PATH,
    normalize_country_name,
)
from src.economics_service import EconomicsService
from src.geo_service import geocode_city, haversine_distance
from src.pipeline import DataPipeline
from src.weather_service import WeatherService

warnings.filterwarnings("ignore")

CURRENT_SCENARIO_YEAR = 2026


def is_offline_mode() -> bool:
    return os.getenv("LOGISTICS_OFFLINE", "1").strip().lower() not in {"0", "false", "no"}


MARKET_BY_COUNTRY = {
    "USA": "USCA",
    "Canada": "USCA",
    "Mexico": "LATAM",
    "France": "Europe",
    "Germany": "Europe",
    "UK": "Europe",
    "Italy": "Europe",
    "Spain": "Europe",
    "Netherlands": "Europe",
    "Belgium": "Europe",
    "Sweden": "Europe",
    "Norway": "Europe",
    "Poland": "Europe",
    "Romania": "Europe",
    "Austria": "Europe",
    "Ireland": "Europe",
    "Finland": "Europe",
    "Portugal": "Europe",
    "Slovakia": "Europe",
    "Bulgaria": "Europe",
    "Brazil": "LATAM",
    "Argentina": "LATAM",
    "Colombia": "LATAM",
    "Chile": "LATAM",
    "Peru": "LATAM",
    "Venezuela": "LATAM",
    "Ecuador": "LATAM",
    "Bolivia": "LATAM",
    "Uruguay": "LATAM",
    "Panama": "LATAM",
    "Honduras": "LATAM",
    "Guatemala": "LATAM",
    "El Salvador": "LATAM",
    "Nicaragua": "LATAM",
    "Cuba": "LATAM",
    "Dominican Republic": "LATAM",
    "Jamaica": "LATAM",
    "China": "Pacific Asia",
    "Japan": "Pacific Asia",
    "South Korea": "Pacific Asia",
    "Australia": "Pacific Asia",
    "New Zealand": "Pacific Asia",
    "India": "Pacific Asia",
    "Indonesia": "Pacific Asia",
    "Philippines": "Pacific Asia",
    "Thailand": "Pacific Asia",
    "Vietnam": "Pacific Asia",
    "Malaysia": "Pacific Asia",
    "Singapore": "Pacific Asia",
    "Bangladesh": "Pacific Asia",
    "Pakistan": "Pacific Asia",
    "Nigeria": "Africa",
    "South Africa": "Africa",
    "Egypt": "Africa",
    "Algeria": "Africa",
    "Morocco": "Africa",
    "Sudan": "Africa",
    "Ivory Coast": "Africa",
    "Ghana": "Africa",
    "Senegal": "Africa",
    "Ethiopia": "Africa",
    "Tanzania": "Africa",
    "Kenya": "Africa",
    "Angola": "Africa",
}

REGION_BY_MARKET = {
    "Europe": "Western Europe",
    "LATAM": "South America",
    "Pacific Asia": "Southeast Asia",
    "Africa": "West Africa",
    "USCA": "US Center",
}

DEPARTMENT_KEYWORDS = (
    ("Technology", ("computer", "electronic", "camera", "video game")),
    ("Book Shop", ("book",)),
    ("Pet Shop", ("pet",)),
    ("Health and Beauty ", ("health", "beauty")),
    ("Golf", ("golf",)),
    ("Apparel", ("apparel", "clothing", "footwear", "cleats")),
    ("Outdoors", ("fishing", "camping", "hiking", "outdoor", "sport")),
)


def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Модель не знайдена: {MODEL_PATH}\n"
            "Спочатку виконайте опцію 1 (папелаплаін) – опцію 2 (навчання)."
        )
    if not os.path.exists(FEATURE_META_PATH):
        raise FileNotFoundError(f"Метадані моделі не знайдені: {FEATURE_META_PATH}")

    model = joblib.load(MODEL_PATH)
    feature_meta = joblib.load(FEATURE_META_PATH)
    return {"model": model, "feature_meta": feature_meta}


def format_input_for_xgboost(df_input: pd.DataFrame, model_features: list, feature_meta: dict):
    df = df_input.copy()

    if model_features is not None:
        medians: dict = feature_meta.get("medians", {})
        for col in model_features:
            if col not in df.columns:
                df[col] = medians.get(col, 0.0)

    for col in CATEGORICAL_COLUMNS_USED:
        if col in df.columns:
            df[col] = df[col].astype("category")

    if model_features is not None:
        available = [feature for feature in model_features if feature in df.columns]
        return df[available]
    return df


def infer_department(category: str) -> str:
    category_lower = str(category).strip().lower()
    for department, keywords in DEPARTMENT_KEYWORDS:
        if any(token in category_lower for token in keywords):
            return department
    return "Outdoors"


def infer_market(country: str) -> str:
    return MARKET_BY_COUNTRY.get(country, "Europe")


def infer_region(country: str, market: str) -> str:
    if country == "UK":
        return "Northern Europe"
    if country in {"Italy", "Spain", "Portugal"}:
        return "Southern Europe"
    if country in {"Mexico", "Guatemala", "Honduras", "El Salvador", "Nicaragua", "Panama"}:
        return "Central America"
    if country in {"Dominican Republic", "Cuba", "Jamaica"}:
        return "Caribbean"
    if country in {"China", "Japan", "South Korea"}:
        return "Eastern Asia"
    if country in {"India", "Pakistan", "Bangladesh"}:
        return "South Asia"
    return REGION_BY_MARKET.get(market, "Western Europe")


@lru_cache(maxsize=1)
def load_reference_mappings() -> tuple[dict, dict, dict]:
    if not os.path.exists(RAW_DATA_PATH):
        return {}, {}, {}

    ref = pd.read_csv(
        RAW_DATA_PATH,
        encoding="latin1",
        low_memory=False,
        usecols=["Order Country", "Order City", "Market", "Order Region"],
    ).copy()
    ref["Order Country"] = ref["Order Country"].map(normalize_country_name)
    ref["Order City"] = ref["Order City"].astype(str).str.strip()

    def most_common(series: pd.Series):
        mode = series.mode()
        return mode.iloc[0] if not mode.empty else series.iloc[0]

    market_by_country = (
        ref.groupby("Order Country")["Market"]
        .agg(most_common)
        .to_dict()
    )
    region_by_country = (
        ref.groupby("Order Country")["Order Region"]
        .agg(most_common)
        .to_dict()
    )
    region_by_city_country = (
        ref.groupby(["Order Country", "Order City"])["Order Region"]
        .agg(most_common)
        .to_dict()
    )
    return market_by_country, region_by_country, region_by_city_country


def infer_market_and_region(country: str, city: str) -> tuple[str, str]:
    market_by_country, region_by_country, region_by_city_country = load_reference_mappings()

    market = market_by_country.get(country, infer_market(country))
    region = region_by_city_country.get((country, city.strip()), region_by_country.get(country))
    if not region:
        region = infer_region(country, market)
    return market, region


def build_input_row(
    econ_svc: EconomicsService,
    weather_svc: WeatherService,
    country: str,
    city: str,
    shipping_mode: str,
    category: str,
    qty: int,
    price: float,
    discount_rate: float,
    store_lat: float,
    store_lon: float,
    sched_days: int,
):
    now = datetime.now()
    resolved_country = normalize_country_name(country)
    stats = econ_svc.get_country_stats(resolved_country, year=CURRENT_SCENARIO_YEAR)
    fuel_price = econ_svc.get_fuel_price(CURRENT_SCENARIO_YEAR, now.month)

    dest_lat, dest_lon = _resolve_destination_coords(resolved_country, city)
    dist_km = _resolve_distance_km(store_lat, store_lon, dest_lat, dest_lon)
    temp_c, rain_mm = _resolve_weather(weather_svc, dest_lat, dest_lon, now)
    order_item_total = _calc_order_total(price, qty, discount_rate)
    total_penalty, distance_penalty, weather_penalty, tariff_penalty = _calc_penalties(
        econ_svc,
        stats,
        shipping_mode,
        category,
        order_item_total,
        dist_km,
        fuel_price,
        rain_mm,
    )
    dist_safe = max(float(dist_km), 1.0)
    market, region = infer_market_and_region(resolved_country, city)

    time_features = _calc_time_features(now)
    price_features = _calc_price_features(
        order_item_total,
        discount_rate,
        qty,
        fuel_price,
        dist_safe,
        shipping_mode,
    )

    gdp_ppp = float(stats.get("gdp_ppp", 25000))
    lpi = float(stats.get("lpi", 3.2))

    input_dict = {
        "Type": "DEBIT",
        "Shipping Mode": shipping_mode,
        "Category Name": category,
        "Order Region": region,
        "Order City": city,
        "Order Country": resolved_country,
        "Market": market,
        "Customer Segment": "Consumer",
        "Department Name": infer_department(category),
        "Order Item Quantity": int(qty),
        "Order Item Product Price": float(price),
        "Order Item Discount Rate": float(discount_rate),
        "Latitude": float(store_lat),
        "Longitude": float(store_lon),
        "Distance_KM": float(dist_km),
        "Days for shipment (scheduled)": float(sched_days),
        "Real_Fuel_Price": float(fuel_price),
        "Real_GDP_PPP": gdp_ppp,
        "Real_LPI": lpi,
        "Real_Temp_C": float(temp_c),
        "Real_Rain_mm": float(rain_mm),
        "order_month": time_features["order_month"],
        "is_peak_season": time_features["is_peak_season"],
        "order_day_of_week": time_features["order_day_of_week"],
        "is_weekend_order": time_features["is_weekend_order"],
        "order_quarter": time_features["order_quarter"],
        "price_per_km": float(price_features["price_per_km"]),
        "discount_x_qty": float(price_features["discount_x_qty"]),
        "fuel_x_distance": float(price_features["fuel_x_distance"]),
        "ship_sla_rank": float(price_features["ship_sla_rank"]),
        "ship_cost_per_km": float(price_features["ship_cost_per_km"]),
        "gdp_x_lpi": float(gdp_ppp * lpi),
        "inflation_risk": 1 if float(stats.get("inf", 3.0)) > 5.0 else 0,
        "is_cold": 1 if temp_c < 0 else 0,
        "is_heavy_rain": 1 if rain_mm > 10.0 else 0,
        "dest_lat_abs": abs(float(dest_lat)),
        "cross_hemisphere": 1 if ((store_lat > 0 and dest_lat < 0) or (store_lat < 0 and dest_lat > 0)) else 0,
    }

    ctx = {
        "stats": stats,
        "dist_km": dist_km,
        "temp_c": temp_c,
        "rain_mm": rain_mm,
        "order_item_total": order_item_total,
        "country": resolved_country,
        "city": city,
        "total_penalty": total_penalty,
        "fuel_price": fuel_price,
        "distance_penalty": distance_penalty,
        "weather_penalty": weather_penalty,
        "tariff_penalty": tariff_penalty,
        "market": market,
        "region": region,
    }
    return pd.DataFrame([input_dict]), ctx


def _resolve_destination_coords(country: str, city: str):
    dest_lat, dest_lon = geocode_city(city, country)
    if np.isnan(dest_lat) or np.isnan(dest_lon):
        dest_lat, dest_lon = COUNTRY_COORDS_FALLBACK.get(country, (np.nan, np.nan))
        if np.isnan(dest_lat):
            print(f"  Попередження: координати для {city}, {country} не знайдено. Використовую (0, 0).")
            dest_lat, dest_lon = 0.0, 0.0
    return dest_lat, dest_lon


def _resolve_distance_km(store_lat: float, store_lon: float, dest_lat: float, dest_lon: float) -> float:
    dist_km = haversine_distance(store_lat, store_lon, dest_lat, dest_lon)
    return 5000.0 if np.isnan(dist_km) else dist_km


def _resolve_weather(weather_svc: WeatherService, dest_lat: float, dest_lon: float, now: datetime):
    weather = weather_svc.get_real_weather(dest_lat, dest_lon, now.date())
    temp_c = weather["temp"] if not np.isnan(weather["temp"]) else 15.0
    rain_mm = weather["rain"] if not np.isnan(weather["rain"]) else 0.0
    return temp_c, rain_mm


def _calc_order_total(price: float, qty: int, discount_rate: float) -> float:
    return float(price) * int(qty) * (1.0 - float(discount_rate))


def _calc_penalties(
    econ_svc: EconomicsService,
    stats: dict,
    shipping_mode: str,
    category: str,
    order_item_total: float,
    dist_km: float,
    fuel_price: float,
    rain_mm: float,
):
    fuel_multiplier = fuel_price / 60.0
    ship_mult = SHIPPING_MODE_COST.get(shipping_mode, 1.0)
    distance_penalty = (dist_km / 1000.0) * fuel_multiplier * ship_mult
    weather_penalty = 2.0 if rain_mm > 5.0 else 0.0
    tariff_penalty = econ_svc.calculate_tariff(stats, category, order_item_total)
    total_penalty = distance_penalty + weather_penalty + tariff_penalty
    return total_penalty, distance_penalty, weather_penalty, tariff_penalty


def _calc_time_features(now: datetime) -> dict:
    order_day_of_week = int(now.weekday())
    order_quarter = int(((now.month - 1) // 3) + 1)
    return {
        "order_month": int(now.month),
        "order_day_of_week": order_day_of_week,
        "order_quarter": order_quarter,
        "is_weekend_order": 1 if order_day_of_week >= 5 else 0,
        "is_peak_season": 1 if now.month in [10, 11, 12, 1] else 0,
    }


def _calc_price_features(
    order_item_total: float,
    discount_rate: float,
    qty: int,
    fuel_price: float,
    dist_safe: float,
    shipping_mode: str,
):
    ship_mult = SHIPPING_MODE_COST.get(shipping_mode, 1.0)
    return {
        "price_per_km": order_item_total / dist_safe,
        "discount_x_qty": float(discount_rate) * int(qty),
        "fuel_x_distance": (fuel_price / 60.0) * (dist_safe / 1000.0),
        "ship_sla_rank": SHIPPING_MODE_RANK.get(shipping_mode, 3),
        "ship_cost_per_km": ship_mult * fuel_price / dist_safe,
    }


def predict_profitability(df_input: pd.DataFrame, artifacts: dict):
    model = artifacts["model"]
    feature_meta = artifacts["feature_meta"]
    optimal_thresh = feature_meta.get("optimal_threshold", 0.5)

    model_feats = model.feature_names_in_.tolist() if hasattr(model, "feature_names_in_") else None
    df_model = format_input_for_xgboost(df_input, model_feats, feature_meta)

    prob_profit = float(model.predict_proba(df_model)[0][1])
    label = 1 if prob_profit >= optimal_thresh else 0
    return label, prob_profit * 100.0, optimal_thresh * 100.0


def get_float_input(prompt: str, default: float, min_val: float = None, max_val: float = None) -> float:
    while True:
        raw = input(prompt).strip() or str(default)
        try:
            val = float(raw)
            if min_val is not None and val < min_val:
                print(f"  Мінімальне значення: {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"  Максимальне значення: {max_val}")
                continue
            return val
        except ValueError:
            print(f"  Введіть числове значення, наприклад: {default}")


def get_int_input(prompt: str, default: int, min_val: int = 1) -> int:
    while True:
        raw = input(prompt).strip() or str(default)
        try:
            val = int(raw)
            if val < min_val:
                print(f"  Мінімальне значення: {min_val}")
                continue
            return val
        except ValueError:
            print(f"  Введіть ціле число, наприклад: {default}")


def main():
    econ_svc = EconomicsService()
    weather_svc = WeatherService()

    def show_menu():
        print("\n" + "=" * 55)
        print("  LOGISTICS DECISION SUPPORT SYSTEM v3.2")
        print("  Оцінка ризику прибутковості доставки")
        print("=" * 55)
        print("  1. Запустити пайплайн даних")
        print("  2. Навчити модель")
        print("  3. Ручна оцінка прибутковості")
        print("  0. Вихід")
        print("=" * 55)
        print(f"  Режим даних: {'Offline demo mode' if is_offline_mode() else 'Online API mode'}")
        print("=" * 55)

    def run_pipeline_option():
        limit_raw = input("Ліміт рядків (Enter = всі, рекомендовано 1000 для тесту): ").strip()
        limit = int(limit_raw) if limit_raw.isdigit() else None
        print(f"\nОбробка {'всього датасету' if not limit else f'перших {limit} рядків'}...")
        try:
            DataPipeline(mode="train", limit=limit).run()
        except FileNotFoundError as exc:
            print(f"Помилка: {exc}")
        except Exception as exc:
            print(f"Помилка папелайну: {exc}")

    def run_training_option():
        if not os.path.exists(ENRICHED_OUTPUT_PATH):
            print(f"Помилка: файл {ENRICHED_OUTPUT_PATH} не знайдено. Спочатку виконайте опцію 1.")
            return
        try:
            from src.train_model import ProfitabilityClassifier

            df = pd.read_csv(ENRICHED_OUTPUT_PATH, encoding="utf-8", low_memory=False)
            print(f"Завантажено {len(df):,} рядків для навчання.")
            ProfitabilityClassifier().train_evaluate(df)
        except Exception as exc:
            print(f"Помилка навчання: {exc}")
            traceback.print_exc()

    def run_manual_assessment():
        try:
            artifacts = load_artifacts()
        except FileNotFoundError as exc:
            print(f"Помилка: {exc}")
            return

        print("\nПараметри доставки")
        try:
            store_lat = get_float_input("Широта складу (default 40.71): ", 40.71, -90, 90)
            store_lon = get_float_input("Довгота складу (default -74.00): ", -74.00, -180, 180)
            country = input("Країна доставки (default France): ").strip() or "France"
            city = input("Місто доставки (default Paris): ").strip() or "Paris"

            print("Методи: Standard Class | Second Class | First Class | Same Day")
            shipping_mode = input("Метод доставки [default Standard Class]: ").strip() or "Standard Class"
            if shipping_mode not in SHIPPING_MODE_COST:
                print("  Невідомий метод, використовую Standard Class.")
                shipping_mode = "Standard Class"

            sched_days = get_int_input("Плановий термін доставки, днів (default 4): ", 4)
            category = input("Категорія товару (default Electronics): ").strip() or "Electronics"
            qty = get_int_input("Кількість одиниць (default 1): ", 1)
            price = get_float_input("Ціна за одиницю, USD (default 500.0): ", 500.0, 0.01)
            discount_rate = get_float_input("Знижка від 0 до 1 (default 0.1): ", 0.1, 0.0, 1.0)
        except (KeyboardInterrupt, EOFError):
            print("\nСкасовано.")
            return

        print("\nРозраховую...")
        try:
            df_input, ctx = build_input_row(
                econ_svc,
                weather_svc,
                country,
                city,
                shipping_mode,
                category,
                qty,
                price,
                discount_rate,
                store_lat,
                store_lon,
                sched_days,
            )
            is_profitable, prob_profit, thresh = predict_profitability(df_input, artifacts)
        except Exception as exc:
            print(f"Помилка передбачення: {exc}")
            traceback.print_exc()
            return

        status = "ПРИБУТКОВО" if is_profitable else "ЗБИТКОВО (РИЗИК)"

        print("\n" + "-" * 65)
        print(f"  Маршрут          : Склад -> {ctx['city']}, {ctx['country']}")
        print(f"  Регіон / ринок   : {ctx['region']} / {ctx['market']}")
        print(f"  Відстань         : {ctx['dist_km']:,.0f} км")
        print(f"  Сума замовлення  : ${ctx['order_item_total']:,.2f}")
        print(f"  Паливо (Brent)   : ${ctx['fuel_price']:.1f}/bbl")
        print(f"  Температура/дощ  : {ctx['temp_c']:.1f} C / {ctx['rain_mm']:.1f} мм")
        print("-" * 65)
        print(f"  Логістика -$     : {ctx['distance_penalty']:,.2f}")
        print(f"  Погода   -$      : {ctx['weather_penalty']:,.2f}")
        print(f"  Мито     -$      : {ctx['tariff_penalty']:,.2f}")
        print(f"  Разом витрат     : -${ctx['total_penalty']:,.2f}")
        print("-" * 65)
        print(f"  ВЕРДИКТ          : {status}")
        print(f"  Ймовірність      : {prob_profit:.1f}% (поріг: {thresh:.1f}%)")
        print("  Note             : Оцінка для підтримки рішення, не гарантія прибутку")
        print("-" * 65 + "\n")

    while True:
        show_menu()

        choice = input("Оберіть опцію (0-3): ").strip()

        if choice == "1":
            run_pipeline_option()

        elif choice == "2":
            run_training_option()

        elif choice == "3":
            run_manual_assessment()

        elif choice == "0":
            print("До побачення.")
            break
        else:
            print("Невідома опція. Введіть число від 0 до 3.")


if __name__ == "__main__":
    main()
