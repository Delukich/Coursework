import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import warnings

from src.weather_service import WeatherService
from src.economics_service import EconomicsService
from src.geo_service import haversine_distance, geocode_city
from src.config import (
    RAW_DATA_PATH,
    ENRICHED_OUTPUT_PATH,
    COUNTRY_COORDS_FALLBACK,
    normalize_country_name,
)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

FEATURES_TO_KEEP = [
    'Type', 'Shipping Mode', 'Category Name', 'Order Region',
    'Order City', 'Order Country', 'Market', 'Customer Segment', 'Department Name',

    'Order Item Quantity',
    'Order Item Product Price',
    'Order Item Discount Rate',

    'Latitude', 'Longitude',
    'Days for shipment (scheduled)',
    'Distance_KM',

    'Real_Fuel_Price',
    'Real_GDP_PPP',
    'Real_LPI',
    'Real_Temp_C',
    'Real_Rain_mm',

    'order_month',
    'is_peak_season',

    'order_day_of_week',
    'is_weekend_order',
    'order_quarter',

    'price_per_km',
    'discount_x_qty',
    'fuel_x_distance',
    'ship_sla_rank',
    'ship_cost_per_km',

    'gdp_x_lpi',
    'inflation_risk',

    'is_cold',
    'is_heavy_rain',

    'dest_lat_abs',
    'cross_hemisphere',

    'Is_Profitable',
]

SHIP_COST_MULT = {
    'Same Day': 3.0,
    'First Class': 2.0,
    'Second Class': 1.2,
    'Standard Class': 1.0,
}


class DataPipeline:
    def __init__(self, file_path=None, output_path=None, mode='train', limit=None):
        self.file_path = file_path or RAW_DATA_PATH
        self.output_path = output_path or ENRICHED_OUTPUT_PATH
        self.mode = mode
        self.year = 2018 if mode == 'train' else 2026
        self.limit = limit
        self.weather_svc = WeatherService()
        self.econ_svc = EconomicsService()
        logger.info(f"Пайплайн | mode={mode} | year={self.year} | output={self.output_path}")

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['order_date'] = pd.to_datetime(df['order date (DateOrders)'], errors='coerce')
        df['order_month'] = df['order_date'].dt.month.fillna(1).astype(int)
        df['order_quarter'] = df['order_date'].dt.quarter.fillna(1).astype(int)
        df['order_day_of_week'] = df['order_date'].dt.dayofweek.fillna(0).astype(int)
        df['is_peak_season'] = df['order_month'].isin([10, 11, 12, 1]).astype(int)
        df['is_weekend_order'] = (df['order_day_of_week'] >= 5).astype(int)
        return df

    def _normalize_reference_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'Order Country' in df.columns:
            df['Order Country'] = df['Order Country'].map(normalize_country_name)
        return df

    def _add_geo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("[2/6] Геокодування пунктів призначення...")
        unique_locs = df[['Order City', 'Order Country']].drop_duplicates()
        dest_coords: dict = {}

        for _, row in tqdm(unique_locs.iterrows(), total=len(unique_locs), desc="Geocoding"):
            lat, lon = geocode_city(row['Order City'], row['Order Country'])
            if np.isnan(lat) or np.isnan(lon):
                lat, lon = COUNTRY_COORDS_FALLBACK.get(row['Order Country'], (np.nan, np.nan))
            dest_coords[(row['Order City'], row['Order Country'])] = (lat, lon)

        def calc_dist(row):
            dest = dest_coords.get((row['Order City'], row['Order Country']))
            if not dest or pd.isna(row.get('Latitude')) or pd.isna(row.get('Longitude')):
                return 5000.0
            d = haversine_distance(row['Latitude'], row['Longitude'], dest[0], dest[1])
            return d if not np.isnan(d) else 5000.0

        df['Distance_KM'] = df.apply(calc_dist, axis=1)

        df['_dest_lat'] = df.apply(
            lambda r: dest_coords.get((r['Order City'], r['Order Country']), (np.nan, np.nan))[0], axis=1)
        df['_dest_lon'] = df.apply(
            lambda r: dest_coords.get((r['Order City'], r['Order Country']), (np.nan, np.nan))[1], axis=1)

        df['dest_lat_abs'] = df['_dest_lat'].abs()
        df['cross_hemisphere'] = (
            (df['Latitude'].fillna(0) > 0) & (df['_dest_lat'].fillna(0) < 0) |
            (df['Latitude'].fillna(0) < 0) & (df['_dest_lat'].fillna(0) > 0)
        ).astype(int)

        return df, dest_coords

    def _add_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("[3/6] Погодні дані...")

        def get_weather(row):
            if pd.isna(row['_dest_lat']) or pd.isna(row['_dest_lon']) or pd.isna(row['order_date']):
                return 15.0, 0.0
            w = self.weather_svc.get_real_weather(
                row['_dest_lat'], row['_dest_lon'], row['order_date'].date()
            )
            return w['temp'], w['rain']

        results = [get_weather(r) for _, r in tqdm(df.iterrows(), total=len(df), desc="Weather")]
        df['Real_Temp_C'], df['Real_Rain_mm'] = zip(*results)

        df['is_cold'] = (df['Real_Temp_C'] < 0).astype(int)
        df['is_heavy_rain'] = (df['Real_Rain_mm'] > 10.0).astype(int)

        return df

    def _add_econ_features(self, df: pd.DataFrame) -> tuple:
        print("[4/6] Макроекономічні дані...")
        unique_countries = df['Order Country'].unique()
        econ_map = {c: self.econ_svc.get_country_stats(c, self.year) for c in unique_countries}

        df['Real_GDP_PPP'] = df['Order Country'].map(
            lambda c: econ_map.get(c, {}).get('gdp_ppp', 20000))
        df['Real_LPI'] = df['Order Country'].map(
            lambda c: econ_map.get(c, {}).get('lpi', 3.0))
        df['Real_Fuel_Price'] = df['order_month'].map(
            lambda m: self.econ_svc.get_fuel_price(self.year, m))

        df['gdp_x_lpi'] = df['Real_GDP_PPP'] * df['Real_LPI']
        df['inflation_risk'] = df['Order Country'].map(
            lambda c: 1 if econ_map.get(c, {}).get('inf', 3.0) > 5.0 else 0)

        return df, econ_map

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        price = df['Order Item Product Price'].fillna(0)
        qty = df['Order Item Quantity'].fillna(1).clip(lower=1)
        discount = df['Order Item Discount Rate'].fillna(0)
        dist = df['Distance_KM'].fillna(5000).clip(lower=1)
        fuel = df['Real_Fuel_Price'].fillna(60)
        sched = df['Days for shipment (scheduled)'].fillna(4).clip(lower=1)

        df['price_per_km'] = (price * qty * (1 - discount)) / dist

        df['discount_x_qty'] = discount * qty

        df['fuel_x_distance'] = (fuel / 60.0) * (dist / 1000.0)

        SHIP_SLA = {'Same Day': 0, 'First Class': 1, 'Second Class': 2, 'Standard Class': 3}
        df['ship_sla_rank'] = df['Shipping Mode'].map(SHIP_SLA).fillna(3)

        SHIP_COST = {'Same Day': 3.0, 'First Class': 2.0, 'Second Class': 1.2, 'Standard Class': 1.0}
        df['ship_cost_per_km'] = df['Shipping Mode'].map(SHIP_COST).fillna(1.0) * df['Real_Fuel_Price'] / dist

        return df

    def _compute_target(self, df: pd.DataFrame, econ_map: dict) -> pd.DataFrame:
        print("[5/6] Розрахунок реалістичної прибутковості (таргет)...")

        fuel_mult = df['Real_Fuel_Price'] / 60.0
        ship_mult = df['Shipping Mode'].map(SHIP_COST_MULT).fillna(1.0)

        distance_penalty = (df['Distance_KM'] / 1000.0) * fuel_mult * ship_mult
        weather_penalty = np.where(df['Real_Rain_mm'] > 5.0, 2.0, 0.0)

        def apply_tariff(row):
            stats = econ_map.get(row['Order Country'], {})
            return self.econ_svc.calculate_tariff(
                stats,
                str(row.get('Category Name', 'Other')),
                row.get('Order Item Total', 0.0)
            )

        tariff_penalty = df.apply(apply_tariff, axis=1)

        df['Realistic_Profit'] = (
            df['Order Profit Per Order']
            - distance_penalty
            - weather_penalty
            - tariff_penalty
        )
        df['Is_Profitable'] = (df['Realistic_Profit'] >= 0).astype(int)
        return df

    def enrich_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        print("[1/6] Часові ознаки...")
        df = self._normalize_reference_fields(df)
        df = self._add_time_features(df)

        df, dest_coords = self._add_geo_features(df)
        df = self._add_weather_features(df)
        df, econ_map = self._add_econ_features(df)
        df = self._add_derived_features(df)
        df = self._compute_target(df, econ_map)

        print("[6/6] Фільтрація та обробка пропусків...")
        available_cols = [c for c in FEATURES_TO_KEEP if c in df.columns]
        missing_feats = [c for c in FEATURES_TO_KEEP if c not in df.columns]
        if missing_feats:
            logger.warning(f"Відсутні ознаки: {missing_feats}")

        df_clean = df[available_cols].copy()

        num_cols = df_clean.select_dtypes(include='number').columns
        cat_cols = df_clean.select_dtypes(include='object').columns
        df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].median())
        df_clean[cat_cols] = df_clean[cat_cols].fillna('Unknown')

        pos_rate = df_clean['Is_Profitable'].mean() * 100
        print(f"\nДатасет готовий: {len(df_clean):,} рядків | "
              f"{len(df_clean.columns)} ознак (+ таргет) | "
              f"Прибуткових: {pos_rate:.1f}% | Збиткових: {100 - pos_rate:.1f}%")

        return df_clean

    def run(self):
        import os
        print(f"\n{'='*60}")
        print(f"  DataPipeline | mode={self.mode} | {self.file_path}")
        print(f"{'='*60}")

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Вхідний файл не знайдено: {self.file_path}")

        df = pd.read_csv(self.file_path, encoding='latin1', low_memory=False)
        print(f"Завантажено {len(df):,} рядків, {len(df.columns)} колонок")

        if self.limit:
            df = df.head(self.limit)
            print(f"Обмежено до {self.limit} рядків")

        enriched_df = self.enrich_features(df)
        enriched_df.to_csv(self.output_path, index=False, encoding='utf-8')
        print(f"Збережено: {self.output_path}")
        return enriched_df


if __name__ == "__main__":
    DataPipeline(limit=1000).run()
