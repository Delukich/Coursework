import logging
import os
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.weather_service import WeatherService
from src.economics_service import EconomicsService
from src.geo_service import haversine_distance, geocode_city
from src.config import (
    RAW_DATA_PATH,
    ENRICHED_OUTPUT_PATH,
    COUNTRY_COORDS_FALLBACK,
    SHIPPING_MODE_COST,
    SHIPPING_MODE_RANK,
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

DEFAULT_DISTANCE_KM = 5000.0
DEFAULT_WEATHER = (15.0, 0.0)
MARKET_COORDS_FALLBACK = {
    'Africa': (7.0, 20.0),
    'Europe': (50.0, 10.0),
    'LATAM': (-15.0, -60.0),
    'Pacific Asia': (15.0, 115.0),
    'USCA': (39.0, -98.0),
}
REGION_COORDS_FALLBACK = {
    'Canada': (56.0, -106.0),
    'Caribbean': (18.0, -72.0),
    'Central Africa': (2.0, 20.0),
    'Central America': (15.0, -90.0),
    'Central Asia': (43.0, 68.0),
    'East Africa': (-3.0, 36.0),
    'East of USA': (40.0, -75.0),
    'Eastern Asia': (35.0, 120.0),
    'Eastern Europe': (50.0, 25.0),
    'North Africa': (27.0, 18.0),
    'Northern Europe': (56.0, 10.0),
    'Oceania': (-25.0, 135.0),
    'South America': (-15.0, -60.0),
    'South Asia': (20.0, 78.0),
    'South of  USA': (32.0, -95.0),
    'Southeast Asia': (10.0, 106.0),
    'Southern Africa': (-22.0, 24.0),
    'Southern Europe': (41.0, 15.0),
    'US Center': (39.0, -98.0),
    'West Africa': (10.0, -1.0),
    'West Asia': (32.0, 45.0),
    'West of USA': (36.0, -120.0),
    'Western Europe': (48.0, 5.0),
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
        logger.info(f"Pipeline | mode={mode} | year={self.year} | output={self.output_path}")

    def _load_source_data(self) -> pd.DataFrame:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Input file not found: {self.file_path}")

        df = pd.read_csv(self.file_path, encoding='latin1', low_memory=False)
        print(f"Loaded {len(df):,} rows and {len(df.columns)} columns")

        if self.limit:
            df = df.head(self.limit)
            print(f"Limited to {self.limit} rows")

        return df

    def _save_enriched_data(self, df: pd.DataFrame):
        df.to_csv(self.output_path, index=False, encoding='utf-8')
        print(f"Saved: {self.output_path}")

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
        print("[2/6] Geocoding destinations...")
        unique_locs = df[['Order City', 'Order Country', 'Order Region', 'Market']].drop_duplicates()
        dest_coords = self._build_dest_coords(unique_locs)
        df = self._attach_dest_coords(df, dest_coords)
        df['Distance_KM'] = df.apply(
            lambda row: self._calc_distance_km(row, dest_coords), axis=1
        )

        df['dest_lat_abs'] = df['_dest_lat'].abs()
        df['cross_hemisphere'] = (
            (df['Latitude'].fillna(0) > 0) & (df['_dest_lat'].fillna(0) < 0) |
            (df['Latitude'].fillna(0) < 0) & (df['_dest_lat'].fillna(0) > 0)
        ).astype(int)

        return df

    def _build_dest_coords(self, unique_locs: pd.DataFrame) -> dict:
        dest_coords: dict = {}
        for _, row in tqdm(unique_locs.iterrows(), total=len(unique_locs), desc="Geocoding"):
            lat, lon = self._resolve_destination_coords(row)
            dest_coords[(row['Order City'], row['Order Country'])] = (lat, lon)
        return dest_coords

    def _resolve_destination_coords(self, row: pd.Series) -> tuple[float, float]:
        lat, lon = geocode_city(row['Order City'], row['Order Country'])
        if not np.isnan(lat) and not np.isnan(lon):
            return lat, lon

        country = row.get('Order Country')
        if country in COUNTRY_COORDS_FALLBACK:
            return COUNTRY_COORDS_FALLBACK[country]

        region = str(row.get('Order Region', '')).strip()
        if region in REGION_COORDS_FALLBACK:
            return REGION_COORDS_FALLBACK[region]

        market = str(row.get('Market', '')).strip()
        if market in MARKET_COORDS_FALLBACK:
            return MARKET_COORDS_FALLBACK[market]

        return np.nan, np.nan

    def _attach_dest_coords(self, df: pd.DataFrame, dest_coords: dict) -> pd.DataFrame:
        df['_dest_lat'] = df.apply(
            lambda r: dest_coords.get((r['Order City'], r['Order Country']), (np.nan, np.nan))[0], axis=1
        )
        df['_dest_lon'] = df.apply(
            lambda r: dest_coords.get((r['Order City'], r['Order Country']), (np.nan, np.nan))[1], axis=1
        )
        return df

    def _calc_distance_km(self, row: pd.Series, dest_coords: dict) -> float:
        dest = dest_coords.get((row['Order City'], row['Order Country']))
        if not dest or pd.isna(row.get('Latitude')) or pd.isna(row.get('Longitude')):
            return DEFAULT_DISTANCE_KM
        distance = haversine_distance(row['Latitude'], row['Longitude'], dest[0], dest[1])
        return distance if not np.isnan(distance) else DEFAULT_DISTANCE_KM

    def _add_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("[3/6] Weather features...")
        results = [self._get_weather_for_row(r) for _, r in tqdm(df.iterrows(), total=len(df), desc="Weather")]
        df['Real_Temp_C'], df['Real_Rain_mm'] = zip(*results)

        df['is_cold'] = (df['Real_Temp_C'] < 0).astype(int)
        df['is_heavy_rain'] = (df['Real_Rain_mm'] > 10.0).astype(int)

        return df

    def _get_weather_for_row(self, row: pd.Series) -> tuple:
        if pd.isna(row['order_date']):
            return DEFAULT_WEATHER

        lat = row['_dest_lat']
        lon = row['_dest_lon']
        if pd.isna(lat) or pd.isna(lon):
            lat, lon = self._resolve_destination_coords(row)
        if pd.isna(lat) or pd.isna(lon):
            return DEFAULT_WEATHER

        weather = self.weather_svc.get_real_weather(lat, lon, row['order_date'].date())
        return weather['temp'], weather['rain']

    def _add_econ_features(self, df: pd.DataFrame) -> tuple:
        print("[4/6] Economic features...")
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
        dist = df['Distance_KM'].fillna(DEFAULT_DISTANCE_KM).clip(lower=1)
        fuel = df['Real_Fuel_Price'].fillna(60)

        df['price_per_km'] = (price * qty * (1 - discount)) / dist

        df['discount_x_qty'] = discount * qty

        df['fuel_x_distance'] = (fuel / 60.0) * (dist / 1000.0)

        df['ship_sla_rank'] = df['Shipping Mode'].map(SHIPPING_MODE_RANK).fillna(3)
        df['ship_cost_per_km'] = df['Shipping Mode'].map(SHIPPING_MODE_COST).fillna(1.0) * df['Real_Fuel_Price'] / dist

        return df

    def _compute_target(self, df: pd.DataFrame, econ_map: dict) -> pd.DataFrame:
        print("[5/6] Calculating realistic profitability target...")

        fuel_mult = df['Real_Fuel_Price'] / 60.0
        ship_mult = df['Shipping Mode'].map(SHIPPING_MODE_COST).fillna(1.0)

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

        print("[1/6] Time features...")
        df = self._normalize_reference_fields(df)
        df = self._add_time_features(df)

        df = self._add_geo_features(df)
        df = self._add_weather_features(df)
        df, econ_map = self._add_econ_features(df)
        df = self._add_derived_features(df)
        df = self._compute_target(df, econ_map)

        print("[6/6] Filtering columns and handling missing values...")
        available_cols = [c for c in FEATURES_TO_KEEP if c in df.columns]
        missing_feats = [c for c in FEATURES_TO_KEEP if c not in df.columns]
        if missing_feats:
            logger.warning(f"Missing features: {missing_feats}")

        df_clean = df[available_cols].copy()

        num_cols = df_clean.select_dtypes(include='number').columns
        cat_cols = df_clean.select_dtypes(include='object').columns
        df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].median())
        df_clean[cat_cols] = df_clean[cat_cols].fillna('Unknown')

        pos_rate = df_clean['Is_Profitable'].mean() * 100
        print(f"\nDataset ready: {len(df_clean):,} rows | "
              f"{len(df_clean.columns)} features (+ target) | "
              f"Profitable: {pos_rate:.1f}% | Loss-making: {100 - pos_rate:.1f}%")

        return df_clean

    def run(self):
        print(f"\n{'='*60}")
        print(f"  DataPipeline | mode={self.mode} | {self.file_path}")
        print(f"{'='*60}")

        df = self._load_source_data()
        enriched_df = self.enrich_features(df)
        self._save_enriched_data(enriched_df)
        return enriched_df


if __name__ == "__main__":
    DataPipeline(limit=1000).run()
