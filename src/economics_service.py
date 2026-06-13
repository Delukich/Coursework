import logging
from src.config import (
    normalize_country_name, CATEGORY_TARIFFS,
    COUNTRY_STATS_2018, DEFAULT_STATS_2018,
    COUNTRY_STATS_CURRENT, DEFAULT_STATS_CURRENT,
    REAL_FUEL_PRICES_2018, REAL_FUEL_PRICE_CURRENT
)

logger = logging.getLogger(__name__)


class EconomicsService:
    FREE_TRADE_PAIRS = {
        ('US_TERRITORY', 'NAFTA'),
        ('US_TERRITORY', 'USMCA'),
        ('US_TERRITORY', 'US_TERRITORY'),
        ('EU', 'EU'),
    }

    def get_country_stats(self, raw_country_name: str, year: int = 2018) -> dict:
        clean_name = normalize_country_name(raw_country_name)
        return self._get_stats_for_year(clean_name, year)

    def _get_stats_for_year(self, clean_name: str, year: int) -> dict:
        if year <= 2018:
            return COUNTRY_STATS_2018.get(clean_name, DEFAULT_STATS_2018).copy()
        return COUNTRY_STATS_CURRENT.get(clean_name, DEFAULT_STATS_CURRENT).copy()

    def get_fuel_price(self, year: int, month: int) -> float:
        if year <= 2018:
            return self._fuel_price_for_2018(year, month)
        return float(REAL_FUEL_PRICE_CURRENT)

    def _fuel_price_for_2018(self, year: int, month: int) -> float:
        try:
            return float(REAL_FUEL_PRICES_2018[year][month - 1])
        except (KeyError, IndexError):
            logger.warning(f"No fuel price for {year}-{month:02d}; using 60.0")
            return 60.0

    def calculate_tariff(self, country_stats: dict, category: str, order_total: float) -> float:
        origin_zone = 'US_TERRITORY'
        dest_zone = country_stats.get('tariff_zone', 'WTO')

        if (origin_zone, dest_zone) in self.FREE_TRADE_PAIRS:
            return 0.0

        base_rate = CATEGORY_TARIFFS.get(category, CATEGORY_TARIFFS.get('Other', 0.05))

        lpi = country_stats.get('lpi', 3.0)
        lpi_adj = 1.0 + max(0.0, (3.0 - lpi) * 0.02)

        tariff = order_total * base_rate * lpi_adj
        logger.debug(
            f"Tariff: zone={dest_zone}, cat={category}, "
            f"rate={base_rate:.2%}, lpi_adj={lpi_adj:.3f}, total={tariff:.2f}"
        )
        return tariff
