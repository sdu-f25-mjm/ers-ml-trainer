from enum import Enum


class TableEnum(str, Enum):
    ENERGY = "energy_data"
    PRODUCTION = "production_data"
    CONSUMPTION = "consumption_data"
    EXCHANGE = "exchange_data"
    CARBON_INTENSITY = "carbon_intensity_data"
    AGGREGATED_PRODUCTION = "aggregated_production"
    COMPARISON_ANALYSIS = "comparison_analysis"
    CONSUMPTION_FORECAST = "consumption_forecast"
    CACHE_WEIGHTS = "derived_data_cache_weights"


