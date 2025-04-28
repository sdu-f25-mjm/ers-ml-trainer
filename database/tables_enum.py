from enum import Enum


class TableEnum(str, Enum):
    energy_data = "energy_data"
    production_data = "production_data"
    consumption_data = "consumption_data"
    exchange_data = "exchange_data"
    carbon_intensity = "carbon_intensity"
    aggregated_production = "aggregated_production"
    comparison_analysis = "comparison_analysis"
    consumption_forecast = "consumption_forecast"
    cache_metrics = "cache_metrics"


