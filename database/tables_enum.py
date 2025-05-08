from enum import Enum


class TableEnum(str, Enum):
    production_data = "production"
    consumption_data = "consumption"
    exchange_data = "exchange"
    cache_metrics = "cache_metrics"
