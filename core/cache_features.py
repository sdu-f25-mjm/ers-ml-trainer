# # core/cache_features.py
# from datetime import datetime, timedelta
# from typing import Dict, List, Any, Optional
#
# import numpy as np
#
#
# class CacheFeatureCalculator:
#     def __init__(self, current_time: datetime = None):
#         """Initialize the cache feature calculator with the current time.
#
#         Args:
#             current_time: Current datetime, defaults to now if not provided
#         """
#         self.current_time = current_time or datetime.now()
#         self.access_history: Dict[str, List[datetime]] = {}  # Track access history by data ID
#         self.volatility_history: Dict[str, List[float]] = {}  # Track value changes for volatility calc
#
#         # Constants for feature weights
#         self.weights = {
#             'recency': 0.25,  # Recent data is more important
#             'access_frequency': 0.20,  # Frequently accessed data
#             'time_relevance': 0.15,  # Data from relevant time periods
#             'production_importance': 0.15,  # Important production data (wind/solar)
#             'volatility': 0.15,  # Data that changes frequently
#             'complexity': 0.10  # Complex data that's expensive to recompute
#         }
#
#     def calculate_cache_priority(self, data_id: str, data: Dict[str, Any]) -> float:
#         """Calculate overall cache priority score for a data item.
#
#         Args:
#             data_id: Unique identifier for the data
#             data: Dictionary containing energy data fields
#
#         Returns:
#             float: Priority score between 0-1, higher means keep in cache
#         """
#         # Calculate individual features
#         recency = self.calculate_recency(data_id)
#         frequency = self.calculate_usage_frequency(data_id)
#         time_relevance = self.calculate_time_relevance(data)
#         production_importance = self.calculate_production_importance(data)
#         volatility = self.calculate_data_volatility(data_id, data)
#         complexity = self.calculate_complexity(data)
#
#         # Combine weighted features
#         priority = (
#                 self.weights['recency'] * recency +
#                 self.weights['access_frequency'] * frequency +
#                 self.weights['time_relevance'] * time_relevance +
#                 self.weights['production_importance'] * production_importance +
#                 self.weights['volatility'] * volatility +
#                 self.weights['complexity'] * complexity
#         )
#
#         return min(1.0, max(0.0, priority))
#
#     def calculate_usage_frequency(self, data_id: str) -> float:
#         """Calculate how often the data is accessed relative to other items.
#
#         Args:
#             data_id: Unique identifier for the data
#
#         Returns:
#             float: Normalized access frequency (0-1)
#         """
#         if data_id not in self.access_history or not self.access_history[data_id]:
#             return 0.0
#
#         # Calculate access rate over the last week
#         week_ago = self.current_time - timedelta(days=7)
#         recent_accesses = sum(1 for ts in self.access_history[data_id] if ts >= week_ago)
#
#         # Normalize by maximum possible accesses (arbitrary cap at 1000)
#         return min(1.0, recent_accesses / 1000)
#
#     def calculate_recency(self, data_id: str) -> float:
#         """Calculate how recently the data was accessed.
#
#         Args:
#             data_id: Unique identifier for the data
#
#         Returns:
#             float: Recency score (0-1), 1 is most recent
#         """
#         if data_id not in self.access_history or not self.access_history[data_id]:
#             return 0.0
#
#         last_access = max(self.access_history[data_id])
#         hours_since_access = (self.current_time - last_access).total_seconds() / 3600
#
#         # Exponential decay: 0.5 after 24 hours, approaches 0 after a week
#         return max(0.0, min(1.0, np.exp(-hours_since_access / 24)))
#
#     def calculate_time_relevance(self, data: Dict[str, Any]) -> float:
#         """Calculate how relevant the data's timestamp is.
#
#         Args:
#             data: Dictionary containing energy data
#
#         Returns:
#             float: Time relevance score (0-1)
#         """
#         # Check if timestamp exists in data
#         timestamp = data.get('HourUTC') or data.get('Timestamp')
#         if not timestamp:
#             return 0.5  # Default if no timestamp
#
#         if isinstance(timestamp, str):
#             timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
#
#         hours_difference = abs((self.current_time - timestamp).total_seconds() / 3600)
#
#         # Recent data (last 48h) is most important, then decay
#         if hours_difference <= 48:
#             return 1.0 - (hours_difference / 48) * 0.5
#         else:
#             # Exponential decay for older data
#             return 0.5 * np.exp(-(hours_difference - 48) / 168)  # 168h = 1 week
#
#     def calculate_production_importance(self, data: Dict[str, Any]) -> float:
#         """Calculate importance based on energy production mix.
#
#         Args:
#             data: Dictionary containing energy data
#
#         Returns:
#             float: Production importance score (0-1)
#         """
#         # Default score if we don't have necessary data
#         if not data:
#             return 0.5
#
#         # Check if this is a production or carbon intensity record
#         has_wind = 'WindTotal_MWh' in data or 'Wind_Percent' in data
#         has_solar = 'SolarTotal_MWh' in data or 'Solar_Percent' in data
#
#         if not (has_wind or has_solar):
#             return 0.5  # Not production data
#
#         # Calculate importance based on renewable percentage
#         total_production = 0
#         renewable_production = 0
#
#         # For production data
#         if 'WindTotal_MWh' in data:
#             wind = float(data.get('WindTotal_MWh', 0) or 0)
#             solar = float(data.get('SolarTotal_MWh', 0) or 0)
#             total = float(data.get('TotalProduction_MWh', 0) or 0)
#
#             # If total production isn't available, sum up components
#             if not total:
#                 central = float(data.get('CentralPower_MWh', 0) or 0)
#                 local = float(data.get('LocalPower_MWh', 0) or 0)
#                 total = central + local + wind + solar
#
#             renewable_production = wind + solar
#             total_production = max(0.1, total)  # Avoid division by zero
#
#         # For carbon intensity data
#         elif 'Wind_Percent' in data:
#             wind_pct = float(data.get('Wind_Percent', 0) or 0)
#             solar_pct = float(data.get('Solar_Percent', 0) or 0)
#             renewable_production = wind_pct + solar_pct
#             total_production = 100
#
#         # Higher importance for higher renewable percentage
#         renewable_ratio = renewable_production / total_production
#         return min(1.0, max(0.0, 0.4 + renewable_ratio * 0.6))
#
#     def calculate_data_volatility(self, data_id: str, data: Dict[str, Any]) -> float:
#         """Calculate data volatility (how much it changes).
#
#         Args:
#             data_id: Unique identifier for the data
#             data: Dictionary containing energy data
#
#         Returns:
#             float: Volatility score (0-1), higher for more volatile data
#         """
#         # Extract key numeric values for volatility tracking
#         key_metrics = self._extract_key_metrics(data)
#
#         if not key_metrics:
#             return 0.5  # Default volatility if no metrics
#
#         # Initialize history for this data if needed
#         if data_id not in self.volatility_history:
#             self.volatility_history[data_id] = [key_metrics]
#             return 0.5  # Default for first occurrence
#
#         # Add new metrics to history (keep last 24 values)
#         self.volatility_history[data_id].append(key_metrics)
#         if len(self.volatility_history[data_id]) > 24:
#             self.volatility_history[data_id].pop(0)
#
#         # Calculate coefficient of variation across history
#         history = self.volatility_history[data_id]
#         if len(history) < 2:
#             return 0.5
#
#         # Calculate coefficient of variation and normalize
#         mean_val = np.mean(history)
#         if mean_val == 0:
#             return 0.5
#         std_val = np.std(history)
#         cv = std_val / max(0.1, abs(mean_val))  # Avoid division by zero
#
#         # Map coefficient of variation to 0-1 score (sigmoid-like scaling)
#         return min(1.0, max(0.0, 0.5 + 0.5 * (cv / (1 + cv))))
#
#     def calculate_complexity(self, data: Dict[str, Any]) -> float:
#         """Calculate computational complexity of regenerating the data.
#
#         Args:
#             data: Dictionary containing energy data
#
#         Returns:
#             float: Complexity score (0-1), higher for more complex data
#         """
#         # Count number of fields as proxy for complexity
#         field_count = len(data) if data else 0
#
#         # Specific types that are more complex to regenerate
#         has_forecast = any(key for key in data.keys() if 'Forecast' in key)
#         has_carbon = any(key for key in data.keys() if 'Carbon' in key)
#         has_aggregation = any(key for key in data.keys() if 'Period' in key or 'Aggregation' in key)
#
#         # Base complexity from field count
#         complexity = min(1.0, field_count / 20)  # Normalize by assuming max 20 fields
#
#         # Adjust for special fields
#         if has_forecast:
#             complexity = min(1.0, complexity + 0.2)  # Forecasts are expensive
#         if has_carbon:
#             complexity = min(1.0, complexity + 0.15)  # Carbon calcs are complex
#         if has_aggregation:
#             complexity = min(1.0, complexity + 0.1)  # Aggregations are moderately complex
#
#         return complexity
#
#     def update_access_history(self, data_id: str, timestamp: Optional[datetime] = None):
#         """Update access history for a data item.
#
#         Args:
#             data_id: Unique identifier for the data
#             timestamp: Access timestamp, defaults to current time
#         """
#         access_time = timestamp or self.current_time
#
#         if data_id not in self.access_history:
#             self.access_history[data_id] = []
#
#         self.access_history[data_id].append(access_time)
#
#         # Keep only last 100 accesses to prevent unbounded growth
#         if len(self.access_history[data_id]) > 100:
#             self.access_history[data_id] = self.access_history[data_id][-100:]
#
#     def update_current_time(self, current_time: datetime):
#         """Update the current time reference.
#
#         Args:
#             current_time: New current datetime
#         """
#         self.current_time = current_time
#
#     def _extract_key_metrics(self, data: Dict[str, Any]) -> float:
#         """Extract a composite key metric from the data for volatility tracking.
#
#         Args:
#             data: Dictionary containing energy data
#
#         Returns:
#             float: Composite metric value
#         """
#         value = 0
#
#         # Try to extract the most relevant metrics based on data type
#         if 'CentralPower_MWh' in data:
#             value += float(data.get('CentralPower_MWh', 0) or 0)
#         if 'WindTotal_MWh' in data:
#             value += float(data.get('WindTotal_MWh', 0) or 0)
#         if 'SolarTotal_MWh' in data:
#             value += float(data.get('SolarTotal_MWh', 0) or 0)
#         if 'GrossConsumption_MWh' in data:
#             value += float(data.get('GrossConsumption_MWh', 0) or 0)
#         if 'ConsumptionTotal_MWh' in data:
#             value += float(data.get('ConsumptionTotal_MWh', 0) or 0)
#         if 'CarbonIntensity_gCO2perKWh' in data:
#             value += float(data.get('CarbonIntensity_gCO2perKWh', 0) or 0)
#
#         return value
