"""
Cache Metrics and Their Role in RL Model Training
------------------------------------------------

Each cache metric (feature column) provides the RL agent with information about the query, the endpoint, or the data being considered for caching. Including the right metrics helps the agent learn which items are most valuable to cache, improving hit rates and overall efficiency.

**Common cache metrics and their benefits:**

- `cache_name` (or endpoint): Identifies the API endpoint or resource. Helps the agent learn which endpoints are frequently accessed and should be prioritized for caching.
- `cache_key`: A unique identifier for the cache entry. Useful for distinguishing between different queries or parameterizations.
- `hit_ratio`: Indicates how often this item is requested and found in cache. High hit ratios suggest valuable cache candidates.
- `item_count`: Number of items returned by the query. Larger item counts may be more expensive to recompute, so caching them can save resources.
- `load_time_ms`: Time taken to load or compute the data. High load times indicate expensive queries, which are good candidates for caching.
- `size_bytes`: Size of the cached item. Helps the agent balance cache space usage and avoid evicting small, frequently used items in favor of large, rarely used ones.
- `traffic_intensity`: Frequency or volume of requests for this item. High-traffic items are more beneficial to cache.
- `policy_triggered`: Indicates if a cache policy (e.g., TTL expiry, manual invalidation) was triggered. Can help the agent learn about cache churn and stability.
- `rl_action_taken`: Records the RL agent's previous action (cache or not). Useful for learning from past decisions.
- `timestamp`: When the data was generated or cached. Helps the agent learn about recency and time-based patterns.
- `in_cache`: Boolean indicating if the item is currently in cache. Useful for tracking cache state.
- `calculated_priority`: A composite score (possibly from a heuristic or another model) indicating the importance of caching this item.

**How these metrics help:**
- By providing the RL agent with a rich set of features, it can learn complex patterns (e.g., "cache large, slow-to-compute items that are frequently accessed").
- Metrics like `hit_ratio`, `traffic_intensity`, and `load_time_ms` directly inform the agent about the value and cost of caching each item.
- Including endpoint patterns (not just unique URLs) helps the agent generalize and avoid overfitting to specific queries.
- The agent can learn to balance cache space, computational cost, and user-perceived latency by weighing these features during training.

For best results, select metrics that reflect both the cost of recomputation and the value of fast access for your workload.
"""

# core/cache_environment.py
import logging
from typing import List, Dict, Tuple, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from core.utils import build_db_url
from database.database_connection import (
    create_database_connection,
    get_available_tables,
    table_exists,
    load_data_from_database,
    get_table_schema
)

db_url = build_db_url()  # or fallback if db_url is None
engine = create_database_connection(db_url)

from core.utils import list_available_models

# Get all models
models = list_available_models()
print(f"Found {len(models)} models:")
for m in models:
    print(
        f"- {m['algorithm'].upper()} (cache_size_mb: {m.get('cache_size_mb')}MB) trained on {m['device'].upper()} at {m['created_at']}"
    )

class MariaDBCacheEnvironment(gym.Env):
    """
    A gymnasium environment for database cache optimization using MariaDB.

    This environment simulates a cache for a database, allowing an agent to
    decide what to keep in a limited cache to optimize query performance.
    """

    metadata = {'render_modes': ['human']}

    def __init__(
            self,
            db_url: str = None,
            cache_size: int = 10,
            cache_size_mb: int = None,  # New: cache size in MB
            feature_columns: List[str] = None,
            target_column: str = None,
            max_queries: int = 500,
            table_name: str = None,  # Optional table name parameter
            cache_weights: Optional[List[str]] = None  # <-- add param
    ):
        """
        Initialize the MariaDB cache environment.

        Args:
            db_url: The database connection URL
            cache_size: Size of the cache (number of rows)
            cache_size_mb: Size of the cache in MB
            feature_columns: Features to use for observation space
            target_column: Target column for optimization
            max_queries: Maximum number of queries to run
            table_name: Optional specific table to use (will auto-discover if None)
            cache_weights: Optional list of columns to use for weighted reward calculation
        """
        super().__init__()

        self.logger = logging.getLogger(__name__)
        self.cache_size = cache_size
        self.cache_size_mb = cache_size_mb
        self.max_queries = max_queries
        self.db_url = db_url
        self.cache_weights = cache_weights

        # Create database connection
        self.logger.info(f"Connecting to database: {self.db_url}")
        try:
            self.engine = create_database_connection(self.db_url)
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            raise

        # Discover available tables
        self.available_tables = get_available_tables(self.engine)
        if not self.available_tables:
            raise ValueError("No tables available in the database")

        self.logger.info(f"Available tables: {', '.join(self.available_tables)}")

        # Select table to use - either specified or first available
        if table_name:
            if table_exists(self.engine, table_name):
                self.table_name = table_name
                self.logger.info(f"Using specified table: {self.table_name}")
            else:
                self.logger.warning(f"Specified table {table_name} not found")
                # Fall back to first available table
                if self.available_tables:
                    self.table_name = self.available_tables[0]
                    self.logger.info(f"Using first available table: {self.table_name}")
        else:
            # No table specified, use first available
            if self.available_tables:
                self.table_name = self.available_tables[0]
                self.logger.info(f"No table specified. Using first available: {self.table_name}")
            else:
                raise ValueError("No tables available in the database")

        # Get table schema to understand data structure
        self.table_schema = get_table_schema(self.engine, self.table_name)

        # Load data from database
        self.logger.info(f"Loading data from table: {self.table_name}")
        self.data = load_data_from_database(
            self.engine,
            self.table_name,
            limit=self.max_queries * 2  # Load more data than needed for variety
        )

        if len(self.data) == 0:
            raise ValueError(f"No data found in table {self.table_name}")

        self.logger.info(f"Loaded {len(self.data)} rows from {self.table_name}")

        # Determine cache size (number of items) based on MB if provided
        if self.cache_size_mb is not None:
            if "size_bytes" in self.data.columns:
                avg_item_size = self.data["size_bytes"].mean()
                if avg_item_size > 0:
                    self.cache_size = max(1, int((self.cache_size_mb * 1024 * 1024) / avg_item_size))
                    self.logger.info(f"Cache size set to {self.cache_size} items (from {self.cache_size_mb} MB, avg item size {avg_item_size:.2f} bytes)")
                else:
                    self.logger.warning("Average item size is zero, falling back to default cache_size")
            else:
                self.logger.warning("Column 'size_bytes' not found in data, cannot compute cache size from MB. Using default cache_size.")

        # Set feature columns
        if feature_columns:
            # Verify all specified columns exist in the data
            missing_columns = [col for col in feature_columns if col not in self.data.columns]
            if missing_columns:
                self.logger.warning(f"Columns not found in data: {', '.join(missing_columns)}")
                feature_columns = [col for col in feature_columns if col in self.data.columns]

            if not feature_columns:
                self.logger.warning("No valid feature columns provided, using all numeric columns")
                self.feature_columns = self._get_default_feature_columns()
            else:
                self.feature_columns = feature_columns
        else:
            self.logger.info("No feature columns specified, using all numeric columns")
            self.feature_columns = self._get_default_feature_columns()

        # Define spaces
        feature_count = len(self.feature_columns)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(feature_count + self.cache_size,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)  # 0: don't cache, 1: cache

        # Initialize cache and metrics
        self.cache = []
        self.current_query_idx = 0
        self.queries_executed = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def _get_default_feature_columns(self) -> List[str]:
        """Select default feature columns based on data types."""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            # If no numeric columns, use all columns
            return self.data.columns.tolist()[:5]  # Limit to first 5 to avoid huge state space
        return numeric_cols[:5]  # Limit to first 5 numeric columns

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        self.cache = []
        self.current_query_idx = 0
        self.queries_executed = 0
        self.cache_hits = 0
        self.cache_misses = 0

        # Get initial observation
        query_features = self._get_query_features(self.current_query_idx)
        cache_features = np.zeros(self.cache_size)
        obs = np.concatenate([query_features, cache_features])

        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        self.queries_executed += 1

        # Process the action - 1 means add to cache, 0 means don't
        current_query = self.data.iloc[self.current_query_idx]

        cache_hit = self._check_cache_hit(current_query)

        if action == 1 and not cache_hit:
            # Add to cache if not already in cache
            self._update_cache(current_query)

        # --- Custom weighted reward ---
        if self.cache_weights:
            reward = 0.0
            for col in self.cache_weights:
                try:
                    val = float(current_query[col])
                except Exception:
                    val = 0.0
                reward += val
            reward = reward if cache_hit else -0.1 * reward
        else:
            reward = 1.0 if cache_hit else -0.1

        # Move to next query
        self.current_query_idx = (self.current_query_idx + 1) % len(self.data)

        # Get next observation
        next_query_features = self._get_query_features(self.current_query_idx)
        cache_features = self._get_cache_features()
        obs = np.concatenate([next_query_features, cache_features])

        # Check if done
        done = self.queries_executed >= self.max_queries

        # Additional info
        info = {
            "cache_hit_rate": self.cache_hits / max(1, self.queries_executed),
            "cache_miss_rate": self.cache_misses / max(1, self.queries_executed),
            "cache_size": len(self.cache),
            "queries_executed": self.queries_executed
        }

        return obs, reward, done, False, info

    def _get_query_features(self, query_idx: int) -> np.ndarray:
        """Extract features for the current query."""
        query = self.data.iloc[query_idx]
        features = []

        for col in self.feature_columns:
            value = query[col]
            # Convert non-numeric values to numeric representation
            if not isinstance(value, (int, float)):
                if isinstance(value, str):
                    value = hash(value) % 1000 / 1000.0
                else:
                    value = 0.0
            features.append(float(value))

        return np.array(features, dtype=np.float32)

    def _get_cache_features(self) -> np.ndarray:
        """Get features representing the current cache state."""
        features = np.zeros(self.cache_size, dtype=np.float32)

        # Fill with values representing items in cache
        for i, item in enumerate(self.cache):
            if i >= self.cache_size:
                break
            features[i] = 1.0  # Mark cache slot as filled

        return features

    def _check_cache_hit(self, query) -> bool:
        """Check if the query would be a cache hit."""
        # Simple implementation - check if exact row exists in cache
        for cached_item in self.cache:
            if self._items_match(cached_item, query):
                self.cache_hits += 1
                return True

        self.cache_misses += 1
        return False

    def _items_match(self, item1, item2) -> bool:
        """Check if two items match for cache purposes."""
        # Compare only the feature columns
        for col in self.feature_columns:
            if item1[col] != item2[col]:
                return False
        return True

    def _update_cache(self, query):
        """Add a new item to the cache, removing oldest if full."""
        if len(self.cache) >= self.cache_size:
            self.cache.pop(0)  # Remove oldest (FIFO strategy)

        self.cache.append(query)

    def render(self, mode='human'):
        """Render the current state of the environment."""
        if mode == 'human':
            print(f"Queries executed: {self.queries_executed}/{self.max_queries}")
            print(f"Cache hits: {self.cache_hits}, Cache misses: {self.cache_misses}")
            print(f"Current hit rate: {self.cache_hits / max(1, self.queries_executed):.2f}")
            print(f"Cache size: {len(self.cache)}/{self.cache_size}")

    def close(self):
        """Clean up resources."""
        if hasattr(self, 'engine') and self.engine:
            # SQLAlchemy engines don't need explicit closing
            # The connection pool will be disposed when the object is deleted
            pass


def create_mariadb_cache_env(
        db_url: str = None,
        cache_size: int = 10,
        cache_size_mb: int = None,
        feature_columns: list[str] = None,
        max_queries: int = 500,
        table_name: str = None,
        cache_weights: Optional[List[str]] = None
) -> MariaDBCacheEnvironment:
    """Factory function to create a MariaDB cache environment instance."""
    return MariaDBCacheEnvironment(
        db_url=db_url,
        cache_size=cache_size,
        cache_size_mb=cache_size_mb,
        feature_columns=feature_columns,
        max_queries=max_queries,
        table_name=table_name,
        cache_weights=cache_weights
    )

