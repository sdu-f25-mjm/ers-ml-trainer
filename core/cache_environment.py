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

import logging
from typing import List, Tuple, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from core.utils import build_db_url
from database.database_connection import (
    create_database_connection,
    get_available_tables,
    table_exists,
    load_data_from_database,
    get_table_schema,
)

# Establish DB connection immediately for module introspection
db_url = build_db_url()
engine = create_database_connection(db_url)

from core.utils import list_available_models

# List existing trained models at import time
models = list_available_models()
print(f"Found {len(models)} models:")
for model in models:
    print(
        f"- {model['algorithm'].upper()} "
        f"(cache size: {model['cache_size']}) "
        f"trained on {model['device'].upper()} at {model['created_at']}"
    )


class MariaDBCacheEnvironment(gym.Env):
    def __init__(
        self,
        db_url: str = None,
        cache_size: int = 10,
        feature_columns: List[str] = None,
        max_queries: int = 500,
        table_name: str = None,
        cache_weights: Optional[List[str]] = None,
    ):
        """
        Initialize the MariaDB cache environment.

        Args:
            db_url: The database connection URL
            cache_size: Size of the cache (number of rows)
            feature_columns: Features to use for observation space
            max_queries: Maximum number of queries to run
            table_name: Specific table to use (None = auto-discover)
            cache_weights: List of columns for weighted reward
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # Core parameters
        self.db_url = db_url
        self.cache_size = cache_size
        self.max_queries = max_queries
        self.cache_weights = cache_weights

        # DB setup
        self.logger.info(f"Connecting to database: {self.db_url}")
        self.engine = create_database_connection(self.db_url)

        # Table discovery
        self.available_tables = get_available_tables(self.engine)
        if not self.available_tables:
            raise ValueError("No tables available in the database")
        self.logger.info(f"Available tables: {', '.join(self.available_tables)}")

        # Table selection
        if table_name and table_exists(self.engine, table_name):
            self.table_name = table_name
        else:
            if table_name:
                self.logger.warning(f"Table '{table_name}' not found; defaulting to first available")
            self.table_name = self.available_tables[0]
        self.logger.info(f"Using table: {self.table_name}")

        # Load schema & data
        self.table_schema = get_table_schema(self.engine, self.table_name)
        self.data = load_data_from_database(
            self.engine,
            self.table_name,
            limit=self.max_queries * 2,
        )
        if self.data.empty:
            raise ValueError(f"No data found in table '{self.table_name}'")
        self.logger.info(f"Loaded {len(self.data)} rows from '{self.table_name}'")

        # Feature columns
        if feature_columns:
            cols = [c for c in feature_columns if c in self.data.columns]
            if not cols:
                self.logger.warning("No valid feature columns provided; using defaults")
                self.feature_columns = self._get_default_feature_columns()
            else:
                self.feature_columns = cols
        else:
            self.feature_columns = self._get_default_feature_columns()
        self.logger.info(f"Feature columns: {self.feature_columns}")

        # --- Add debug: print first few rows of data for inspection ---
        self.logger.info(f"First 3 rows of loaded data:\n{self.data[self.feature_columns].head(3).to_string(index=False)}")

        # --- Add debug: print cache_size and check ---
        self.logger.info(f"Configured cache_size: {self.cache_size}")
        if self.cache_size < 1:
            self.logger.error("Cache size must be at least 1. Please check the value passed from the frontend.")

        # Gym spaces
        obs_dim = len(self.feature_columns) + self.cache_size
        self.logger.debug(f"Observation space dimension: {obs_dim} (features: {len(self.feature_columns)}, cache_size: {self.cache_size})")
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # 0 = skip, 1 = cache

        # Metrics
        self.cache = []
        self.current_query_idx = 0
        self.queries_executed = 0
        self.cache_hits = 0
        self.cache_misses = 0

        # --- Add debug: confirm init complete ---
        self.logger.info("MariaDBCacheEnvironment __init__ complete.")

    def _get_default_feature_columns(self) -> List[str]:
        numeric = self.data.select_dtypes(include=[np.number]).columns.tolist()
        return numeric[:5] if numeric else self.data.columns.tolist()[:5]

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.cache.clear()
        self.current_query_idx = 0
        self.queries_executed = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.logger.info(
            f"Environment reset: max_queries={self.max_queries}, "
            f"cache_size={self.cache_size}, data_len={len(self.data)}"
        )
        self.logger.info(seed)
        q_feat = self._get_query_features(0)
        c_feat = np.zeros(self.cache_size, dtype=np.float32)
        self.logger.debug("Reset: returning initial observation")
        self.logger.info("Reset complete, returning initial observation")
        self.logger.info(f"reset() returns obs shape: {q_feat.shape} + {c_feat.shape} = {np.concatenate([q_feat, c_feat]).shape}")
        return np.concatenate([q_feat, c_feat]), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment."""
        try:
            self.queries_executed += 1
            current = self.data.iloc[self.current_query_idx]

            # Hit check & update
            hit = self._check_cache_hit(current)
            if action == 1 and not hit:
                self._update_cache(current)

            # Compute raw reward
            if self.cache_weights:
                raw = 0.0
                for col in self.cache_weights:
                    raw += float(current.get(col, 0.0))
                raw = raw if hit else -raw
            else:
                raw = 1.0 if hit else -1.0

            # Scale/clip reward to [-1,1]
            reward = float(np.tanh(raw))

            # Advance pointer
            self.current_query_idx = (self.current_query_idx + 1) % len(self.data)

            # Next observation
            q_feat = self._get_query_features(self.current_query_idx)
            c_feat = self._get_cache_features()
            obs = np.concatenate([q_feat, c_feat])

            done = self.queries_executed >= self.max_queries

            # Add detailed debug logging for every step (throttle to every 1 step for now)
            self.logger.debug(
                f"Step {self.queries_executed}/{self.max_queries} | "
                f"Current idx: {self.current_query_idx} | "
                f"Cache size: {len(self.cache)} | "
                f"Hit: {hit} | "
                f"Reward: {reward:.3f} | "
                f"Done: {done}"
            )

            self.logger.debug(f"step() returns obs shape: {obs.shape}, reward: {reward}, done: {done}")

            info = {
                "cache_hit_rate": self.cache_hits / max(1, self.queries_executed),
                "cache_miss_rate": self.cache_misses / max(1, self.queries_executed),
                "cache_size": len(self.cache),
                "queries_executed": self.queries_executed,
            }
            return obs, reward, done, False, info

        except Exception as e:
            self.logger.error(f"Error in step(): {e}")
            raise

    def _get_query_features(self, idx: int) -> np.ndarray:
        row = self.data.iloc[idx]
        feats = []
        for col in self.feature_columns:
            val = row[col]
            if not isinstance(val, (int, float)):
                val = (hash(val) % 1000) / 1000.0 if isinstance(val, str) else 0.0
            feats.append(float(val))
        return np.array(feats, dtype=np.float32)

    def _get_cache_features(self) -> np.ndarray:
        feat = np.zeros(self.cache_size, dtype=np.float32)
        for i, itm in enumerate(self.cache):
            if i >= self.cache_size:
                break
            feat[i] = 1.0
        return feat

    def _check_cache_hit(self, query) -> bool:
        for itm in self.cache:
            if all(itm[c] == query[c] for c in self.feature_columns):
                self.cache_hits += 1
                return True
        self.cache_misses += 1
        return False

    def _update_cache(self, query) -> None:
        if len(self.cache) >= self.cache_size:
            self.cache.pop(0)
        self.cache.append(query)

    def render(self, mode="human"):
        print(
            f"Executed {self.queries_executed}/{self.max_queries} | "
            f"Hits={self.cache_hits}, Misses={self.cache_misses} | "
            f"HitRate={self.cache_hits/max(1,self.queries_executed):.2f} | "
            f"CacheSize={len(self.cache)}/{self.cache_size}"
        )

    def close(self):
        # SQLAlchemy engine cleanup is automatic
        pass


def create_mariadb_cache_env(
    db_url: str = None,
    cache_size: int = 10,
    feature_columns: List[str] = None,
    max_queries: int = 500,
    table_name: str = None,
    cache_weights: Optional[List[str]] = None,
) -> MariaDBCacheEnvironment:
    """Factory to build the cache environment."""
    return MariaDBCacheEnvironment(
        db_url=db_url,
        cache_size=cache_size,
        feature_columns=feature_columns,
        max_queries=max_queries,
        table_name=table_name,
        cache_weights=cache_weights,
    )
