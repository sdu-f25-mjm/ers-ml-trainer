# core/cache_environment.py
import gymnasium as gym
import numpy as np
import pandas as pd
import logging
import mysql.connector
import os
from typing import List, Dict, Tuple, Any, Optional
from gymnasium import spaces
from database_connection import create_database_connection
from sqlalchemy import inspect, text


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
            feature_columns: List[str] = None,
            target_column: str = None,
            max_queries: int = 500,
            table_name: str = None  # Optional table name parameter
    ):
        """
        Initialize the MariaDB cache environment.

        Args:
            db_url: The database connection URL
            cache_size: Size of the cache (number of rows)
            feature_columns: Features to use for observation space
            target_column: Target column for optimization
            max_queries: Maximum number of queries to run
            table_name: Optional specific table to use (will auto-discover if None)
        """
        super().__init__()

        self.logger = logging.getLogger(__name__)
        self.cache_size = cache_size
        self.max_queries = max_queries
        self.db_url = db_url or os.environ.get('DB_URL',
                                               'mysql+mysqlconnector://cacheuser:cachepass@mariadb:3306/cache_db')

        # Create database connection
        self.logger.info(f"Connecting to database: {self.db_url}")
        try:
            self.engine = create_database_connection(self.db_url)
            self.logger.info("Database connection established")
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            raise

        # Auto-discover available tables if not specified
        self.available_tables = self._get_available_tables()
        if not self.available_tables:
            self.logger.error("No tables found in database")
            raise ValueError("No tables available in the database")

        # Use specified table if provided and exists, otherwise use first available
        if table_name and table_name in self.available_tables:
            self.table_name = table_name
        else:
            if table_name:
                self.logger.warning(
                    f"Specified table '{table_name}' not found. Using '{self.available_tables[0]}' instead.")
            else:
                self.logger.info(f"No table specified. Using '{self.available_tables[0]}'")
            self.table_name = self.available_tables[0]

        # Load data sample to get schema
        try:
            self.data = self._load_table_data()
            self.logger.info(f"Loaded {len(self.data)} rows from {self.table_name}")
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise

        # Set feature columns
        if feature_columns is None:
            # Use numeric columns for features if not specified
            numeric_cols = self.data.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) > 5:  # Limit to prevent observation space explosion
                numeric_cols = numeric_cols[:5]
            self.feature_columns = numeric_cols
            self.logger.info(f"No feature columns specified. Using {self.feature_columns}")
        else:
            # Validate feature columns exist in the data
            valid_columns = [col for col in feature_columns if col in self.data.columns]
            if not valid_columns:
                raise ValueError(f"None of the specified feature columns {feature_columns} exist in the data")
            self.feature_columns = valid_columns

        # Set target column
        if target_column is None:
            # Use first numeric column as target if not specified
            numeric_cols = self.data.select_dtypes(include=['number']).columns.tolist()
            self.target_column = numeric_cols[0] if numeric_cols else self.data.columns[0]
            self.logger.info(f"No target column specified. Using {self.target_column}")
        else:
            if target_column not in self.data.columns:
                raise ValueError(f"Target column {target_column} does not exist in the data")
            self.target_column = target_column

        # Initialize cache
        self.cache = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_queries = 0
        self.current_query_index = 0

        # Define observation and action spaces
        # Observation: Features of data + cache state
        feature_dim = len(self.feature_columns)
        cache_status_dim = self.cache_size

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(feature_dim + cache_status_dim,),
            dtype=np.float32
        )

        # Action space: Which item to evict from cache (0 to cache_size-1) or no eviction (cache_size)
        self.action_space = spaces.Discrete(self.cache_size + 1)

        # Generate query sequence
        self._generate_query_sequence()

        self.logger.info(f"MariaDBCacheEnvironment initialized with cache size {self.cache_size}")

    def _get_available_tables(self) -> List[str]:
        """Discover available tables in the database"""
        try:
            # Use SQLAlchemy's inspector to get table names
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            self.logger.info(f"Found tables: {tables}")
            return tables
        except Exception as e:
            self.logger.error(f"Error discovering tables: {e}")
            return []

    def _load_table_data(self) -> pd.DataFrame:
        """Load data from the selected table"""
        try:
            # Use a limit to avoid loading too much data
            query = f"SELECT * FROM {self.table_name} LIMIT 10000"
            df = pd.read_sql(query, self.engine)
            return df
        except Exception as e:
            self.logger.error(f"Failed to load data from {self.table_name}: {e}")
            raise

    def _generate_query_sequence(self):
        """Generate a sequence of database queries to simulate"""
        # Generate query indices following a power law distribution
        data_size = len(self.data)
        if data_size == 0:
            raise ValueError(f"No data available in table {self.table_name}")

        # Power law distribution - some items are accessed more frequently
        a = 1.5  # Power law exponent
        weights = np.power(np.arange(1, data_size + 1), -a)
        weights = weights / weights.sum()

        self.query_indices = np.random.choice(
            data_size,
            size=self.max_queries,
            p=weights
        )

        self.logger.info(f"Generated {self.max_queries} queries with power law distribution")

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)

        # Clear cache
        self.cache = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_queries = 0
        self.current_query_index = 0

        # Get first observation
        obs = self._get_observation()

        return obs, {}

    def _get_observation(self) -> np.ndarray:
        """Get the current observation"""
        # Get current query features
        if self.current_query_index >= len(self.query_indices):
            features = np.zeros(len(self.feature_columns))
        else:
            idx = self.query_indices[self.current_query_index]
            query_row = self.data.iloc[idx]
            features = query_row[self.feature_columns].values.astype(np.float32)

        # Create cache status representation
        cache_status = np.zeros(self.cache_size, dtype=np.float32)
        for i in range(min(len(self.cache), self.cache_size)):
            cache_status[i] = 1.0

        # Combine into observation
        obs = np.concatenate([features, cache_status])
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment."""
        # Check if we've reached the query limit
        if self.current_query_index >= self.max_queries:
            return self._get_observation(), 0.0, True, False, self._get_info()

        # Process the current query
        query_idx = self.query_indices[self.current_query_index]
        self.total_queries += 1

        # Check if the query is in cache (cache hit)
        is_cache_hit = query_idx in self.cache

        if is_cache_hit:
            self.cache_hits += 1
            reward = 1.0  # Reward for cache hit
        else:
            self.cache_misses += 1
            reward = -0.1  # Small penalty for cache miss

            # Apply the action (cache eviction and addition)
            if len(self.cache) >= self.cache_size:
                # Only evict if cache is full
                if action < len(self.cache):  # Valid eviction action
                    self.cache.pop(action)  # Remove item at action index
                    self.cache.append(query_idx)  # Add new item to cache
            else:
                # Cache not full, just add the item
                self.cache.append(query_idx)

        # Move to next query
        self.current_query_index += 1

        # Check if we're done
        done = self.current_query_index >= self.max_queries

        # Get new observation
        obs = self._get_observation()

        return obs, reward, done, False, self._get_info()

    def _get_info(self) -> Dict:
        """Get additional info about the environment state"""
        hit_rate = self.cache_hits / max(1, self.total_queries)
        return {
            "cache_size": self.cache_size,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_queries": self.total_queries,
            "cache_hit_rate": hit_rate,
            "current_query": self.current_query_index,
            "max_queries": self.max_queries
        }

    def render(self, mode="human"):
        """Render the environment"""
        if mode == "human":
            hit_rate = self.cache_hits / max(1, self.total_queries)
            print(f"Step: {self.current_query_index}/{self.max_queries}, "
                  f"Hit Rate: {hit_rate:.4f}")

    def close(self):
        """Close resources"""
        if hasattr(self, 'engine'):
            self.engine.dispose()


def create_mariadb_cache_env(
        db_url: str = None,
        cache_size: int = 10,
        feature_columns: List[str] = None,
        target_column: str = None,
        max_queries: int = 500,
        table_name: str = None
):
    """
    Create a MariaDB cache environment with dynamic table discovery.

    Args:
        db_url: Database connection URL
        cache_size: Size of the cache (number of rows)
        feature_columns: Columns to use as features for decision making
        target_column: Column to optimize for
        max_queries: Maximum number of queries to execute
        table_name: Specific table to use, if None will discover available tables

    Returns:
        MariaDBCacheEnvironment instance
    """
    import os
    import logging

    logger = logging.getLogger(__name__)

    # Use environment variables with fallbacks
    db_url = db_url or os.environ.get('DB_URL', 'mysql+mysqlconnector://cacheuser:cachepass@mariadb:3306/cache_db')

    if feature_columns is None:
        # Default features from OpenAPI schema fields that are useful for caching
        feature_columns = ["HourUTC", "PriceArea"]

    logger.info(f"Creating cache environment with DB_URL: {db_url}")
    logger.info(f"Table name specified: {table_name or 'Auto-discovery'}")

    # Create and return the environment
    return MariaDBCacheEnvironment(
        db_url=db_url,
        cache_size=cache_size,
        feature_columns=feature_columns,
        target_column=target_column,
        max_queries=max_queries,
        table_name=table_name
    )