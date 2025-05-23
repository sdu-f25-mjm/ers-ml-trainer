# database/database_connection.py
import random
import time
from typing import List, Dict, Optional, Any, Union

import dotenv
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError, OperationalError

from core.utils import build_db_url

dotenv.load_dotenv()

build_db_url()


def retry_with_backoff(max_retries=5, initial_backoff=1, max_backoff=60):
    """
    Decorator that retries the function with exponential backoff on database errors.

    Args:
        max_retries: Maximum number of retry attempts
        initial_backoff: Initial backoff time in seconds
        max_backoff: Maximum backoff time in seconds
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            backoff = initial_backoff
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except (SQLAlchemyError, OperationalError) as e:
                    retries += 1
                    if retries > max_retries:
                        raise e
                    time.sleep(backoff)
                    backoff = min(backoff * 2, max_backoff)

        return wrapper

    return decorator


@retry_with_backoff()
def create_database_connection(db_url=None) -> Engine:
    """
    Create a SQLAlchemy database connection with connection pooling and retry logic.

    Args:
        db_url: Optional database URL. If not provided, will use build_db_url()
    """
    logger = logging.getLogger(__name__)
    try:
        if db_url is None:
            db_url = build_db_url()

        # Set connection arguments based on database type
        connect_args = {}
        if db_url.startswith('mysql') or db_url.startswith('mariadb'):
            connect_args['connect_timeout'] = 10

        # Common engine parameters for all database types
        engine_params = {
            'pool_size': 5,
            'max_overflow': 10,
            'pool_timeout': 30,
            'pool_recycle': 1800,
            'pool_pre_ping': True
        }

        # Add connect_args only if they're not empty
        if connect_args:
            engine_params['connect_args'] = connect_args

        engine = create_engine(db_url, **engine_params)

        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Successfully connected to database.")
        return engine
    except SQLAlchemyError as e:
        logger.error(f"Database connection error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error connecting to database: {e}")
        raise


def check_connection_health(engine: Engine) -> bool:
    """
    Check if database connection is healthy.

    Args:
        engine: SQLAlchemy Engine instance

    Returns:
        True if connection is healthy, False otherwise
    """
    logger = logging.getLogger(__name__)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1")).fetchone()
        return True
    except Exception as e:
        logger.warning(f"Database connection health check failed: {e}")
        return False


def get_available_tables(engine: Engine) -> List[str]:
    """
    Discover available tables in the database.

    Args:
        engine: SQLAlchemy Engine instance

    Returns:
        List of table names
    """
    logger = logging.getLogger(__name__)
    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        logger.info(f"Found {len(tables)} tables in database")
        return tables
    except Exception as e:
        logger.error(f"Error discovering tables: {e}")
        return []


def table_exists(engine: Engine, table_name: str) -> bool:
    """
    Check if a specific table exists in the database.

    Args:
        engine: SQLAlchemy Engine instance
        table_name: Name of the table to check

    Returns:
        True if table exists, False otherwise
    """
    try:
        tables = get_available_tables(engine)
        return table_name in tables
    except Exception:
        return False


def get_table_schema(engine, table_name: str) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    try:
        inspector = inspect(engine)
        if not table_exists(engine, table_name):
            logger.warning(f"Table '{table_name}' does not exist")
            return {}

        columns = inspector.get_columns(table_name)
        pk_constraint = inspector.get_pk_constraint(table_name)
        primary_keys = pk_constraint.get('constrained_columns', [])
        foreign_keys = inspector.get_foreign_keys(table_name)

        return {
            "columns": columns,
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys
        }
    except Exception as e:
        logger.error(f"Error getting schema for table {table_name}: {e}")
        return {}


def load_data_from_database(engine, table_name: str, limit: int = 1000,
                            order_by: Optional[str] = None, where_clause: Optional[str] = None) -> pd.DataFrame:
    query = f"SELECT * FROM {table_name}"
    if where_clause:
        query += f" WHERE {where_clause}"
    if order_by:
        query += f" ORDER BY {order_by}"
    query += f" LIMIT {limit}"
    try:
        return pd.read_sql_query(text(query), con=engine)
    except Exception as e:
        logging.getLogger(__name__).error(f"Error loading data from {table_name}: {e}")
        return pd.DataFrame()


def execute_query(
        engine: Engine,
        query: str,
        params: Optional[Dict[str, Any]] = None
) -> Union[pd.DataFrame, int]:
    """
    Execute a custom SQL query with proper parameter binding.

    Args:
        engine: SQLAlchemy Engine instance
        query: SQL query string
        params: Dictionary of query parameters

    Returns:
        DataFrame for SELECT queries or row count for other queries
    """
    logger = logging.getLogger(__name__)

    try:
        if query.strip().lower().startswith("select"):
            # For SELECT queries, return DataFrame
            return pd.read_sql(query, engine, params=params)
        else:
            # For other queries (INSERT, UPDATE, DELETE), return affected rows
            with engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                conn.commit()
                return result.rowcount
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        raise


def get_database_connection(db_url, max_retries=5, initial_backoff=1, max_backoff=30):
    """Create a database engine with retry logic."""
    logger = logging.getLogger(__name__)
    retries = 0
    backoff = initial_backoff

    while True:
        try:
            logger.info(f"Attempting to connect to database (attempt {retries + 1}/{max_retries})...")

            # Set connection arguments based on database type
            connect_args = {}
            if db_url.startswith('mysql') or db_url.startswith('mariadb'):
                connect_args['connect_timeout'] = 10

            # Common engine parameters for all database types
            engine_params = {
                'pool_size': 5,
                'max_overflow': 10,
                'pool_timeout': 30,
                'pool_recycle': 1800,
                'pool_pre_ping': True
            }

            # Add connect_args only if they're not empty
            if connect_args:
                engine_params['connect_args'] = connect_args

            engine = create_engine(db_url, **engine_params)

            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            masked_url = db_url.replace("://", "://***:***@", 1).split("@")[-1]
            logger.info(f"Successfully connected to database at {masked_url}")
            return engine

        except (SQLAlchemyError, OperationalError) as e:
            # Rest of your existing retry logic
            retries += 1
            if retries > max_retries:
                logger.error(f"Failed to connect to database after {max_retries} attempts: {e}")
                return None
            jitter = random.uniform(0, 0.1 * backoff)
            sleep_time = backoff + jitter
            logger.warning(f"Database connection failed. Retrying in {sleep_time:.2f}s. Error: {e}")
            time.sleep(sleep_time)
            backoff = min(backoff * 2, max_backoff)
        except Exception as e:
            logger.error(f"Unexpected error connecting to database: {e}")
            return None

from sqlalchemy import text
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def save_best_model_base64(engine, metadata: dict, model_base64: str, input_dimension: int):
    """
    Insert a base64–encoded model into the rl_models table, including
    input_dimension and all of the extended metadata fields.
    """
    # extract top-level fields
    algorithm   = metadata.get("algorithm")
    device      = metadata.get("device")
    cache_size  = metadata.get("cache_size")
    max_queries = metadata.get("max_queries")
    table_name  = metadata.get("table_name")
    timesteps   = metadata.get("timesteps")
    db_type     = metadata.get("db_type")
    trained_at  = metadata.get("trained_at") or datetime.utcnow().isoformat()

    # extract nested fields
    feature_columns       = metadata.get("feature_columns", [])
    cache_weights         = metadata.get("cache_weights", [])
    hyperparameters       = metadata.get("hyperparameters", {})
    network_architecture  = metadata.get("network_architecture", {})
    final_metrics         = metadata.get("final_metrics", {})
    reward_history        = final_metrics.get("reward_history", [])
    hit_rate_history      = final_metrics.get("hit_rate_history", [])
    training_duration_sec = final_metrics.get("training_duration_seconds")

    stmt = text("""
    INSERT INTO rl_models (
        algorithm,
        created_at,
        model_base64,
        device,
        cache_size,
        max_queries,
        table_name,
        timesteps,
        feature_columns,
        cache_weights,
        db_type,
        hyperparameters,
        network_architecture,
        reward_history,
        hit_rate_history,
        training_duration_seconds,
        input_dimension,
        trained_at
    )
    VALUES (
        :algorithm,
        NOW(),
        :model_base64,
        :device,
        :cache_size,
        :max_queries,
        :table_name,
        :timesteps,
        :feature_columns,
        :cache_weights,
        :db_type,
        :hyperparameters,
        :network_architecture,
        :reward_history,
        :hit_rate_history,
        :training_duration_seconds,
        :input_dimension,
        :trained_at
    )
    """)

    params = {
        "algorithm":                    algorithm,
        "model_base64":                 model_base64,
        "device":                       device,
        "cache_size":                   cache_size,
        "max_queries":                  max_queries,
        "table_name":                   table_name,
        "timesteps":                    timesteps,
        "feature_columns":              json.dumps(feature_columns),
        "cache_weights":                json.dumps(cache_weights),
        "db_type":                      db_type,
        "hyperparameters":              json.dumps(hyperparameters),
        "network_architecture":         json.dumps(network_architecture),
        "reward_history":               json.dumps(reward_history),
        "hit_rate_history":             json.dumps(hit_rate_history),
        "training_duration_seconds":    training_duration_sec,
        "input_dimension":              input_dimension,
        "trained_at":                   trained_at,
    }

    with engine.connect() as conn:
        try:
            logger.info("Executing SQL statement to save best model: %s", stmt)
            conn.execute(stmt, params)
            conn.commit()
            logger.info("Successfully saved model to rl_models.")
        except Exception as e:
            logger.error("Failed to save best model to database: %s", e)
            raise