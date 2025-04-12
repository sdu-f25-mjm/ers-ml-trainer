# database/database_connection.py
import logging
import random
import time
import urllib.parse
from typing import List, Dict, Optional, Any, Union

import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError, OperationalError, DatabaseError
from sqlalchemy.pool import QueuePool


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
def create_database_connection(db_url: str) -> Engine:
    """
    Create a SQLAlchemy database connection with connection pooling and retry logic.

    Args:
        db_url: Database connection URL

    Returns:
        SQLAlchemy Engine instance

    Raises:
        Exception: If connection fails after retries
    """
    logger = logging.getLogger(__name__)
    try:
        # Create engine with connection pooling and improved parameters, using valid keyword for connection timeout.
        engine = create_engine(
            db_url,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,  # Recycle connections every 30 minutes
            pool_pre_ping=True,
            connect_args={'connection_timeout': 10}  # Updated keyword argument
        )
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
