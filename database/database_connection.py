# database_connection.py
import logging
import pandas as pd
import os
from typing import List, Dict, Optional, Any, Union
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool
import urllib.parse


def create_database_connection(db_url: str) -> Engine:
    """
    Create a SQLAlchemy database connection with connection pooling.

    Args:
        db_url: Database connection URL

    Returns:
        SQLAlchemy Engine instance

    Raises:
        Exception: If connection fails
    """
    logger = logging.getLogger(__name__)

    try:
        # Create engine with connection pooling
        engine = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800  # Recycle connections every 30 minutes
        )

        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        # Log successful connection with masked credentials
        masked_url = mask_connection_url(db_url)
        logger.info(f"Successfully connected to database at {masked_url}")
        return engine

    except SQLAlchemyError as e:
        logger.error(f"Database connection error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error connecting to database: {e}")
        raise


def mask_connection_url(db_url: str) -> str:
    """
    Mask sensitive information in database URL for logging.

    Args:
        db_url: Database connection URL

    Returns:
        Masked URL with password hidden
    """
    try:
        parts = urllib.parse.urlparse(db_url)
        if '@' in parts.netloc:
            # Extract credentials part
            auth, host = parts.netloc.split('@', 1)
            if ':' in auth:
                username, _ = auth.split(':', 1)
                masked_auth = f"{username}:****"
                masked_netloc = f"{masked_auth}@{host}"
                parts = parts._replace(netloc=masked_netloc)
                return urllib.parse.urlunparse(parts)
        return db_url
    except Exception:
        # If parsing fails, return a completely masked URL
        return "****://****:****@****:****/****"


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


def get_table_schema(engine: Engine, table_name: str) -> Dict[str, Any]:
    """
    Get schema information for a table.

    Args:
        engine: SQLAlchemy Engine instance
        table_name: Name of the table

    Returns:
        Dictionary containing table schema information
    """
    logger = logging.getLogger(__name__)
    try:
        inspector = inspect(engine)
        if not table_exists(engine, table_name):
            logger.warning(f"Table '{table_name}' does not exist")
            return {}

        columns = inspector.get_columns(table_name)
        primary_keys = inspector.get_primary_keys(table_name)
        foreign_keys = inspector.get_foreign_keys(table_name)

        return {
            "columns": columns,
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys
        }
    except Exception as e:
        logger.error(f"Error getting schema for table {table_name}: {e}")
        return {}


def load_data_from_database(
        engine: Engine,
        table_name: str,
        limit: int = 10000,
        columns: Optional[List[str]] = None,
        order_by: Optional[str] = None,
        where_clause: Optional[str] = None
) -> pd.DataFrame:
    """
    Load data from database table into pandas DataFrame with pagination support.

    Args:
        engine: SQLAlchemy Engine instance
        table_name: Name of the table to query
        limit: Maximum number of rows to return
        columns: List of specific columns to query (None for all)
        order_by: Column to order results by
        where_clause: WHERE clause for filtering data

    Returns:
        DataFrame containing query results

    Raises:
        ValueError: If table doesn't exist
        Exception: For other query errors
    """
    logger = logging.getLogger(__name__)

    try:
        if not table_exists(engine, table_name):
            logger.error(f"Table '{table_name}' does not exist")
            raise ValueError(f"Table '{table_name}' does not exist in the database")

        # Build query
        columns_str = ", ".join(columns) if columns else "*"
        query = f"SELECT {columns_str} FROM {table_name}"

        if where_clause:
            query += f" WHERE {where_clause}"

        if order_by:
            query += f" ORDER BY {order_by}"

        query += f" LIMIT {limit}"

        logger.debug(f"Executing query: {query}")
        df = pd.read_sql(query, engine)
        logger.info(f"Loaded {len(df)} rows from {table_name}")
        return df

    except SQLAlchemyError as e:
        logger.error(f"SQL error loading data from {table_name}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {table_name}: {e}")
        raise


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