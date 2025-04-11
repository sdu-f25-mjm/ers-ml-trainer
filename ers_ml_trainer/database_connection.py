# database_connection.py
import logging
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text

def create_database_connection(db_url: str):
    """Create a SQLAlchemy database connection"""
    logger = logging.getLogger(__name__)
    try:
        engine = create_engine(db_url)
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info(f"Successfully connected to database at {db_url}")
        return engine
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise

def load_data_from_database(engine, table_name: str):
    """Load data from database table into pandas DataFrame"""
    logger = logging.getLogger(__name__)
    try:
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, engine)
        logger.info(f"Loaded {len(df)} rows from {table_name}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data from {table_name}: {e}")
        raise