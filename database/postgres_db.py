# database/postgres_db.py
import logging

import psycopg2
from psycopg2.extras import DictCursor

from database.base_db import DatabaseHandler

logger = logging.getLogger(__name__)


class PostgreSQLHandler(DatabaseHandler):
    """PostgreSQL-specific database handler"""

    def __init__(self):
        self.conn = None
        self.cursor = None

    def connect(self, host, port, user, password, database=None):
        """Connect to PostgreSQL database"""
        try:
            # Connect to postgres database first
            self.conn = psycopg2.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                dbname="postgres"  # Connect to default postgres database
            )
            self.conn.autocommit = True  # Required for creating databases
            self.cursor = self.conn.cursor(cursor_factory=DictCursor)
            logger.info(f"Connected to PostgreSQL at {host}:{port}")

            if database:
                # Create database if not exists
                self.create_database_if_not_exists(database)

                # Close connection to postgres and connect to the specified database
                self.close()
                self.conn = psycopg2.connect(
                    host=host,
                    port=port,
                    user=user,
                    password=password,
                    dbname=database
                )
                self.cursor = self.conn.cursor(cursor_factory=DictCursor)
                logger.info(f"Using database: {database}")

            return True
        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL: {e}")
            return False

    def create_database_if_not_exists(self, database_name):
        """Create a database if it doesn't exist"""
        try:
            # Check if database exists
            self.cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (database_name,)
            )
            if not self.cursor.fetchone():
                self.cursor.execute(f"CREATE DATABASE {database_name}")
                logger.info(f"Created database: {database_name}")
            return True
        except Exception as e:
            logger.error(f"Error creating database: {e}")
            return False

    def execute_query(self, query, params=None):
        """Execute a SQL query with parameters"""
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            return self.cursor
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return None

    def commit(self):
        """Commit the current transaction"""
        if self.conn:
            self.conn.commit()

    def close(self):
        """Close the database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None

    def get_placeholder_symbol(self):
        """Get the placeholder symbol for parameterized queries"""
        return "%s"

    def get_auto_increment_syntax(self):
        """Get the syntax for auto-increment columns"""
        return "SERIAL"

    def get_timestamp_type(self):
        """Get the data type for timestamp columns"""
        return "TIMESTAMP"
