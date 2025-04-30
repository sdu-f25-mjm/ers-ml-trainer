# database/mysql_db.py
import logging

import mysql.connector

from database.base_db import DatabaseHandler

logger = logging.getLogger(__name__)


class MySQLHandler(DatabaseHandler):
    """MySQL-specific database handler"""

    def __init__(self):
        self.conn = None
        self.cursor = None

    def connect(self, host, port, user, password, database=None):
        """Connect to MySQL database"""
        try:
            # Connect without specifying a database first
            self.conn = mysql.connector.connect(
                host=host,
                port=port,
                user=user,
                password=password
            )
            self.cursor = self.conn.cursor(dictionary=True)
            logger.info(f"Connected to MySQL at {host}:{port}")

            # If database is specified, select it
            if database:
                self.create_database_if_not_exists(database)
                self.cursor.execute(f"USE {database}")
                logger.info(f"Using database: {database}")

            return True
        except Exception as e:
            logger.error(f"Error connecting to MySQL: {e}")
            return False

    def create_database_if_not_exists(self, database_name):
        """Create a database if it doesn't exist"""
        try:
            self.cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name}")
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

    def get_placeholder_symbol(self):
        """Get the placeholder symbol for parameterized queries"""
        return "%s"

    def get_auto_increment_syntax(self):
        """Get the syntax for auto-increment columns"""
        return "AUTO_INCREMENT"

    def get_timestamp_type(self):
        """Get the data type for timestamp columns"""
        return "DATETIME"
