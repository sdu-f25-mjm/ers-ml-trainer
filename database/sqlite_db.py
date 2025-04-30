# database/sqlite_db.py
import logging
import os
import sqlite3

from database.base_db import DatabaseHandler

logger = logging.getLogger(__name__)


class SQLiteHandler(DatabaseHandler):
    """SQLite-specific database handler"""

    def __init__(self):
        self.conn = None
        self.cursor = None

    def connect(self, host, port, user, password, database):
        """Connect to SQLite database (ignores host/port/user/password)"""
        try:
            # Ensure directory exists
            db_dir = os.path.dirname(database)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)

            self.conn = sqlite3.connect(database)
            # Make sqlite return dictionaries
            self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()
            logger.info(f"Connected to SQLite database: {database}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to SQLite: {e}")
            return False

    def create_database_if_not_exists(self, database_name):
        """No-op for SQLite as database is created with connection"""
        return True

    def execute_query(self, query, params=None):
        """Execute a SQL query with parameters"""
        try:
            # Replace MySQL-specific syntax with SQLite syntax
            query = query.replace("AUTO_INCREMENT", "AUTOINCREMENT")

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
        return "?"

    def get_auto_increment_syntax(self):
        """Get the syntax for auto-increment columns"""
        return "AUTOINCREMENT"

    def get_timestamp_type(self):
        """Get the data type for timestamp columns"""
        return "TIMESTAMP"
