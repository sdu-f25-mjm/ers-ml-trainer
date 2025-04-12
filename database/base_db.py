# database/base_db.py
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class DatabaseHandler(ABC):
    """Abstract base class for database handlers"""

    @abstractmethod
    def connect(self, host, port, user, password, database):
        """Connect to the database"""
        pass

    @abstractmethod
    def create_database_if_not_exists(self, database_name):
        """Create a database if it doesn't exist"""
        pass

    @abstractmethod
    def execute_query(self, query, params=None):
        """Execute a SQL query with parameters"""
        pass

    @abstractmethod
    def commit(self):
        """Commit the current transaction"""
        pass

    @abstractmethod
    def close(self):
        """Close the database connection"""
        pass

    @abstractmethod
    def get_placeholder_symbol(self):
        """Get the placeholder symbol for parameterized queries"""
        pass

    @abstractmethod
    def get_auto_increment_syntax(self):
        """Get the syntax for auto-increment columns"""
        pass

    @abstractmethod
    def get_timestamp_type(self):
        """Get the data type for timestamp columns"""
        pass