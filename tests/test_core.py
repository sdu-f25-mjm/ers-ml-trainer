import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import os
from typing import List

from ers_ml_trainer.core.cache_environment import MariaDBCacheEnvironment, create_mariadb_cache_env


@pytest.fixture
def mock_engine():
    """Create a mock SQLAlchemy engine with mock inspector that returns test tables"""
    mock = MagicMock()

    # Configure the inspector to return test tables
    mock_inspector = MagicMock()
    mock_inspector.get_table_names.return_value = ["energy_data", "production_data", "consumption_data"]

    # Setup connection mock
    mock_connection = MagicMock()
    mock.connect.return_value.__enter__.return_value = mock_connection

    return mock, mock_inspector


@pytest.fixture
def mock_dataframe():
    """Create a mock DataFrame with sample energy data"""
    return pd.DataFrame({
        'HourUTC': pd.date_range(start='2023-01-01', periods=100, freq='H'),
        'PriceArea': np.random.choice(['DK1', 'DK2'], size=100),
        'WindTotal_MWh': np.random.rand(100) * 1000,
        'SolarTotal_MWh': np.random.rand(100) * 500,
        'CentralPower_MWh': np.random.rand(100) * 2000,
        'ConsumptionTotal_MWh': np.random.rand(100) * 3000
    })


def test_table_discovery(mock_engine):
    """Test the table discovery functionality"""
    mock_db, mock_inspector = mock_engine

    with patch("ers_ml_trainer.core.cache_environment.create_database_connection", return_value=mock_db):
        with patch("sqlalchemy.inspect", return_value=mock_inspector):
            # Create environment
            env = MariaDBCacheEnvironment(
                db_url="mock://db",
                cache_size=10,
                table_name=None  # Should trigger auto-discovery
            )

            # Check if tables were discovered
            assert env.available_tables == ["energy_data", "production_data", "consumption_data"]
            # Check if first table was selected
            assert env.table_name == "energy_data"


def test_specific_table_selection(mock_engine):
    """Test selecting a specific table"""
    mock_db, mock_inspector = mock_engine

    with patch("ers_ml_trainer.core.cache_environment.create_database_connection", return_value=mock_db):
        with patch("sqlalchemy.inspect", return_value=mock_inspector):
            # Create environment with specific table
            env = MariaDBCacheEnvironment(
                db_url="mock://db",
                cache_size=10,
                table_name="production_data"
            )

            # Check that the specified table was selected
            assert env.table_name == "production_data"


def test_nonexistent_table_fallback(mock_engine, caplog):
    """Test fallback behavior when specified table doesn't exist"""
    mock_db, mock_inspector = mock_engine

    with patch("ers_ml_trainer.core.cache_environment.create_database_connection", return_value=mock_db):
        with patch("sqlalchemy.inspect", return_value=mock_inspector):
            # Create environment with non-existent table
            env = MariaDBCacheEnvironment(
                db_url="mock://db",
                cache_size=10,
                table_name="nonexistent_table"
            )

            # Check that it fell back to the first available table
            assert env.table_name == "energy_data"
            # Verify warning was logged
            assert "not found" in caplog.text


def test_empty_database(caplog):
    """Test behavior when database has no tables"""
    mock_db = MagicMock()
    mock_inspector = MagicMock()
    mock_inspector.get_table_names.return_value = []

    with patch("ers_ml_trainer.core.cache_environment.create_database_connection", return_value=mock_db):
        with patch("sqlalchemy.inspect", return_value=mock_inspector):
            # Should raise ValueError due to no tables
            with pytest.raises(ValueError, match="No tables available"):
                MariaDBCacheEnvironment(
                    db_url="mock://db",
                    cache_size=10
                )


def test_table_data_loading(mock_engine, mock_dataframe):
    """Test loading data from the selected table"""
    mock_db, mock_inspector = mock_engine

    with patch("ers_ml_trainer.core.cache_environment.create_database_connection", return_value=mock_db):
        with patch("sqlalchemy.inspect", return_value=mock_inspector):
            with patch("pandas.read_sql", return_value=mock_dataframe):
                # Create environment
                env = MariaDBCacheEnvironment(
                    db_url="mock://db",
                    cache_size=10,
                    table_name="energy_data"
                )

                # Check that data was loaded
                assert len(env.data) == 100
                assert "WindTotal_MWh" in env.data.columns


def test_create_mariadb_cache_env_function(mock_engine, mock_dataframe):
    """Test the create_mariadb_cache_env factory function"""
    mock_db, mock_inspector = mock_engine

    with patch("ers_ml_trainer.core.cache_environment.create_database_connection", return_value=mock_db):
        with patch("sqlalchemy.inspect", return_value=mock_inspector):
            with patch("pandas.read_sql", return_value=mock_dataframe):
                # Test with explicit table name
                env = create_mariadb_cache_env(
                    db_url="mock://db",
                    cache_size=15,
                    feature_columns=["HourUTC", "PriceArea"],
                    table_name="consumption_data"
                )

                # Verify parameters were passed correctly
                assert env.table_name == "consumption_data"
                assert env.cache_size == 15
                assert env.feature_columns == ["HourUTC", "PriceArea"]


def test_environment_reset_step(mock_engine, mock_dataframe):
    """Test reset and step functionality of the environment"""
    mock_db, mock_inspector = mock_engine

    with patch("ers_ml_trainer.core.cache_environment.create_database_connection", return_value=mock_db):
        with patch("sqlalchemy.inspect", return_value=mock_inspector):
            with patch("pandas.read_sql", return_value=mock_dataframe):
                # Create environment
                env = create_mariadb_cache_env(
                    db_url="mock://db",
                    cache_size=5,
                    max_queries=100
                )

                # Test reset
                obs, info = env.reset(seed=42)
                assert len(obs) == len(env.feature_columns) + env.cache_size
                assert env.cache_hits == 0
                assert env.cache == []

                # Test step
                obs, reward, done, truncated, info = env.step(0)
                assert isinstance(reward, float)
                assert isinstance(done, bool)
                assert isinstance(info, dict)
                assert "cache_hit_rate" in info


def test_error_handling_during_connection(caplog):
    """Test error handling when database connection fails"""
    with patch("ers_ml_trainer.core.cache_environment.create_database_connection",
               side_effect=Exception("Connection error")):
        with pytest.raises(Exception, match="Connection error"):
            MariaDBCacheEnvironment(db_url="invalid://url")


def test_environment_with_env_variables():
    """Test that environment variables are used when parameters are not provided"""
    mock_db = MagicMock()
    mock_inspector = MagicMock()
    mock_inspector.get_table_names.return_value = ["test_table"]
    mock_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

    with patch("ers_ml_trainer.core.cache_environment.create_database_connection", return_value=mock_db):
        with patch("sqlalchemy.inspect", return_value=mock_inspector):
            with patch("pandas.read_sql", return_value=mock_df):
                with patch.dict(os.environ, {"DB_URL": "env://var/db"}):
                    env = create_mariadb_cache_env()
                    # Should use the DB_URL from environment
                    assert env.db_url == "env://var/db"