# mock/mock_db.py
import logging
import os
import random
from datetime import datetime, timedelta

from database.create_tables import create_tables
from mock.simulate_live import simulate_visits

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/application.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
DB_DRIVER = os.getenv("DB_DRIVER", "mysql")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", 3306)
DB_NAME = os.getenv("DB_NAME", "cache_db")
DB_USER = os.getenv("DB_USER", "cacheuser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "cachepass")


def get_random_date_between(start_date, end_date):
    """
    Generate a random date between two given dates.

    Args:
        start_date (datetime): The starting date.
        end_date (datetime): The ending date.

    Returns:
        datetime: A random date between start_date and end_date.
    """
    if start_date > end_date:
        raise ValueError("start_date must be earlier than end_date")

    # Calculate the total seconds between the two dates
    total_seconds = int((end_date - start_date).total_seconds())

    # Generate a random number of seconds within this range
    random_seconds = random.randint(0, total_seconds)

    # Add the random seconds to the start_date
    random_date = start_date + timedelta(seconds=random_seconds)
    return random_date


def get_db_handler(db_type):
    """Get the appropriate database handler based on type"""
    if db_type == 'mysql':
        from database.mysql_db import MySQLHandler
        return MySQLHandler()
    elif db_type == 'postgres':
        from database.postgres_db import PostgreSQLHandler
        return PostgreSQLHandler()
    elif db_type == 'sqlite':
        from database.sqlite_db import SQLiteHandler
        return SQLiteHandler()
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


def generate_production_data(db_handler, hours, price_areas, types):
    placeholder = db_handler.get_placeholder_symbol()
    insert_query = f"""
    INSERT INTO production (
        hourUTC, price_area_id, central_power, local_power, commercial_power,
        commercial_power_self_consumption, offshore_wind_lt_100mw, offshore_wind_ge_100mw,
        onshore_wind_lt_50kw, onshore_wind_ge_50kw, hydro_power, solar_power_lt_10kw,
        solar_power_ge_10kw_lt_40kw, solar_power_ge_40kw, solar_power_self_consumption,
        unknown_production
    ) VALUES (
        {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder},
        {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder},
        {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}
    )
    """
    for hour_offset in range(hours):
        ts = get_random_date_between(datetime(2020, 1, 1, 0, 0, 0, 0), datetime.now())
        for price_area_id in [1, 2]:  # DK1=1, DK2=2
            params = (
                ts,
                price_area_id,
                random.randint(100, 1000),  # central_power
                random.randint(50, 500),  # local_power
                random.randint(50, 500),  # commercial_power
                random.randint(10, 100),  # commercial_power_self_consumption
                random.randint(10, 200),  # offshore_wind_lt_100mw
                random.randint(10, 200),  # offshore_wind_ge_100mw
                random.randint(10, 200),  # onshore_wind_lt_50kw
                random.randint(10, 200),  # onshore_wind_ge_50kw
                random.randint(10, 200),  # hydro_power
                random.randint(10, 200),  # solar_power_lt_10kw
                random.randint(10, 200),  # solar_power_ge_10kw_lt_40kw
                random.randint(10, 200),  # solar_power_ge_40kw
                random.randint(10, 200),  # solar_power_self_consumption
                random.randint(10, 200),  # unknown_production
            )
            db_handler.execute_query(insert_query, params)
    db_handler.commit()
    logger.info("Mock production data generation completed.")


def generate_consumption_data(db_handler, hours, price_areas):
    placeholder = db_handler.get_placeholder_symbol()
    insert_query = f"""
    INSERT INTO consumption (
        hourUTC, municipality_id, branche_id, consumption
    ) VALUES (
        {placeholder}, {placeholder}, {placeholder}, {placeholder}
    )
    """
    for i in range(hours):
        ts = get_random_date_between(datetime(2020, 1, 1, 0, 0, 0, 0), datetime.now())
        for area_id in [1, 2]:  # DK1=1, DK2=2
            municipality_id = random.choice(range(101, 1000))  # Use seeded IDs
            branche_id = random.choice([1, 2, 3])  # 1=Erhverv, 2=Offentligt, 3=Privat
            consumption = random.randint(100, 1000)
            params = (ts, municipality_id, branche_id, consumption)
            db_handler.execute_query(insert_query, params)
    db_handler.commit()
    logger.info("Mock consumption data generation completed.")


def generate_exchange_data(db_handler, hours, countries):
    placeholder = db_handler.get_placeholder_symbol()
    insert_query = f"""
    INSERT INTO exchange (
        hourUTC, price_area_id, exchange_to_norway, exchange_to_sweden,
        exchange_to_germany, exchange_to_netherlands, exchange_to_gb,
        exchange_over_the_great_belt
    ) VALUES (
        {placeholder}, {placeholder}, {placeholder}, {placeholder},
        {placeholder}, {placeholder}, {placeholder}, {placeholder}
    )
    """
    for hour in range(hours):
        ts = get_random_date_between(datetime(2020, 1, 1, 0, 0, 0, 0), datetime.now())
        for price_area_id in [1, 2]:  # DK1=1, DK2=2
            params = (
                ts,
                price_area_id,
                random.randint(10, 200),  # exchange_to_norway
                random.randint(10, 200),  # exchange_to_sweden
                random.randint(10, 200),  # exchange_to_germany
                random.randint(10, 200),  # exchange_to_netherlands
                random.randint(10, 200),  # exchange_to_gb
                random.randint(10, 200),  # exchange_over_the_great_belt
            )
            db_handler.execute_query(insert_query, params)
    db_handler.commit()
    logger.info("Mock exchange data generation completed.")


def generate_mock_database(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT,
        hours=1000,
        db_type=DB_DRIVER,
        data_types=None
):
    """
    Generate a mock database with sample data.

    For cache_metrics, cache_name is now a normalized endpoint template (no date/time params)
    to improve RL model generalization and cache efficiency.
    """
    logger.info(data_types)
    # Convert enums to their value strings if needed.
    if data_types is not None:
        if not isinstance(data_types, list):
            data_types = [data_types]
        data_types = [dt.value if hasattr(dt, "value") else dt for dt in data_types]
        logger.info(f"Data types provided: {data_types}")
        if len(data_types) == 0:
            logger.warning("No data_types selected, skipping mock data generation.")
            return True
    else:
        data_types = [
            "production", "consumption", "exchange_data", "cache_metrics"
        ]

    price_areas = ["DK1", "DK2"]
    countries = ["germany", "greatbritain", "netherlands", "norway", "sweden"]
    types = ["wind", "solar", "hydro", "commercialPower", "centralPower", "local"]

    if db_type not in ['mysql', 'postgres', 'sqlite']:
        logger.error(f"Unsupported database type: {db_type}")
        return False

    # If no selection provided, generate all mock data.
    if data_types is None:
        data_types = [
            "production", "consumption_data", "exchange_data", "cache_metrics"
        ]

    logger.info(f"Generating mock database for {db_type} with {hours} hours of data for: {', '.join(price_areas)}")
    try:
        db_handler = get_db_handler(db_type)

        if db_type == 'sqlite':
            if not db_handler.connect('', 0, '', '', database):
                logger.error("Failed to connect to SQLite database")
                return False
        else:
            if not db_handler.connect(host, port, user, password, database):
                logger.error(f"Failed to connect to {db_type} database")
                return False

        # Create tables based on the schema
        create_tables(db_handler)

        # Generate mock data based on selection

        if "production" in data_types:
            logger.info("Generating mock production data")
            generate_production_data(db_handler, hours, price_areas, types)
        if "consumption_data" in data_types:
            logger.info("Generating mock consumption data")
            generate_consumption_data(db_handler, hours, price_areas)
        if "exchange_data" in data_types:
            logger.info("Generating mock exchange data")
            generate_exchange_data(db_handler, hours, countries)
        if "cache_metrics" in data_types:
            logger.info("Simulating visits to generate cache_metrics via API endpoints")
            simulate_visits(
                n=hours,
                sleep=0,
                run_duration=None,
                stop_event=None
            )

        db_handler.commit()
        db_handler.close()

        logger.info("Mock database generation completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error generating mock database: {e}")
        return False
