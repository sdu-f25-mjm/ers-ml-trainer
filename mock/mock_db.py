# mock/mock_db.py
import json
import logging
import os
import random
from datetime import datetime, timedelta

from database.create_tables import create_tables
from mock.simulate_live import simulate_visits
from mock.simulation import simulate_cache_metrics

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


def generate_energy_data(db_handler, hours, price_areas):
    """
    Generate mock energy data for a given number of hours for each price area and insert the data into the energy_data table.

    Args:
        db_handler: A database handler instance.
        hours (int): Number of hourly data points to generate.
        price_areas (list): List of price area strings (e.g., ["DK1", "DK2"]).
    """
    start_time = datetime.utcnow()
    placeholder = db_handler.get_placeholder_symbol()

    # Build the INSERT query using the handler's placeholder symbol.
    query = f"""
    INSERT INTO energy_data (
        timestamp, price_area, central_power_mwh, local_power_mwh,
        gross_consumption_mwh, exchange_no_mwh, exchange_se_mwh, exchange_de_mwh,
        solar_power_self_con_mwh, grid_loss_transmission_mwh
    ) VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
    """

    for hour in range(hours):
        # Calculate timestamp for current hour
        hour_timestamp = get_random_date_between(datetime(1970, 1, 1, 0, 0, 0, 000), datetime.now()).isoformat()
        for area in price_areas:
            # Generate random energy values
            central_power = random.uniform(0, 1000)
            local_power = random.uniform(0, 1000)
            gross_consumption = random.uniform(0, 1000)
            exchange_no = random.uniform(-100, 100)
            exchange_se = random.uniform(-100, 100)
            exchange_de = random.uniform(-100, 100)
            solar_self_con = random.uniform(0, 100)
            grid_loss_trans = random.uniform(0, 100)

            params = (
                hour_timestamp, area, central_power, local_power,
                gross_consumption, exchange_no, exchange_se, exchange_de,
                solar_self_con, grid_loss_trans
            )
            db_handler.execute_query(query, params)

    db_handler.commit()
    logger.info("Mock energy data generation completed.")


def generate_production_data(db_handler, hours, price_areas, types):
    placeholder = db_handler.get_placeholder_symbol()

    insert_query = f"""
    INSERT INTO production_data (
                timestamp, price_area, wind_total_mwh, offshore_wind_total_mwh, offshore_wind_lt100mw_mwh,
                offshore_wind_ge100mw_mwh, onshore_wind_total_mwh,onshore_wind_ge50_kw_mwh, solar_total_mwh,
                 solar_total_no_self_consumption_mwh, solar_power_self_consumption_mwh, solar_power_ge40_kw_mwh,
                  solar_power_ge10_lt40_kw_mwh, solar_power_lt10_kw_mwh, commercial_power_mwh,
                commercial_power_self_consumption_mwh, central_power_mwh, hydro_power_mwh,local_power_mwh
                ) VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder},
                {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder},
                {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}
                )"""

    for hour_offset in range(hours):
        ts = get_random_date_between(datetime(1970, 1, 1, 0, 0, 0, 000), datetime.now()).isoformat()
        for price_area in price_areas:
            # Initialize all production values to zero.
            wind_total = offshore_total = offshore_lt = offshore_ge = onshore = 0
            solar_total = solar_no_self = solar_self = solar_ge40 = solar_ge10 = 0
            commercial = commercial_self = central = hydro = local = 0

            # Simulate wind production values.
            if "wind" in types:
                wind_total = random.uniform(50, 150)
                offshore_total = random.uniform(20, 100)
                offshore_lt = random.uniform(10, 50)
                offshore_ge = offshore_total - offshore_lt
                onshore = random.uniform(20, 60)
                onshore_ge50 = random.uniform(5, 20)

            # Simulate solar production values.
            if "solar" in types:
                solar_total = random.uniform(30, 90)
                solar_no_self = solar_total * random.uniform(0.5, 0.9)
                solar_self = solar_total - solar_no_self
                solar_ge40 = random.uniform(5, 20)
                solar_ge10 = random.uniform(10, 30)
                solar_lt10 = random.uniform(8, 25)

            # Simulate hydro production.
            if "hydro" in types:
                hydro = random.uniform(10, 40)

            # Simulate commercial power production.
            if "commercialPower" in types:
                commercial = random.uniform(30, 100)
                commercial_self = random.uniform(5, 20)

            # Simulate central power production.
            if "centralPower" in types:
                central = random.uniform(40, 120)

            # Simulate local production, if applicable.
            if "local" in types:
                local = random.uniform(10, 50)

            params = (ts, price_area, wind_total, offshore_total, offshore_lt,
                      offshore_ge, onshore, onshore_ge50, solar_total, solar_no_self,
                      solar_self, solar_ge40, solar_ge10, solar_lt10, commercial,
                      commercial_self, central, hydro, local)
            db_handler.execute_query(insert_query, params)

    db_handler.commit()
    logger.info("Mock production data generation completed.")


def generate_consumption_data(db_handler, hours, price_areas):
    """
    Generate mock consumption data for the specified number of hours and price areas.

    Args:
        db_handler: instance of the database handler.
        hours: number of hours for which to generate data.
        price_areas: list of price areas (e.g., ['DK1', 'DK2']).
    """
    placeholder = db_handler.get_placeholder_symbol()

    # Create a starting timestamp (current time minus 'hours' hours)
    start_time = datetime.now() - timedelta(hours=hours)

    # Loop over each hour and each price area to generate a record.
    for i in range(hours):
        timestamp = get_random_date_between(datetime(1970, 1, 1, 0, 0, 0, 000), datetime.now()).isoformat()
        for area in price_areas:
            consumption_total = random.uniform(100, 500)  # Example ranges
            consumption_private = random.uniform(50, consumption_total)
            consumption_public = random.uniform(10, consumption_total - consumption_private)
            consumption_commertial = random.uniform(5, consumption_total - consumption_private - consumption_public)
            grid_loss_transmission = random.uniform(0, 50)
            grid_loss_interconnectors = random.uniform(0, 20)
            grid_loss_distribution = random.uniform(0, 30)
            power_to_heat = random.uniform(0, 40)

            query = f"""
            INSERT INTO consumption_data (
                timestamp,
                price_area,
                consumption_total_mwh,
                consumption_private_mwh,
                consumption_public_total_mwh,
                consumption_commertial_total_mwh,
                grid_loss_transmission_mwh,
                grid_loss_interconnectors_mwh,
                grid_loss_distribution_mwh,
                power_to_heat_mwh
            ) VALUES (
                {placeholder}, {placeholder}, {placeholder}, {placeholder},
                {placeholder}, {placeholder}, {placeholder}, {placeholder},
                {placeholder}, {placeholder}
            )
            """

            params = (
                timestamp, area, consumption_total, consumption_private,
                consumption_public, consumption_commertial, grid_loss_transmission,
                grid_loss_interconnectors, grid_loss_distribution, power_to_heat
            )
            db_handler.execute_query(query, params)

    db_handler.commit()
    logger.info("Mock consumption data generation completed.")


def generate_exchange_data(db_handler, hours, countries):
    base_time = datetime.now()
    # Define possible price areas for exchange records
    price_areas = ["DK1", "DK2"]
    for hour in range(hours):
        # Generate a timestamp for this hour
        ts = get_random_date_between(datetime(1970, 1, 1, 0, 0, 0, 000), datetime.now()).isoformat()
        for country in countries:
            # Randomly select a price area
            price_area = random.choice(price_areas)
            # Generate random exchange values
            import_mwh = round(random.uniform(50, 150), 2)
            export_mwh = round(random.uniform(50, 150), 2)
            net_exchange = round(export_mwh - import_mwh, 2)

            query = """
                    INSERT INTO exchange_data (timestamp,
                                               price_area,
                                               exchange_country,
                                               import_mwh,
                                               export_mwh,
                                               net_exchange_mwh)
                    VALUES (%s, %s, %s, %s, %s, %s) \
                    """
            params = (ts, price_area, country, import_mwh, export_mwh, net_exchange)
            db_handler.execute_query(query, params)

    db_handler.commit()


def generate_mock_database(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT,
        hours=1000,
        db_type=DB_DRIVER,
        data_types=None  # New parameter for selecting specific mock data
):
    """
    Generate a mock database with sample data.

    If data_types is provided, only the specified mock data will be created.
    Available types: TableEnum.ENERGY, TableEnum.PRODUCTION, TableEnum.CONSUMPTION,
    TableEnum.EXCHANGE, TableEnum.CARBON_INTENSITY, TableEnum.AGGREGATED_PRODUCTION,
    TableEnum.COMPARISON_ANALYSIS, TableEnum.CONSUMPTION_FORECAST, TableEnum.CACHE_WEIGHTS
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
            "energy_data", "production_data", "consumption_data", "exchange_data",
            "carbon_intensity", "aggregated_production",
            "comparison_analysis", "consumption_forecast", "cache_metrics"
        ]

    price_areas = ["DK1", "DK2"]
    countries = ["germany", "greatbritain", "netherlands", "norway", "sweden"]
    types = ["wind", "solar", "hydro", "commercialPower", "centralPower", "local"]
    aggregationTypes = ["hourly", "daily", "weekly", "monthly", "yearly"]
    comparisonTypes = ["absolute", "percentage", "yearOverYear", "custom"]
    horizon = 24

    if db_type not in ['mysql', 'postgres', 'sqlite']:
        logger.error(f"Unsupported database type: {db_type}")
        return False

    # If no selection provided, generate all mock data.
    if data_types is None:
        data_types = [
            "energy_data", "production_data", "consumption_data", "exchange_data",
            "carbon_intensity", "aggregated_production",
            "comparison_analysis", "consumption_forecast", "cache_metrics"
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
        if "energy_data" in data_types:
            logger.info("Generating mock energy data")
            generate_energy_data(db_handler, hours, price_areas)
        if "production_data" in data_types:
            logger.info("Generating mock production data")
            generate_production_data(db_handler, hours, price_areas, types)
        if "consumption_data" in data_types:
            logger.info("Generating mock consumption data")
            generate_consumption_data(db_handler, hours, price_areas)
        if "exchange_data" in data_types:
            logger.info("Generating mock exchange data")
            generate_exchange_data(db_handler, hours, countries)
        if "cache_metrics" in data_types:
            logger.info("Generating derived data cache weights")
            simulate_visits(
                n=10000,  # or any desired number of visits
                db_handler=db_handler,
                run_duration=10  # or adjust as needed
            )

        db_handler.commit()
        db_handler.close()

        logger.info("Mock database generation completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error generating mock database: {e}")
        return False
