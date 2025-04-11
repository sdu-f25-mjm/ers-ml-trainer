# caching_gpu/mock/mock_db.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import warnings
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import mysql.connector

    logger.info("MySQL connector imported successfully")
except ImportError:
    logger.error("MySQL connector not found. Installing...")
    import subprocess

    try:
        subprocess.check_call(["pip", "install", "mysql-connector-python==8.0.33"])
        import mysql.connector

        logger.info("MySQL connector installed successfully")
    except Exception as e:
        logger.error(f"Failed to install MySQL connector: {e}")
        raise ImportError("Please install mysql-connector-python: pip install mysql-connector-python==8.0.33")

try:
    from faker import Faker
except ImportError:
    logger.error("Faker not found. Installing...")
    import subprocess

    subprocess.check_call(["pip", "install", "Faker"])
    from faker import Faker

try:
    from sqlalchemy import create_engine
except ImportError:
    logger.error("SQLAlchemy not found. Installing...")
    import subprocess

    subprocess.check_call(["pip", "install", "sqlalchemy"])
    from sqlalchemy import create_engine

# Ignore SQLAlchemy deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


def get_engine_url(config):
    """Create SQLAlchemy connection string with explicit connector"""
    connector = "mysql+mysqlconnector"
    return f"{connector}://{config['user']}:{config['password']}@{config['host']}/{config['database']}"


def generate_mock_database(host='mariadb', user='cacheuser', password='cachepass', database='cache_db',
                           hours=168, price_areas=None):
    """Generate a mock energy database based on OpenAPI schema"""
    if price_areas is None:
        price_areas = ["DK1", "DK2"]

    logger.info(f"Connecting to database: {host}, user: {user}")

    # Create connection to MariaDB
    try:
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password
        )
        logger.info("Connection established to server")
    except Exception as e:
        logger.error(f"Connection error: {e}")
        return False

    cursor = conn.cursor()

    # Create database if not exists
    try:
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database}")
        cursor.execute(f"USE {database}")
        logger.info(f"Database '{database}' selected")
    except Exception as e:
        logger.error(f"Database creation/selection error: {e}")
        conn.close()
        return False

    # Create tables based on OpenAPI schema
    try:
        # Main energy data table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS energy_data (
            id INT PRIMARY KEY AUTO_INCREMENT,
            HourUTC DATETIME,
            PriceArea VARCHAR(10),
            CentralPower_MWh DECIMAL(10, 2),
            LocalPower_MWh DECIMAL(10, 2),
            GrossConsumption_MWh DECIMAL(10, 2),
            ExchangeNO_MWh DECIMAL(10, 2),
            ExchangeSE_MWh DECIMAL(10, 2),
            ExchangeDE_MWh DECIMAL(10, 2),
            ExchangeNL_MWh DECIMAL(10, 2),
            ExchangeGB_MWh DECIMAL(10, 2),
            SolarPowerSelfConMWh DECIMAL(10, 2),
            GridLossTransmissionMWh DECIMAL(10, 2),
            INDEX(HourUTC),
            INDEX(PriceArea)
        )
        """)

        # Production details table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS production_data (
            id INT PRIMARY KEY AUTO_INCREMENT,
            HourUTC DATETIME,
            PriceArea VARCHAR(10),
            CentralPower_MWh DECIMAL(10, 2),
            LocalPower_MWh DECIMAL(10, 2),
            CommercialPower_MWh DECIMAL(10, 2),
            CommercialPowerSelfConsumption_MWh DECIMAL(10, 2),
            HydroPower_MWh DECIMAL(10, 2),
            WindTotal_MWh DECIMAL(10, 2),
            OffshoreWindTotal_MWh DECIMAL(10, 2),
            OffshoreWindLt100MW_MWh DECIMAL(10, 2),
            OffshoreWindGe100MW_MWh DECIMAL(10, 2),
            OnshoreWindTotal_MWh DECIMAL(10, 2),
            OnshoreWindGe50kW_MWh DECIMAL(10, 2),
            SolarTotal_MWh DECIMAL(10, 2),
            SolarTotalNoSelfConsumption_MWh DECIMAL(10, 2),
            SolarPowerSelfConsumption_MWh DECIMAL(10, 2),
            SolarPowerLt10kW_MWh DECIMAL(10, 2),
            SolarPowerGe10Lt40kW_MWh DECIMAL(10, 2),
            SolarPowerGe40kW_MWh DECIMAL(10, 2),
            INDEX(HourUTC),
            INDEX(PriceArea)
        )
        """)

        # Consumption details table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS consumption_data (
            id INT PRIMARY KEY AUTO_INCREMENT,
            HourUTC DATETIME,
            PriceArea VARCHAR(10),
            ConsumptionTotal_MWh DECIMAL(10, 2),
            ConsumptionPrivate_MWh DECIMAL(10, 2),
            ConsumptionPublicTotal_MWh DECIMAL(10, 2),
            ConsumptionCommertialTotal_MWh DECIMAL(10, 2),
            GridLossTransmission_MWh DECIMAL(10, 2),
            GridLossDistribution_MWh DECIMAL(10, 2),
            GridLossInterconnectors_MWh DECIMAL(10, 2),
            PowerToHeatMWh DECIMAL(10, 2),
            INDEX(HourUTC),
            INDEX(PriceArea)
        )
        """)

        # Exchange data table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS exchange_data (
            id INT PRIMARY KEY AUTO_INCREMENT,
            HourUTC DATETIME,
            Country VARCHAR(20),
            PriceArea VARCHAR(10),
            Export_MWh DECIMAL(10, 2),
            Import_MWh DECIMAL(10, 2),
            NetExchange_MWh DECIMAL(10, 2),
            INDEX(HourUTC),
            INDEX(Country),
            INDEX(PriceArea)
        )
        """)

        # Carbon intensity table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS carbon_intensity (
            id INT PRIMARY KEY AUTO_INCREMENT,
            Timestamp DATETIME,
            PriceArea VARCHAR(10),
            CarbonIntensity_gCO2perKWh DECIMAL(10, 2),
            Wind_Percent DECIMAL(5, 2),
            Solar_Percent DECIMAL(5, 2),
            Hydro_Percent DECIMAL(5, 2),
            Fossil_Percent DECIMAL(5, 2),
            Nuclear_Percent DECIMAL(5, 2),
            Biomass_Percent DECIMAL(5, 2),
            Other_Percent DECIMAL(5, 2),
            INDEX(Timestamp),
            INDEX(PriceArea)
        )
        """)

        conn.commit()
        logger.info("Energy data tables created successfully")

    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        conn.close()
        return False

    # Clear existing data
    try:
        cursor.execute("SET FOREIGN_KEY_CHECKS=0")
        cursor.execute("TRUNCATE TABLE energy_data")
        cursor.execute("TRUNCATE TABLE production_data")
        cursor.execute("TRUNCATE TABLE consumption_data")
        cursor.execute("TRUNCATE TABLE exchange_data")
        cursor.execute("TRUNCATE TABLE carbon_intensity")
        cursor.execute("SET FOREIGN_KEY_CHECKS=1")
        conn.commit()
        logger.info("All tables cleared successfully")
    except Exception as e:
        logger.error(f"Error clearing tables: {e}")
        conn.commit()

    # Generate mock data
    fake = Faker()
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)

    countries = ["norway", "sweden", "germany", "netherlands", "greatbritain"]

    try:
        # Generate hourly energy data
        for hour_offset in range(hours):
            current_time = start_time + timedelta(hours=hour_offset)

            for area in price_areas:
                # Base values with small random variations to create patterns
                hour_of_day = current_time.hour
                day_of_week = current_time.weekday()

                # Consumption has daily pattern (higher during day, lower at night)
                consumption_factor = 0.6 + 0.4 * np.sin(np.pi * hour_of_day / 12)
                if 8 <= hour_of_day <= 18:  # Working hours
                    consumption_factor *= 1.3

                # Weekend vs weekday pattern
                if day_of_week >= 5:  # Weekend
                    consumption_factor *= 0.8

                # Random seasonal component (simplified)
                month = current_time.month
                seasonal_factor = 0.8 + 0.4 * np.sin(np.pi * (month - 3) / 6)  # Peak in winter

                # Base values
                central_power = max(0, np.random.normal(500, 100) * seasonal_factor)
                local_power = max(0, np.random.normal(200, 50) * seasonal_factor)
                gross_consumption = max(0, np.random.normal(1000, 200) * consumption_factor * seasonal_factor)

                # Wind and solar have weather patterns
                wind_factor = max(0, np.random.normal(0.8, 0.3))  # Wind is variable
                solar_factor = max(0,
                                   np.sin(np.pi * hour_of_day / 12) * np.random.normal(0.7, 0.2))  # Solar peaks mid-day

                wind_power = max(0, np.random.normal(300, 100) * wind_factor * seasonal_factor)
                solar_power = max(0, np.random.normal(150, 50) * solar_factor)

                # Energy data
                cursor.execute("""
                INSERT INTO energy_data (
                    HourUTC, PriceArea, CentralPower_MWh, LocalPower_MWh, 
                    GrossConsumption_MWh, ExchangeNO_MWh, ExchangeSE_MWh, 
                    ExchangeDE_MWh, ExchangeNL_MWh, ExchangeGB_MWh,
                    SolarPowerSelfConMWh, GridLossTransmissionMWh
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    current_time, area, central_power, local_power,
                    gross_consumption,
                    np.random.normal(100, 50), np.random.normal(120, 60),
                    np.random.normal(200, 80), np.random.normal(150, 70), np.random.normal(180, 90),
                    solar_power * 0.2, gross_consumption * np.random.uniform(0.02, 0.05)
                ))

                # Production data
                offshore_wind = wind_power * 0.6
                onshore_wind = wind_power * 0.4

                cursor.execute("""
                INSERT INTO production_data (
                    HourUTC, PriceArea, CentralPower_MWh, LocalPower_MWh,
                    CommercialPower_MWh, CommercialPowerSelfConsumption_MWh, HydroPower_MWh,
                    WindTotal_MWh, OffshoreWindTotal_MWh, OffshoreWindLt100MW_MWh,
                    OffshoreWindGe100MW_MWh, OnshoreWindTotal_MWh, OnshoreWindGe50kW_MWh,
                    SolarTotal_MWh, SolarTotalNoSelfConsumption_MWh, SolarPowerSelfConsumption_MWh,
                    SolarPowerLt10kW_MWh, SolarPowerGe10Lt40kW_MWh, SolarPowerGe40kW_MWh
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """, (
                    current_time, area, central_power, local_power,
                    central_power * 0.7, central_power * 0.1, np.random.normal(50, 20),
                    wind_power, offshore_wind, offshore_wind * 0.3,
                    offshore_wind * 0.7, onshore_wind, onshore_wind * 0.9,
                    solar_power, solar_power * 0.8, solar_power * 0.2,
                    solar_power * 0.3, solar_power * 0.3, solar_power * 0.4
                ))

                # Consumption data
                cursor.execute("""
                INSERT INTO consumption_data (
                    HourUTC, PriceArea, ConsumptionTotal_MWh, ConsumptionPrivate_MWh,
                    ConsumptionPublicTotal_MWh, ConsumptionCommertialTotal_MWh,
                    GridLossTransmission_MWh, GridLossDistribution_MWh,
                    GridLossInterconnectors_MWh, PowerToHeatMWh
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """, (
                    current_time, area, gross_consumption, gross_consumption * 0.4,
                    gross_consumption * 0.2, gross_consumption * 0.4,
                    gross_consumption * 0.03, gross_consumption * 0.02,
                    gross_consumption * 0.01, gross_consumption * np.random.uniform(0.05, 0.15)
                ))

                # Carbon intensity
                wind_percent = (wind_power / (central_power + local_power + wind_power + solar_power)) * 100 if (
                                                                                                                            central_power + local_power + wind_power + solar_power) > 0 else 0
                solar_percent = (solar_power / (central_power + local_power + wind_power + solar_power)) * 100 if (
                                                                                                                              central_power + local_power + wind_power + solar_power) > 0 else 0
                hydro_percent = np.random.uniform(1, 3)
                fossil_percent = max(0, 100 - wind_percent - solar_percent - hydro_percent - np.random.uniform(5, 15))

                # Calculate carbon intensity based on energy mix
                carbon_intensity = (fossil_percent * 820 + (100 - fossil_percent) * 20) / 100

                cursor.execute("""
                INSERT INTO carbon_intensity (
                    Timestamp, PriceArea, CarbonIntensity_gCO2perKWh,
                    Wind_Percent, Solar_Percent, Hydro_Percent,
                    Fossil_Percent, Nuclear_Percent, Biomass_Percent, Other_Percent
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """, (
                    current_time, area, carbon_intensity,
                    wind_percent, solar_percent, hydro_percent,
                    fossil_percent, 0, np.random.uniform(5, 15), np.random.uniform(0, 2)
                ))

                # Exchange data
                for country in countries:
                    export_val = np.random.normal(100, 50)
                    import_val = np.random.normal(80, 40)

                    cursor.execute("""
                    INSERT INTO exchange_data (
                        HourUTC, Country, PriceArea, Export_MWh, Import_MWh, NetExchange_MWh
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s
                    )
                    """, (
                        current_time, country, area, export_val, import_val, export_val - import_val
                    ))

            # Commit in batches
            if hour_offset % 24 == 0:
                conn.commit()
                logger.info(f"Generated data for {hour_offset} hours")

        conn.commit()
        logger.info(f"Generated {hours} hours of energy data for {', '.join(price_areas)}")
    except Exception as e:
        logger.error(f"Error generating mock data: {e}")
        conn.close()
        return False

    cursor.close()
    conn.close()
    logger.info("Energy database generation completed successfully")
    return True


if __name__ == "__main__":
    # Use environment variables to make it flexible for Docker
    host = os.environ.get('DB_HOST', 'localhost')
    user = os.environ.get('DB_USER', 'cacheuser')
    password = os.environ.get('DB_PASSWORD', 'cachepass')
    database = os.environ.get('DB_NAME', 'cache_db')

    # Generate database with Docker-compatible defaults
    generate_mock_database(
        host=host,
        user=user,
        password=password,
        database=database,
        hours=10000, # hours of data to generate
        price_areas=["DK1", "DK2"]
    )

    logger.info("Mock energy database setup completed!")