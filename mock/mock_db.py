# mock/mock_db.py
import json
import logging
import os
import random
from datetime import datetime, timedelta

from database.create_tables import create_tables
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
        hour_timestamp = get_random_date_between(datetime(1970,1,1,0,0,0,000), datetime.now()).isoformat()
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
        ts = get_random_date_between(datetime(1970,1,1,0,0,0,000), datetime.now()).isoformat()
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
        timestamp = get_random_date_between(datetime(1970,1,1,0,0,0,000), datetime.now()).isoformat()
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
        ts = get_random_date_between(datetime(1970,1,1,0,0,0,000), datetime.now()).isoformat()
        for country in countries:
            # Randomly select a price area
            price_area = random.choice(price_areas)
            # Generate random exchange values
            import_mwh = round(random.uniform(50, 150), 2)
            export_mwh = round(random.uniform(50, 150), 2)
            net_exchange = round(export_mwh - import_mwh, 2)

            query = """
            INSERT INTO exchange_data (
                timestamp,
                price_area,
                exchange_country,
                import_mwh,
                export_mwh,
                net_exchange_mwh
            ) VALUES (%s, %s, %s, %s, %s, %s)
            """
            params = (ts, price_area, country, import_mwh, export_mwh, net_exchange)
            db_handler.execute_query(query, params)

    db_handler.commit()


def generate_carbon_intensity_data(db_handler, hours, price_areas):
    """
    Generate sample carbon intensity data.

    Args:
        db_handler: Database handler instance.
        hours: How many hours in the past to generate data for.
        price_areas: List of price areas (e.g. ["DK1", "DK2"]).
    """
    now = datetime.now()

    for h in range(hours):
        timestamp = get_random_date_between(datetime(1970,1,1,0,0,0,000), datetime.now()).isoformat()
        for area in price_areas:
            # Generate a random carbon intensity value.
            carbon_intensity = round(random.uniform(50, 150), 2)

            # Generate an energy mix as percentages.
            energy_mix = {
                "Wind": round(random.uniform(20, 80), 2),
                "Solar": round(random.uniform(5, 25), 2),
                "Hydro": round(random.uniform(0, 20), 2),
                "Fossil": round(random.uniform(0, 50), 2),
                "Nuclear": round(random.uniform(0, 30), 2),
                "Biomass": round(random.uniform(0, 20), 2),
                "Other": round(random.uniform(0, 10), 2)
            }
            # Encode the energy mix to a JSON string.
            energy_mix_json = json.dumps(energy_mix)

            # Insert into carbon_intensity table.
            query = f"""
            INSERT INTO carbon_intensity (
                timestamp, price_area, carbon_intensity_gco2per_kwh, energy_mix
            ) VALUES ({db_handler.get_placeholder_symbol()}, {db_handler.get_placeholder_symbol()},
                      {db_handler.get_placeholder_symbol()}, {db_handler.get_placeholder_symbol()})
            """

            params = (timestamp, area, carbon_intensity, energy_mix_json)
            db_handler.execute_query(query, params)

    db_handler.commit()


def generate_aggregated_production(db_handler, hours, aggregationTypes, price_areas):
    """
    Simulate aggregated production data and insert rows into the aggregated_production table.

    Args:
        db_handler: Database handler instance.
        hours: Number of simulation iterations (hours).
        aggregationTypes: List of aggregation types (e.g. "hourly", "daily", "weekly", "monthly", "yearly").
        price_areas: List of price areas (e.g. ["DK1", "DK2"]).
    """
    placeholder = db_handler.get_placeholder_symbol()
    base_time = datetime.now()

    # Loop over each simulation hour
    for i in range(hours):
        # Calculate the random time from 1970-01-01-00 to now in  YYYY-MM-DD HH:MM:SS.mmmmmm

        current_time = get_random_date_between(datetime(1970,1,1,0,0,0,000), datetime.now())
        # Loop over each price area
        for area in price_areas:
            # For each defined aggregation type
            for agg_type in aggregationTypes:

                # Determine period end based on aggregation type
                if agg_type == "hourly":
                    period_end = current_time + timedelta(hours=1)
                elif agg_type == "daily":
                    period_end = current_time + timedelta(days=1)
                elif agg_type == "weekly":
                    period_end = current_time + timedelta(weeks=1)
                elif agg_type == "monthly":
                    period_end = current_time + timedelta(days=30)
                elif agg_type == "yearly":
                    period_end = current_time + timedelta(days=365)
                else:
                    period_end = current_time + timedelta(hours=1)

                # Simulate production values
                wind = random.uniform(0, 1000)
                solar = random.uniform(0, 500)
                hydro = random.uniform(0, 200)
                commercial = random.uniform(0, 800)
                central = random.uniform(0, 1200)
                total = wind + solar + hydro + commercial + central

                query = f"""
                INSERT INTO aggregated_production (
                    period_start, period_end, price_area, aggregation_type, total_production_mwh,
                    wind_production_mwh, solar_production_mwh, hydro_production_mwh, commercial_production_mwh,
                    central_production_mwh
                ) VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder},
                    {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
                """
                params = (
                    current_time.isoformat(),
                    period_end,
                    area,
                    agg_type,
                    total,
                    wind,
                    solar,
                    hydro,
                    commercial,
                    central
                )
                db_handler.execute_query(query, params)

    db_handler.commit()


def generate_comparison_analysis(db_handler, hours, comparisonTypes, price_areas):
    """
    Generate simulated comparison analysis data and insert into the database.

    For each price area and each comparison type, this function:
     - Defines Period1 as a recent period ending now minus the given hours.
     - Defines Period2 immediately preceding Period1 (with a 1-hour gap).
     - Simulates main and comparison data for TotalConsumption_MWh, WindProduction_MWh, and SolarProduction_MWh.
     - Computes both the absolute difference and percentage change.
     - Inserts the computed values into the 'comparison_analysis' table.
    """
    now = get_random_date_between(datetime(1970,1,1,0,0,0,000), datetime.now())
    period_duration = timedelta(days=1)  # Duration for each period
    placeholder = db_handler.get_placeholder_symbol()

    for price_area in price_areas:
        for comp_type in comparisonTypes:
            # Define periods; Period1 is the recent period and Period2 is the preceding period
            period1_end = now - timedelta(hours=hours)
            period1_start = period1_end - period_duration
            period2_end = period1_start - timedelta(hours=1)  # 1-hour gap between periods
            period2_start = period2_end - period_duration

            # Simulate sample data for Period1 (main period)
            main_data = {
                "TotalConsumption_MWh": random.uniform(500, 1500),
                "WindProduction_MWh": random.uniform(50, 300),
                "SolarProduction_MWh": random.uniform(10, 100)
            }
            # Simulate sample data for Period2 (comparison period)
            compare_data = {
                "TotalConsumption_MWh": main_data["TotalConsumption_MWh"] * random.uniform(0.9, 1.1),
                "WindProduction_MWh": main_data["WindProduction_MWh"] * random.uniform(0.9, 1.1),
                "SolarProduction_MWh": main_data["SolarProduction_MWh"] * random.uniform(0.9, 1.1)
            }

            # Calculate the differences and percentage changes
            difference = {key: main_data[key] - compare_data.get(key, 0) for key in main_data}
            percentage_change = {}
            for key in main_data:
                comp_value = compare_data.get(key, 0)
                if comp_value != 0:
                    percentage_change[key] = round((difference[key] / comp_value) * 100, 2)
                else:
                    percentage_change[key] = 0

            # Prepare the INSERT query with parameter placeholders
            query = f"""
            INSERT INTO comparison_analysis (
                generated_on,
                price_area,
                period1_start,
                period1_end,
                period2_start,
                period2_end,
                comparison_type,
                difference,
                percentage_change
            ) VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
            """
            params = (
                now,
                price_area,
                period1_start,
                period1_end,
                period2_start,
                period2_end,
                comp_type,
                json.dumps(difference),
                json.dumps(percentage_change)
            )

            db_handler.execute_query(query, params)

    db_handler.commit()


def generate_consumption_forecast(db_handler, hours, horizon, price_areas):
    """
    Generate consumption forecast data and insert into the database.

    Args:
        db_handler: Database handler instance.
        hours: Total number of hours (not directly used here, but could control historical data range).
        horizon: Forecast horizon (number of hours to forecast).
        price_areas: List of price areas to forecast for.
    """
    request_date = datetime.now().isoformat()
    forecast_entries = []

    for price_area in price_areas:
        forecast_data = []
        forecast_start = get_random_date_between(datetime(1970,1,1,0,0,0,000), datetime.now())

        for hour in range(hours):
            for i in range(horizon):
                forecast_time = forecast_start + timedelta(hours=i + 1)
                # Generate dummy consumption forecast and confidence intervals.
                base_consumption = random.uniform(2000, 3000)
                fluctuation = random.uniform(50, 150)
                forecast_entry = {
                    "Timestamp": forecast_time.isoformat(),
                    "ConsumptionForecast_MWh": round(base_consumption, 2),
                    "ConfidenceLow_MWh": round(base_consumption - fluctuation, 2),
                    "ConfidenceHigh_MWh": round(base_consumption + fluctuation, 2)
                }
                forecast_data.append(forecast_entry)

        # Convert the forecast data list to a JSON string.
        forecast_json = json.dumps(forecast_data)

        # Insert into consumption_forecast table.
        placeholder = db_handler.get_placeholder_symbol()
        query = f"""
        INSERT INTO consumption_forecast (request_date, price_area, forecast_horizon, forecast_data)
        VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder})
        """
        db_handler.execute_query(query, (request_date, price_area, horizon, forecast_json))
        forecast_entries.append({
            "RequestDate": request_date,
            "PriceArea": price_area,
            "ForecastHorizon": horizon,
            "ForecastData": forecast_data
        })

    db_handler.commit()
    return forecast_entries

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
        if "carbon_intensity" in data_types:
            logger.info("Generating mock carbon intensity data")
            generate_carbon_intensity_data(db_handler, hours, price_areas)
        if "aggregated_production" in data_types:
            logger.info("Generating aggregated production data")
            generate_aggregated_production(db_handler, hours, aggregationTypes, price_areas)
        if "comparison_analysis" in data_types:
            logger.info("Generating comparison analysis data")
            generate_comparison_analysis(db_handler, hours, comparisonTypes, price_areas)
        if "consumption_forecast" in data_types:
            logger.info("Generating consumption forecast data")
            generate_consumption_forecast(db_handler, hours, horizon, price_areas)
        if "cache_metrics" in data_types:
            logger.info("Generating derived data cache weights")
            simulate_cache_metrics(db_handler, update_interval=5, run_duration=10, stop_event=None)

        db_handler.commit()
        db_handler.close()

        logger.info("Mock database generation completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error generating mock database: {e}")
        return False
