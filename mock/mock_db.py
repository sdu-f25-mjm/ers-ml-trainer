# mock/mock_db.py
import logging
import random
from datetime import datetime, timedelta
import numpy as np
import mysql.connector
import json

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_mock_database(host='ers-mariadb', user='cacheuser', password='cachepass', database='cache_db',
                           hours=1000, price_areas=None):
    """Generate a mock energy database with all required tables for the API endpoints"""
    if price_areas is None:
        price_areas = ["DK1", "DK2"]

    logger.info(f"Generating mock database with {hours} hours of data for {', '.join(price_areas)}")

    try:
        # Connect to the database
        conn = mysql.connector.connect(
            host=host, user=user, password=password
        )
        cursor = conn.cursor()

        # Create database if it doesn't exist
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database}")
        cursor.execute(f"USE {database}")

        # Create the tables
        create_tables(cursor)

        # Generate data
        generate_energy_data(cursor, hours, price_areas)
        generate_production_data(cursor, hours, price_areas)
        generate_consumption_data(cursor, hours, price_areas)
        generate_exchange_data(cursor, hours, price_areas)
        generate_carbon_intensity_data(cursor, hours, price_areas)

        # Generate derived data tables
        generate_aggregated_production_data(cursor, hours, price_areas)
        generate_comparison_analysis_data(cursor, hours, price_areas)
        generate_consumption_forecast_data(cursor, price_areas)

        # Generate cache weights
        generate_cache_weights(host, user, password, database)

        conn.commit()
        conn.close()

        logger.info("Mock database generation completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error generating mock database: {e}")
        return False


def create_tables(cursor):
    """Create all necessary database tables"""
    # Base energy data table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS energy_data (
        id INT PRIMARY KEY AUTO_INCREMENT,
        HourUTC DATETIME,
        PriceArea VARCHAR(10),
        CentralPower_MWh FLOAT,
        LocalPower_MWh FLOAT,
        GrossConsumption_MWh FLOAT,
        ExchangeNO_MWh FLOAT,
        ExchangeSE_MWh FLOAT,
        ExchangeDE_MWh FLOAT,
        SolarPowerSelfConMWh FLOAT,
        GridLossTransmissionMWh FLOAT,
        INDEX(HourUTC),
        INDEX(PriceArea)
    )
    """)

    # Production data table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS production_data (
        id INT PRIMARY KEY AUTO_INCREMENT,
        HourUTC DATETIME,
        PriceArea VARCHAR(10),
        CentralPower_MWh FLOAT,
        LocalPower_MWh FLOAT,
        CommercialPower_MWh FLOAT,
        CommercialPowerSelfConsumption_MWh FLOAT,
        HydroPower_MWh FLOAT,
        WindTotal_MWh FLOAT,
        OffshoreWindTotal_MWh FLOAT,
        OffshoreWindLt100MW_MWh FLOAT,
        OffshoreWindGe100MW_MWh FLOAT,
        OnshoreWindTotal_MWh FLOAT,
        OnshoreWindGe50kW_MWh FLOAT,
        SolarTotal_MWh FLOAT,
        SolarTotalNoSelfConsumption_MWh FLOAT,
        SolarPowerSelfConsumption_MWh FLOAT,
        SolarPowerLt10kW_MWh FLOAT,
        SolarPowerGe10Lt40kW_MWh FLOAT,
        SolarPowerGe40kW_MWh FLOAT,
        INDEX(HourUTC),
        INDEX(PriceArea)
    )
    """)

    # Consumption data table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS consumption_data (
        id INT PRIMARY KEY AUTO_INCREMENT,
        HourUTC DATETIME,
        PriceArea VARCHAR(10),
        ConsumptionTotal_MWh FLOAT,
        ConsumptionPrivate_MWh FLOAT,
        ConsumptionPublicTotal_MWh FLOAT,
        ConsumptionCommertialTotal_MWh FLOAT,
        GridLossTransmission_MWh FLOAT,
        GridLossDistribution_MWh FLOAT,
        GridLossInterconnectors_MWh FLOAT,
        PowerToHeatMWh FLOAT,
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
        Export_MWh FLOAT,
        Import_MWh FLOAT,
        NetExchange_MWh FLOAT,
        INDEX(HourUTC),
        INDEX(Country),
        INDEX(PriceArea)
    )
    """)

    # Carbon intensity data table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS carbon_intensity (
        id INT PRIMARY KEY AUTO_INCREMENT,
        HourUTC DATETIME,
        PriceArea VARCHAR(10),
        CarbonIntensity_gCO2perKWh FLOAT,
        Wind_Pct FLOAT,
        Solar_Pct FLOAT,
        Hydro_Pct FLOAT,
        Fossil_Pct FLOAT,
        Nuclear_Pct FLOAT,
        Biomass_Pct FLOAT,
        Other_Pct FLOAT,
        INDEX(HourUTC),
        INDEX(PriceArea)
    )
    """)

    # Aggregated production data table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS aggregated_production (
        id INT PRIMARY KEY AUTO_INCREMENT,
        PeriodStart DATETIME,
        PeriodEnd DATETIME,
        PriceArea VARCHAR(10),
        AggregationType VARCHAR(20),
        TotalProduction_MWh FLOAT,
        WindProduction_MWh FLOAT,
        SolarProduction_MWh FLOAT,
        HydroProduction_MWh FLOAT,
        CommercialProduction_MWh FLOAT,
        CentralProduction_MWh FLOAT,
        INDEX(PeriodStart),
        INDEX(PeriodEnd),
        INDEX(PriceArea),
        INDEX(AggregationType)
    )
    """)

    # Comparison analysis data table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS comparison_analysis (
        id INT PRIMARY KEY AUTO_INCREMENT,
        MainPeriodFrom DATETIME,
        MainPeriodTo DATETIME,
        ComparePeriodFrom DATETIME,
        ComparePeriodTo DATETIME,
        PriceArea VARCHAR(10),
        ComparisonType VARCHAR(20),
        MainPeriodData JSON,
        ComparePeriodData JSON,
        Difference JSON,
        PercentageChange JSON,
        INDEX(MainPeriodFrom),
        INDEX(MainPeriodTo),
        INDEX(PriceArea)
    )
    """)

    # Consumption forecast data table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS consumption_forecast (
        id INT PRIMARY KEY AUTO_INCREMENT,
        RequestDate DATETIME,
        Timestamp DATETIME,
        PriceArea VARCHAR(10),
        ForecastHorizon INT,
        ConsumptionForecast_MWh FLOAT,
        ConfidenceLow_MWh FLOAT,
        ConfidenceHigh_MWh FLOAT,
        INDEX(RequestDate),
        INDEX(Timestamp),
        INDEX(PriceArea)
    )
    """)

    # Cache weights table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cache_weights (
        id INT PRIMARY KEY AUTO_INCREMENT,
        table_name VARCHAR(50),
        query_pattern VARCHAR(255),
        recency FLOAT,
        frequency FLOAT,
        time_relevance FLOAT,
        renewable_ratio FLOAT,
        priority FLOAT,
        last_updated DATETIME
    )
    """)

    # Derived data cache weights table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS derived_data_cache_weights (
        id INT PRIMARY KEY AUTO_INCREMENT,
        endpoint VARCHAR(50),
        parameter_hash VARCHAR(64),
        request_pattern VARCHAR(255),
        recency FLOAT,
        usage_frequency FLOAT,
        time_relevance FLOAT,
        production_importance FLOAT,
        volatility FLOAT,
        complexity FLOAT,
        calculated_priority FLOAT,
        last_accessed DATETIME,
        access_count INT,
        INDEX(endpoint),
        INDEX(parameter_hash)
    )
    """)


def generate_energy_data(cursor, hours, price_areas):
    """Generate basic energy data"""
    logger.info(f"Generating {hours} hours of energy data")

    base_date = datetime.now() - timedelta(hours=hours)

    for hour in range(hours):
        current_time = base_date + timedelta(hours=hour)

        for area in price_areas:
            cursor.execute("""
            INSERT INTO energy_data (
                HourUTC, PriceArea, CentralPower_MWh, LocalPower_MWh, 
                GrossConsumption_MWh, ExchangeNO_MWh, ExchangeSE_MWh, 
                ExchangeDE_MWh, SolarPowerSelfConMWh, GridLossTransmissionMWh
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                current_time,
                area,
                random.uniform(500, 1500),
                random.uniform(50, 200),
                random.uniform(1000, 3000),
                random.uniform(-300, 300),
                random.uniform(-300, 300),
                random.uniform(-500, 500),
                random.uniform(0, 100),
                random.uniform(20, 80)
            ))


def generate_production_data(cursor, hours, price_areas):
    """Generate detailed production data"""
    logger.info(f"Generating {hours} hours of production data")

    base_date = datetime.now() - timedelta(hours=hours)

    for hour in range(hours):
        current_time = base_date + timedelta(hours=hour)

        # Simulate daily and seasonal patterns
        hour_of_day = current_time.hour
        day_of_year = current_time.timetuple().tm_yday

        # Solar production follows daily sun patterns
        solar_factor = max(0, np.sin(np.pi * (hour_of_day - 5) / 14)) if 5 <= hour_of_day <= 19 else 0
        # Seasonal solar adjustment
        seasonal_solar = 0.5 + 0.5 * np.sin(np.pi * (day_of_year - 80) / 180)

        for area in price_areas:
            wind_base = random.uniform(300, 1200)

            cursor.execute("""
            INSERT INTO production_data (
                HourUTC, PriceArea, CentralPower_MWh, LocalPower_MWh, 
                CommercialPower_MWh, CommercialPowerSelfConsumption_MWh, HydroPower_MWh,
                WindTotal_MWh, OffshoreWindTotal_MWh, OffshoreWindLt100MW_MWh,
                OffshoreWindGe100MW_MWh, OnshoreWindTotal_MWh, OnshoreWindGe50kW_MWh,
                SolarTotal_MWh, SolarTotalNoSelfConsumption_MWh, SolarPowerSelfConsumption_MWh,
                SolarPowerLt10kW_MWh, SolarPowerGe10Lt40kW_MWh, SolarPowerGe40kW_MWh
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                current_time,
                area,
                random.uniform(500, 1500),
                random.uniform(50, 200),
                random.uniform(100, 400),
                random.uniform(10, 50),
                random.uniform(10, 50) if area == "DK1" else random.uniform(0, 10),
                wind_base,
                wind_base * 0.6,
                wind_base * 0.2,
                wind_base * 0.4,
                wind_base * 0.4,
                wind_base * 0.35,
                solar_factor * seasonal_solar * random.uniform(200, 700),
                solar_factor * seasonal_solar * random.uniform(150, 600),
                solar_factor * seasonal_solar * random.uniform(20, 100),
                solar_factor * seasonal_solar * random.uniform(10, 50),
                solar_factor * seasonal_solar * random.uniform(30, 150),
                solar_factor * seasonal_solar * random.uniform(100, 400)
            ))


def generate_consumption_data(cursor, hours, price_areas):
    """Generate detailed consumption data"""
    logger.info(f"Generating {hours} hours of consumption data")

    base_date = datetime.now() - timedelta(hours=hours)

    for hour in range(hours):
        current_time = base_date + timedelta(hours=hour)

        # Simulate daily and seasonal patterns
        hour_of_day = current_time.hour
        day_of_year = current_time.timetuple().tm_yday

        # Consumption follows daily patterns (higher during daytime)
        consumption_factor = 0.7 + 0.3 * np.sin(np.pi * (hour_of_day - 3) / 12)
        # Seasonal consumption adjustment (higher in winter)
        seasonal_consumption = 0.8 + 0.4 * np.cos(np.pi * (day_of_year - 15) / 180)

        total_consumption = consumption_factor * seasonal_consumption * random.uniform(1800, 2700)

        for area in price_areas:
            cursor.execute("""
            INSERT INTO consumption_data (
                HourUTC, PriceArea, ConsumptionTotal_MWh, ConsumptionPrivate_MWh, 
                ConsumptionPublicTotal_MWh, ConsumptionCommertialTotal_MWh, 
                GridLossTransmission_MWh, GridLossDistribution_MWh, GridLossInterconnectors_MWh,
                PowerToHeatMWh
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                current_time,
                area,
                total_consumption,
                total_consumption * random.uniform(0.3, 0.4),
                total_consumption * random.uniform(0.1, 0.15),
                total_consumption * random.uniform(0.35, 0.5),
                random.uniform(15, 40),
                random.uniform(20, 60),
                random.uniform(5, 15),
                random.uniform(30, 150) * seasonal_consumption
            ))


def generate_exchange_data(cursor, hours, price_areas):
    """Generate energy exchange data with neighboring countries"""
    logger.info(f"Generating {hours} hours of exchange data")

    base_date = datetime.now() - timedelta(hours=hours)
    countries = {
        "DK1": ["germany", "netherlands", "norway", "sweden"],
        "DK2": ["germany", "sweden"]
    }

    for hour in range(hours):
        current_time = base_date + timedelta(hours=hour)

        for area in price_areas:
            for country in countries.get(area, []):
                export = random.uniform(0, 500)
                import_val = random.uniform(0, 500)

                cursor.execute("""
                INSERT INTO exchange_data (
                    HourUTC, Country, PriceArea, Export_MWh, Import_MWh, NetExchange_MWh
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    current_time,
                    country,
                    area,
                    export,
                    import_val,
                    export - import_val
                ))


def generate_carbon_intensity_data(cursor, hours, price_areas):
    """Generate carbon intensity data"""
    logger.info(f"Generating {hours} hours of carbon intensity data")

    base_date = datetime.now() - timedelta(hours=hours)

    for hour in range(hours):
        current_time = base_date + timedelta(hours=hour)

        # Simulate daily and seasonal patterns
        hour_of_day = current_time.hour
        day_of_year = current_time.timetuple().tm_yday

        # Solar production follows daily sun patterns
        solar_factor = max(0, np.sin(np.pi * (hour_of_day - 5) / 14)) if 5 <= hour_of_day <= 19 else 0
        # Seasonal solar adjustment
        seasonal_solar = 0.5 + 0.5 * np.sin(np.pi * (day_of_year - 80) / 180)

        for area in price_areas:
            # Wind percentage varies more randomly
            wind_pct = random.uniform(30, 70)
            solar_pct = solar_factor * seasonal_solar * random.uniform(0, 20)
            hydro_pct = random.uniform(0, 2)
            biomass_pct = random.uniform(5, 15)

            # Calculate fossil percentage as the remainder, with a minimum
            remaining_pct = 100 - (wind_pct + solar_pct + hydro_pct + biomass_pct)
            fossil_pct = max(5, min(40, remaining_pct))

            # Recalculate to ensure total is 100%
            total_assigned = wind_pct + solar_pct + hydro_pct + biomass_pct + fossil_pct
            adjustment_factor = 100 / total_assigned

            wind_pct *= adjustment_factor
            solar_pct *= adjustment_factor
            hydro_pct *= adjustment_factor
            biomass_pct *= adjustment_factor
            fossil_pct *= adjustment_factor

            # Carbon intensity correlated with fossil percentage
            carbon_intensity = 20 + fossil_pct * 6 + random.uniform(-20, 20)

            cursor.execute("""
            INSERT INTO carbon_intensity (
                HourUTC, PriceArea, CarbonIntensity_gCO2perKWh, 
                Wind_Pct, Solar_Pct, Hydro_Pct, Fossil_Pct,
                Nuclear_Pct, Biomass_Pct, Other_Pct
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                current_time,
                area,
                carbon_intensity,
                wind_pct,
                solar_pct,
                hydro_pct,
                fossil_pct,
                0,  # Nuclear is 0 in Denmark
                biomass_pct,
                0  # Other is 0
            ))


def generate_aggregated_production_data(cursor, hours, price_areas):
    """Generate aggregated production data based on the production_data table"""
    logger.info("Generating aggregated production data")

    # First clear any existing data
    cursor.execute("DELETE FROM aggregated_production")

    # Get base date
    base_date = datetime.now() - timedelta(hours=hours)

    # Generate aggregated data for different time periods
    for aggregation_type in ["daily", "weekly", "monthly"]:
        if aggregation_type == "daily":
            # Create daily aggregations
            for day in range(0, hours, 24):
                if day + 24 > hours:
                    continue

                period_start = base_date + timedelta(hours=day)
                period_end = period_start + timedelta(days=1)

                for area in price_areas:
                    total_prod = random.uniform(20000, 40000)
                    wind_prod = total_prod * random.uniform(0.3, 0.6)
                    solar_prod = total_prod * random.uniform(0.05, 0.2)
                    hydro_prod = total_prod * random.uniform(0, 0.02)
                    commercial_prod = total_prod * random.uniform(0.1, 0.25)
                    central_prod = total_prod * random.uniform(0.2, 0.4)

                    cursor.execute("""
                    INSERT INTO aggregated_production (
                        PeriodStart, PeriodEnd, PriceArea, AggregationType,
                        TotalProduction_MWh, WindProduction_MWh, SolarProduction_MWh,
                        HydroProduction_MWh, CommercialProduction_MWh, CentralProduction_MWh
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        period_start, period_end, area, aggregation_type,
                        total_prod, wind_prod, solar_prod, hydro_prod, commercial_prod, central_prod
                    ))

        elif aggregation_type == "weekly":
            # Create weekly aggregations
            for week in range(0, hours, 168):
                if week + 168 > hours:
                    continue

                period_start = base_date + timedelta(hours=week)
                period_end = period_start + timedelta(days=7)

                for area in price_areas:
                    total_prod = random.uniform(140000, 280000)
                    wind_prod = total_prod * random.uniform(0.3, 0.6)
                    solar_prod = total_prod * random.uniform(0.05, 0.2)
                    hydro_prod = total_prod * random.uniform(0, 0.02)
                    commercial_prod = total_prod * random.uniform(0.1, 0.25)
                    central_prod = total_prod * random.uniform(0.2, 0.4)

                    cursor.execute("""
                    INSERT INTO aggregated_production (
                        PeriodStart, PeriodEnd, PriceArea, AggregationType,
                        TotalProduction_MWh, WindProduction_MWh, SolarProduction_MWh,
                        HydroProduction_MWh, CommercialProduction_MWh, CentralProduction_MWh
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        period_start, period_end, area, aggregation_type,
                        total_prod, wind_prod, solar_prod, hydro_prod, commercial_prod, central_prod
                    ))

        elif aggregation_type == "monthly":
            # Create monthly aggregations (simplified approach)
            for month in range(0, hours, 720):  # ~30 days
                if month + 720 > hours:
                    continue

                period_start = base_date + timedelta(hours=month)
                period_end = period_start + timedelta(days=30)

                for area in price_areas:
                    total_prod = random.uniform(600000, 1200000)
                    wind_prod = total_prod * random.uniform(0.3, 0.6)
                    solar_prod = total_prod * random.uniform(0.05, 0.2)
                    hydro_prod = total_prod * random.uniform(0, 0.02)
                    commercial_prod = total_prod * random.uniform(0.1, 0.25)
                    central_prod = total_prod * random.uniform(0.2, 0.4)

                    cursor.execute("""
                    INSERT INTO aggregated_production (
                        PeriodStart, PeriodEnd, PriceArea, AggregationType,
                        TotalProduction_MWh, WindProduction_MWh, SolarProduction_MWh,
                        HydroProduction_MWh, CommercialProduction_MWh, CentralProduction_MWh
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        period_start, period_end, area, aggregation_type,
                        total_prod, wind_prod, solar_prod, hydro_prod, commercial_prod, central_prod
                    ))


def generate_comparison_analysis_data(cursor, hours, price_areas):
    """Generate comparison analysis data"""
    logger.info("Generating comparison analysis data")

    # First clear any existing data
    cursor.execute("DELETE FROM comparison_analysis")

    # Base date for analysis
    base_date = datetime.now() - timedelta(hours=hours)

    # Generate comparison data for a few sample periods
    for i in range(5):
        # Create two periods that are the same length but different times
        period1_start = base_date + timedelta(days=random.randint(0, int(hours / 48)))
        period1_length = random.randint(7, 30)  # 1 week to 1 month
        period1_end = period1_start + timedelta(days=period1_length)

        # Second period starts after first period
        period2_start = period1_end + timedelta(days=random.randint(1, 30))
        period2_end = period2_start + timedelta(days=period1_length)

        for area in price_areas:
            # Generate data for both periods
            main_data = {
                "TotalConsumption_MWh": random.uniform(800000, 1500000),
                "WindProduction_MWh": random.uniform(300000, 800000),
                "SolarProduction_MWh": random.uniform(20000, 100000),
                "CentralProduction_MWh": random.uniform(200000, 600000)
            }

            # Comparison data with some variations
            compare_data = {
                "TotalConsumption_MWh": main_data["TotalConsumption_MWh"] * random.uniform(0.8, 1.2),
                "WindProduction_MWh": main_data["WindProduction_MWh"] * random.uniform(0.7, 1.3),
                "SolarProduction_MWh": main_data["SolarProduction_MWh"] * random.uniform(0.6, 1.4),
                "CentralProduction_MWh": main_data["CentralProduction_MWh"] * random.uniform(0.8, 1.2)
            }

            # Calculate differences
            differences = {k: compare_data[k] - v for k, v in main_data.items()}
            percentages = {k: 100 * (compare_data[k] - v) / v for k, v in main_data.items()}

            # Insert data
            cursor.execute("""
            INSERT INTO comparison_analysis (
                MainPeriodFrom, MainPeriodTo, ComparePeriodFrom, ComparePeriodTo,
                PriceArea, ComparisonType, MainPeriodData, ComparePeriodData,
                Difference, PercentageChange
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                period1_start, period1_end, period2_start, period2_end,
                area,
                random.choice(["production", "consumption", "exchange"]),
                json.dumps(main_data),
                json.dumps(compare_data),
                json.dumps(differences),
                json.dumps(percentages)
            ))


def generate_consumption_forecast_data(cursor, price_areas):
    """
    Generate consumption forecast data with confidence intervals.
    Creates forecasts at different horizons (24h, 48h, 7d) from different reference points.
    """
    import datetime
    import random
    import numpy as np

    # Create consumption forecast table if it doesn't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS consumption_forecast (
        id INT PRIMARY KEY AUTO_INCREMENT,
        forecast_created DATETIME,
        forecast_timestamp DATETIME,
        price_area VARCHAR(10),
        forecast_horizon VARCHAR(10),
        consumption_forecast_mwh FLOAT,
        confidence_low_mwh FLOAT,
        confidence_high_mwh FLOAT,
        INDEX(price_area),
        INDEX(forecast_timestamp),
        INDEX(forecast_created)
    )
    """)

    print("Generating consumption forecast data...")

    # Generate forecasts from different reference points
    now = datetime.datetime.now()
    forecast_points = [
        now - datetime.timedelta(days=30),
        now - datetime.timedelta(days=14),
        now - datetime.timedelta(days=7),
        now - datetime.timedelta(days=3),
        now - datetime.timedelta(days=1),
        now
    ]

    horizons = [
        ("24h", 24),
        ("48h", 48),
        ("7d", 168)  # 7 days * 24 hours
    ]

    # Use actual consumption patterns to inform forecasts
    cursor.execute("""
    SELECT AVG(consumption_total_mwh) as avg_consumption,
           STDDEV(consumption_total_mwh) as std_consumption
    FROM consumption_data
    """)
    result = cursor.fetchone()
    base_consumption = result[0] if result[0] else 2500.0
    std_consumption = result[1] if result[1] else 300.0

    forecast_records = []

    for price_area in price_areas:
        for created_time in forecast_points:
            for horizon_name, hours in horizons:
                # Add hourly variation pattern (24-hour cycle)
                hourly_pattern = [
                    0.85, 0.75, 0.70, 0.65, 0.60, 0.70,  # 00:00-05:59
                    0.80, 0.90, 1.10, 1.15, 1.10, 1.05,  # 06:00-11:59
                    1.00, 1.05, 1.10, 1.10, 1.15, 1.20,  # 12:00-17:59
                    1.25, 1.30, 1.20, 1.10, 1.00, 0.90  # 18:00-23:59
                ]

                # Generate forecasts for each hour
                for h in range(hours):
                    timestamp = created_time + datetime.timedelta(hours=h)

                    # Base hourly consumption with pattern
                    hour_of_day = timestamp.hour
                    day_factor = 0.90 if timestamp.weekday() >= 5 else 1.05  # Weekend vs weekday

                    # Calculate forecast values
                    hourly_consumption = base_consumption * hourly_pattern[hour_of_day] * day_factor

                    # Add some randomness and increase uncertainty with forecast horizon
                    uncertainty = 0.02 + (h / hours) * 0.10
                    forecast_value = hourly_consumption * (1 + random.uniform(-0.05, 0.05))

                    # Calculate confidence intervals
                    confidence_low = forecast_value * (1 - uncertainty)
                    confidence_high = forecast_value * (1 + uncertainty)

                    forecast_records.append((
                        created_time,
                        timestamp,
                        price_area,
                        horizon_name,
                        forecast_value,
                        confidence_low,
                        confidence_high
                    ))

                    # Don't generate too many records for performance
                    if len(forecast_records) >= 1000:
                        cursor.executemany("""
                        INSERT INTO consumption_forecast (
                            forecast_created, forecast_timestamp, price_area, 
                            forecast_horizon, consumption_forecast_mwh,
                            confidence_low_mwh, confidence_high_mwh
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """, forecast_records)
                        forecast_records = []

    # Insert any remaining records
    if forecast_records:
        cursor.executemany("""
        INSERT INTO consumption_forecast (
            forecast_created, forecast_timestamp, price_area, 
            forecast_horizon, consumption_forecast_mwh,
            confidence_low_mwh, confidence_high_mwh
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, forecast_records)

    print(f"Generated consumption forecast data for {len(price_areas)} price areas")

def generate_cache_weights(host='localhost', user='cacheuser', password='cachepass', database='cache_db'):
    """
    Generate initial cache weights for data items.
    This creates a baseline of weights to be used by the caching system.
    """
    logger.info(f"Generating cache weights on {host}")

    try:
        conn = mysql.connector.connect(
            host=host, user=user, password=password, database=database
        )
        cursor = conn.cursor()

        # Check if weights already exist
        cursor.execute("SELECT COUNT(*) FROM cache_weights")
        count = cursor.fetchone()[0]

        if count > 0:
            logger.info("Cache weights already exist, skipping generation")
            return True

        # Get available timestamps and price areas
        cursor.execute("SELECT DISTINCT timestamp FROM energy_data ORDER BY timestamp")
        timestamps = [row[0] for row in cursor.fetchall()]

        cursor.execute("SELECT DISTINCT price_area FROM energy_data")
        price_areas = [row[0] for row in cursor.fetchall()]

        if not timestamps or not price_areas:
            logger.error("No energy data found to generate weights from")
            return False

        # Generate weights for different data types
        data_types = ["energy", "production", "consumption", "exchange", "carbon"]

        for data_type in data_types:
            for timestamp in timestamps:
                # More recent data gets higher recency
                time_diff = (datetime.now() - timestamp).total_seconds() / 86400  # days
                recency = max(0.1, min(1.0, 1.0 - (time_diff / 30)))  # 0.1-1.0 scale, 30 days to minimum

                for area in price_areas:
                    # Randomize other factors
                    access_frequency = max(0.1, min(1.0, np.random.beta(2, 5)))
                    time_relevance = max(0.1, min(1.0, np.random.beta(2, 2)))
                    production_importance = max(0.1, min(1.0, np.random.beta(2, 3)))

                    # Calculate priority
                    calculated_priority = (
                            0.3 * recency +
                            0.3 * access_frequency +
                            0.2 * time_relevance +
                            0.2 * production_importance
                    )

                    # Set last accessed to a random time in the past
                    last_accessed = datetime.now() - timedelta(minutes=np.random.randint(1, 10080))
                    access_count = np.random.randint(1, 50)

                    cursor.execute("""
                    INSERT INTO cache_weights 
                    (data_type, timestamp, price_area, recency, access_frequency, time_relevance, 
                     production_importance, calculated_priority, last_accessed, access_count)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        data_type, timestamp, area, recency, access_frequency, time_relevance,
                        production_importance, calculated_priority, last_accessed, access_count
                    ))

        # Generate weights for derived data endpoints
        generate_derived_data_weights(cursor)

        conn.commit()
        conn.close()

        logger.info("Successfully generated cache weights")
        return True

    except Exception as e:
        logger.error(f"Error generating cache weights: {str(e)}")
        return False


def generate_derived_data_weights(cursor):
    """Generate initial weights for derived data endpoints"""
    logger.info("Generating derived data cache weights")

    # Check if derived weights already exist
    cursor.execute("SELECT COUNT(*) FROM derived_data_cache_weights")
    count = cursor.fetchone()[0]

    if count > 0:
        logger.info("Derived data cache weights already exist, skipping generation")
        return True

    # Define endpoint types
    derived_endpoints = [
        {
            "name": "aggregated_production",
            "volatility_range": (0.4, 0.8),
            "complexity_range": (0.5, 0.9),
        },
        {
            "name": "analysis_comparison",
            "volatility_range": (0.2, 0.5),
            "complexity_range": (0.7, 1.0),
        },
        {
            "name": "forecast_consumption",
            "volatility_range": (0.6, 1.0),
            "complexity_range": (0.6, 0.9),
        },
        {
            "name": "carbon_intensity",
            "volatility_range": (0.3, 0.7),
            "complexity_range": (0.5, 0.8),
        }
    ]

    # Generate sample requests for each endpoint
    for endpoint in derived_endpoints:
        # Generate multiple parameter combinations per endpoint
        for i in range(30):  # 30 variations per endpoint
            params = generate_random_parameters(endpoint["name"])
            param_hash = generate_parameter_hash(params)

            recency = np.random.beta(1.5, 5)
            usage_frequency = np.random.beta(1.2, 6)
            time_relevance = np.random.beta(2, 2)
            production_importance = np.random.beta(1.8, 3)
            volatility = np.random.uniform(*endpoint["volatility_range"])
            complexity = np.random.uniform(*endpoint["complexity_range"])

            weights = {
                'recency': 0.25,
                'access_frequency': 0.20,
                'time_relevance': 0.15,
                'production_importance': 0.15,
                'volatility': 0.15,
                'complexity': 0.10
            }

            priority = calculate_priority(weights, recency, usage_frequency,
                                          time_relevance, production_importance,
                                          volatility, complexity)

            last_accessed = datetime.now() - timedelta(minutes=np.random.randint(1, 10080))

            cursor.execute("""
            INSERT INTO derived_data_cache_weights (
                endpoint, parameter_hash, request_pattern, recency, usage_frequency, 
                time_relevance, production_importance, volatility, complexity,
                calculated_priority, last_accessed, access_count
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                endpoint["name"], param_hash, str(params), recency, usage_frequency,
                time_relevance, production_importance, volatility, complexity,
                priority, last_accessed, np.random.randint(1, 50)
            ))


def generate_random_parameters(endpoint_name):
    """Generate realistic random parameters for the given endpoint"""
    # Implementation copied from simulation.py
    import random
    from datetime import datetime, timedelta

    base_date = datetime.now()

    if endpoint_name == "aggregated_production":
        return {
            "from": (base_date - timedelta(days=random.randint(30, 365))).isoformat(),
            "to": (base_date - timedelta(days=random.randint(1, 29))).isoformat(),
            "priceArea": random.choice(["DK1", "DK2"]),
            "aggregationType": random.choice(["daily", "weekly", "monthly"]),
            "productionType": random.choice(["wind", "solar", "central", "local", "total"])
        }
    elif endpoint_name == "analysis_comparison":
        period1_start = base_date - timedelta(days=random.randint(60, 365))
        period1_end = period1_start + timedelta(days=random.randint(7, 30))
        period2_start = period1_end + timedelta(days=random.randint(1, 30))
        period2_end = period2_start + timedelta(days=(period1_end - period1_start).days)

        return {
            "from": period1_start.isoformat(),
            "to": period1_end.isoformat(),
            "compareFrom": period2_start.isoformat(),
            "compareTo": period2_end.isoformat(),
            "priceArea": random.choice(["DK1", "DK2"]),
            "comparisonType": random.choice(["production", "consumption", "exchange"])
        }
    elif endpoint_name == "forecast_consumption":
        return {
            "from": base_date.isoformat(),
            "to": (base_date + timedelta(days=random.randint(1, 14))).isoformat(),
            "priceArea": random.choice(["DK1", "DK2"]),
            "forecastHorizon": random.choice(["24h", "48h", "7d", "14d"])
        }
    elif endpoint_name == "carbon_intensity":
        return {
            "from": (base_date - timedelta(days=random.randint(7, 90))).isoformat(),
            "to": (base_date - timedelta(days=random.randint(0, 6))).isoformat(),
            "priceArea": random.choice(["DK1", "DK2"]),
            "resolution": random.choice(["hourly", "daily", "weekly"])
        }

    return {}


def generate_parameter_hash(params):
    """Generate a hash for parameter combination"""
    import hashlib
    import json
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()


def calculate_priority(weights, recency, frequency, time_relevance,
                       production_importance, volatility, complexity):
    """Calculate the priority using the weighted formula"""
    return (
            weights['recency'] * recency +
            weights['access_frequency'] * frequency +
            weights['time_relevance'] * time_relevance +
            weights['production_importance'] * production_importance +
            weights['volatility'] * volatility +
            weights['complexity'] * complexity
    )


if __name__ == "__main__":
    # Generate mock database
    generate_mock_database(
        host='localhost',
        user='cacheuser',
        password='cachepass',
        database='cache_db',
        hours=1000,
        price_areas=["DK1", "DK2"]
    )

    # Generate cache weight data
    generate_cache_weights(
        host='localhost',
        user='cacheuser',
        password='cachepass',
        database='cache_db'
    )