# database/create_tables.py
import logging

logger = logging.getLogger(__name__)


def create_tables(db_handler):
    """Create all necessary database tables using the provided handler"""
    auto_increment = db_handler.get_auto_increment_syntax()
    timestamp_type = db_handler.get_timestamp_type()

    # Energy data table (unchanged)
    energy_table_sql = f"""
    CREATE TABLE IF NOT EXISTS energy_data (
        id INTEGER PRIMARY KEY {auto_increment},
        HourUTC {timestamp_type},
        PriceArea VARCHAR(10),
        CentralPower_MWh FLOAT,
        LocalPower_MWh FLOAT,
        GrossConsumption_MWh FLOAT,
        ExchangeNO_MWh FLOAT,
        ExchangeSE_MWh FLOAT,
        ExchangeDE_MWh FLOAT,
        SolarPowerSelfConMWh FLOAT,
        GridLossTransmissionMWh FLOAT
    )
    """
    db_handler.execute_query(energy_table_sql)
    db_handler.execute_query("CREATE INDEX IF NOT EXISTS idx_energy_hour ON energy_data(HourUTC)")
    db_handler.execute_query("CREATE INDEX IF NOT EXISTS idx_energy_area ON energy_data(PriceArea)")

    # Production data table (updated to include detailed production columns)
    production_table_sql = f"""
    CREATE TABLE IF NOT EXISTS production_data (
        id INTEGER PRIMARY KEY {auto_increment},
        timestamp {timestamp_type},
        PriceArea VARCHAR(10),
        WindTotal_MWh FLOAT,
        OffshoreWindTotal_MWh FLOAT,
        OffshoreWindLt100MW_MWh FLOAT,
        OffshoreWindGe100MW_MWh FLOAT,
        OnshoreWindTotal_MWh FLOAT,
        SolarTotal_MWh FLOAT,
        SolarTotalNoSelfConsumption_MWh FLOAT,
        SolarPowerSelfConsumption_MWh FLOAT,
        SolarPowerGe40kW_MWh FLOAT,
        SolarPowerGe10Lt40kW_MWh FLOAT,
        CommercialPower_MWh FLOAT,
        CommercialPowerSelfConsumption_MWh FLOAT,
        CentralPower_MWh FLOAT,
        HydroPower_MWh FLOAT,
        LocalPower_MWh FLOAT
    )
    """
    db_handler.execute_query(production_table_sql)
    db_handler.execute_query("CREATE INDEX IF NOT EXISTS idx_production_ts ON production_data(timestamp)")
    db_handler.execute_query("CREATE INDEX IF NOT EXISTS idx_production_area ON production_data(PriceArea)")

    # Consumption data table (updated based on OpenAPI schema)
    consumption_table_sql = f"""
    CREATE TABLE IF NOT EXISTS consumption_data (
        id INTEGER PRIMARY KEY {auto_increment},
        timestamp {timestamp_type},
        PriceArea VARCHAR(10),
        ConsumptionTotal_MWh FLOAT,
        ConsumptionPrivate_MWh FLOAT,
        ConsumptionPublicTotal_MWh FLOAT,
        ConsumptionCommertialTotal_MWh FLOAT,
        GridLossTransmission_MWh FLOAT,
        GridLossInterconnectors_MWh FLOAT,
        GridLossDistribution_MWh FLOAT,
        PowerToHeatMWh FLOAT
    )
    """
    db_handler.execute_query(consumption_table_sql)
    db_handler.execute_query("CREATE INDEX IF NOT EXISTS idx_consumption_ts ON consumption_data(timestamp)")
    db_handler.execute_query("CREATE INDEX IF NOT EXISTS idx_consumption_area ON consumption_data(PriceArea)")

    # Exchange data table (updated)
    exchange_table_sql = f"""
    CREATE TABLE IF NOT EXISTS exchange_data (
        id INTEGER PRIMARY KEY {auto_increment},
        timestamp {timestamp_type},
        PriceArea VARCHAR(10),
        exchange_country VARCHAR(50),
        Import_MWh FLOAT,
        Export_MWh FLOAT,
        NetExchange_MWh FLOAT
    )
    """
    db_handler.execute_query(exchange_table_sql)
    db_handler.execute_query("CREATE INDEX IF NOT EXISTS idx_exchange_ts ON exchange_data(timestamp)")
    db_handler.execute_query("CREATE INDEX IF NOT EXISTS idx_exchange_area ON exchange_data(PriceArea)")

    # Carbon intensity data table (updated with energy mix as JSON stored in a TEXT field)
    carbon_table_sql = f"""
    CREATE TABLE IF NOT EXISTS carbon_intensity_data (
        id INTEGER PRIMARY KEY {auto_increment},
        timestamp {timestamp_type},
        PriceArea VARCHAR(10),
        CarbonIntensity_gCO2perKWh FLOAT,
        EnergyMix TEXT
    )
    """
    db_handler.execute_query(carbon_table_sql)
    db_handler.execute_query("CREATE INDEX IF NOT EXISTS idx_carbon_ts ON carbon_intensity_data(timestamp)")
    db_handler.execute_query("CREATE INDEX IF NOT EXISTS idx_carbon_area ON carbon_intensity_data(PriceArea)")

    # Aggregated production table (updated matching OpenAPI)
    aggregated_production_sql = f"""
    CREATE TABLE IF NOT EXISTS aggregated_production (
        id INTEGER PRIMARY KEY {auto_increment},
        PeriodStart {timestamp_type},
        PeriodEnd {timestamp_type},
        PriceArea VARCHAR(10),
        AggregationType VARCHAR(20),
        TotalProduction_MWh FLOAT,
        WindProduction_MWh FLOAT,
        SolarProduction_MWh FLOAT,
        HydroProduction_MWh FLOAT,
        CommercialProduction_MWh FLOAT,
        CentralProduction_MWh FLOAT
    )
    """
    db_handler.execute_query(aggregated_production_sql)
    db_handler.execute_query("CREATE INDEX IF NOT EXISTS idx_agg_prod_start ON aggregated_production(PeriodStart)")
    db_handler.execute_query("CREATE INDEX IF NOT EXISTS idx_agg_prod_area ON aggregated_production(PriceArea)")

    # Comparison analysis table (updated based on OpenAPI schema)
    comparison_sql = f"""
    CREATE TABLE IF NOT EXISTS comparison_analysis (
        id INTEGER PRIMARY KEY {auto_increment},
        generated_on {timestamp_type},
        PriceArea VARCHAR(10),
        Period1_Start {timestamp_type},
        Period1_End {timestamp_type},
        Period2_Start {timestamp_type},
        Period2_End {timestamp_type},
        ComparisonType VARCHAR(30),
        -- Storing difference and percentage change as JSON for flexibility
        Difference TEXT,
        PercentageChange TEXT
    )
    """
    db_handler.execute_query(comparison_sql)
    db_handler.execute_query("CREATE INDEX IF NOT EXISTS idx_comparison_area ON comparison_analysis(PriceArea)")

    # Consumption forecast table (updated)
    consumption_forecast_sql = f"""
    CREATE TABLE IF NOT EXISTS consumption_forecast (
        id INTEGER PRIMARY KEY {auto_increment},
        RequestDate {timestamp_type},
        PriceArea VARCHAR(10),
        ForecastHorizon INTEGER,
        ForecastData LONGTEXT
    )
    """
    db_handler.execute_query(consumption_forecast_sql)
    db_handler.execute_query("CREATE INDEX IF NOT EXISTS idx_forecast_date ON consumption_forecast(RequestDate)")
    db_handler.execute_query("CREATE INDEX IF NOT EXISTS idx_forecast_area ON consumption_forecast(PriceArea)")

    # Derived data cache weights table (remains similar)
    weights_sql = f"""
    CREATE TABLE IF NOT EXISTS derived_data_cache_weights (
        id INTEGER PRIMARY KEY {auto_increment},
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
        last_accessed {timestamp_type},
        access_count INT
    )
    """
    db_handler.execute_query(weights_sql)
    db_handler.execute_query("CREATE INDEX IF NOT EXISTS idx_weights_endpoint ON derived_data_cache_weights(endpoint)")
    db_handler.execute_query("CREATE INDEX IF NOT EXISTS idx_weights_hash ON derived_data_cache_weights(parameter_hash)")

    logger.info("All tables created successfully")
    return True