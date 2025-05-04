import logging

logger = logging.getLogger(__name__)


def safe_execute(db_handler, sql, description):
    try:
        db_handler.execute_query(sql)
        logger.info(f"Table/index for '{description}' created or already exists.")
    except Exception as e:
        logger.error(f"Failed to create table/index for '{description}': {e}")


def create_tables(db_handler):
    """Create all necessary database tables using the provided handler"""
    auto_increment = db_handler.get_auto_increment_syntax()
    timestamp_type = db_handler.get_timestamp_type()

    # Energy data table
    energy_table_sql = f"""
    CREATE TABLE IF NOT EXISTS energy_data (
        id INTEGER PRIMARY KEY {auto_increment},
        timestamp {timestamp_type},
        price_area VARCHAR(10),
        central_power_mwh FLOAT,
        local_power_mwh FLOAT,
        gross_consumption_mwh FLOAT,
        exchange_no_mwh FLOAT,
        exchange_se_mwh FLOAT,
        exchange_de_mwh FLOAT,
        solar_power_self_con_mwh FLOAT,
        grid_loss_transmission_mwh FLOAT
    )
    """
    safe_execute(db_handler, energy_table_sql, "energy_data")
    safe_execute(db_handler, "CREATE INDEX IF NOT EXISTS idx_energy_hour ON energy_data(timestamp)",
                 "energy_data timestamp index")
    safe_execute(db_handler, "CREATE INDEX IF NOT EXISTS idx_energy_area ON energy_data(price_area)",
                 "energy_data price_area index")

    # Production data table
    production_table_sql = f"""
    CREATE TABLE IF NOT EXISTS production_data (
        id INTEGER PRIMARY KEY {auto_increment},
        timestamp {timestamp_type},
        price_area VARCHAR(10),
        wind_total_mwh FLOAT,
        offshore_wind_total_mwh FLOAT,
        offshore_wind_lt100mw_mwh FLOAT,
        offshore_wind_ge100mw_mwh FLOAT,
        onshore_wind_total_mwh FLOAT,
        onshore_wind_ge50_kw_mwh FLOAT,
        solar_total_mwh FLOAT,
        solar_total_no_self_consumption_mwh FLOAT,
        solar_power_self_consumption_mwh FLOAT,
        solar_power_ge40_kw_mwh FLOAT,
        solar_power_ge10_lt40_kw_mwh FLOAT,
        solar_power_lt10_kw_mwh FLOAT,
        commercial_power_mwh FLOAT,
        commercial_power_self_consumption_mwh FLOAT,
        central_power_mwh FLOAT,
        hydro_power_mwh FLOAT,
        local_power_mwh FLOAT
    )
    """
    safe_execute(db_handler, production_table_sql, "production_data")
    safe_execute(db_handler, "CREATE INDEX IF NOT EXISTS idx_production_ts ON production_data(timestamp)",
                 "production_data timestamp index")
    safe_execute(db_handler, "CREATE INDEX IF NOT EXISTS idx_production_area ON production_data(price_area)",
                 "production_data price_area index")

    # Consumption data table
    consumption_table_sql = f"""
    CREATE TABLE IF NOT EXISTS consumption_data (
        id INTEGER PRIMARY KEY {auto_increment},
        timestamp {timestamp_type},
        price_area VARCHAR(10),
        consumption_total_mwh FLOAT,
        consumption_private_mwh FLOAT,
        consumption_public_total_mwh FLOAT,
        consumption_commertial_total_mwh FLOAT,
        grid_loss_transmission_mwh FLOAT,
        grid_loss_interconnectors_mwh FLOAT,
        grid_loss_distribution_mwh FLOAT,
        power_to_heat_mwh FLOAT
    )
    """
    safe_execute(db_handler, consumption_table_sql, "consumption_data")
    safe_execute(db_handler, "CREATE INDEX IF NOT EXISTS idx_consumption_ts ON consumption_data(timestamp)",
                 "consumption_data timestamp index")
    safe_execute(db_handler, "CREATE INDEX IF NOT EXISTS idx_consumption_area ON consumption_data(price_area)",
                 "consumption_data price_area index")

    # Exchange data table
    exchange_table_sql = f"""
    CREATE TABLE IF NOT EXISTS exchange_data (
        id INTEGER PRIMARY KEY {auto_increment},
        timestamp {timestamp_type},
        price_area VARCHAR(10),
        exchange_country VARCHAR(50),
        import_mwh FLOAT,
        export_mwh FLOAT,
        net_exchange_mwh FLOAT
    )
    """
    safe_execute(db_handler, exchange_table_sql, "exchange_data")
    safe_execute(db_handler, "CREATE INDEX IF NOT EXISTS idx_exchange_ts ON exchange_data(timestamp)",
                 "exchange_data timestamp index")
    safe_execute(db_handler, "CREATE INDEX IF NOT EXISTS idx_exchange_area ON exchange_data(price_area)",
                 "exchange_data price_area index")

    # Carbon intensity data table
    carbon_table_sql = f"""
    CREATE TABLE IF NOT EXISTS carbon_intensity (
        id INTEGER PRIMARY KEY {auto_increment},
        timestamp {timestamp_type},
        price_area VARCHAR(10),
        carbon_intensity_gco2per_kwh FLOAT,
        energy_mix TEXT
    )
    """
    safe_execute(db_handler, carbon_table_sql, "carbon_intensity")
    safe_execute(db_handler, "CREATE INDEX IF NOT EXISTS idx_carbon_ts ON carbon_intensity(timestamp)",
                 "carbon_intensity timestamp index")
    safe_execute(db_handler, "CREATE INDEX IF NOT EXISTS idx_carbon_area ON carbon_intensity(price_area)",
                 "carbon_intensity price_area index")

    # Aggregated production table
    aggregated_production_sql = f"""
    CREATE TABLE IF NOT EXISTS aggregated_production (
        id INTEGER PRIMARY KEY {auto_increment},
        period_start {timestamp_type},
        period_end {timestamp_type},
        price_area VARCHAR(10),
        aggregation_type VARCHAR(20),
        total_production_mwh FLOAT,
        wind_production_mwh FLOAT,
        solar_production_mwh FLOAT,
        hydro_production_mwh FLOAT,
        commercial_production_mwh FLOAT,
        central_production_mwh FLOAT
    )
    """
    safe_execute(db_handler, aggregated_production_sql, "aggregated_production")
    safe_execute(db_handler, "CREATE INDEX IF NOT EXISTS idx_agg_prod_start ON aggregated_production(period_start)",
                 "aggregated_production period_start index")
    safe_execute(db_handler, "CREATE INDEX IF NOT EXISTS idx_agg_prod_area ON aggregated_production(price_area)",
                 "aggregated_production price_area index")

    # Comparison analysis table
    comparison_sql = f"""
    CREATE TABLE IF NOT EXISTS comparison_analysis (
        id INTEGER PRIMARY KEY {auto_increment},
        generated_on {timestamp_type},
        price_area VARCHAR(10),
        period1_start {timestamp_type},
        period1_end {timestamp_type},
        period2_start {timestamp_type},
        period2_end {timestamp_type},
        comparison_type VARCHAR(30),
        difference TEXT,
        percentage_change TEXT
    )
    """
    safe_execute(db_handler, comparison_sql, "comparison_analysis")
    safe_execute(db_handler, "CREATE INDEX IF NOT EXISTS idx_comparison_area ON comparison_analysis(price_area)",
                 "comparison_analysis price_area index")

    # Consumption forecast table
    consumption_forecast_sql = f"""
    CREATE TABLE IF NOT EXISTS consumption_forecast (
        id INTEGER PRIMARY KEY {auto_increment},
        request_date {timestamp_type},
        price_area VARCHAR(10),
        forecast_horizon INTEGER,
        forecast_data LONGTEXT
    )
    """
    safe_execute(db_handler, consumption_forecast_sql, "consumption_forecast")
    safe_execute(db_handler, "CREATE INDEX IF NOT EXISTS idx_forecast_date ON consumption_forecast(request_date)",
                 "consumption_forecast request_date index")
    safe_execute(db_handler, "CREATE INDEX IF NOT EXISTS idx_forecast_area ON consumption_forecast(price_area)",
                 "consumption_forecast price_area index")

    # Derived data cache weights table
    weights_sql = f"""
    CREATE TABLE IF NOT EXISTS cache_metrics (
        id INTEGER PRIMARY KEY {auto_increment},
        cache_name LONGTEXT NOT NULL,
        cache_key VARCHAR(255) NOT null,
        hit_ratio FLOAT,
        item_count INT,
        load_time_ms FLOAT,
        policy_triggered bit,
        rl_action_taken VARCHAR(255),
        size_bytes BIGINT,
        timestamp {timestamp_type},
        traffic_intensity FLOAT
    )
"""
    safe_execute(db_handler, weights_sql, "cache_metrics")

    # Best models table
    rl_models_sql = f"""
    CREATE TABLE IF NOT EXISTS rl_models (
        id INTEGER PRIMARY KEY AUTO_INCREMENT,
        model_name VARCHAR(255) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        model_base64 LONGTEXT NOT NULL,
        description LONGTEXT,
        model_type VARCHAR(255) NULL,
        input_dimension INT NULL
    )
    """
    safe_execute(db_handler, rl_models_sql, "rl_models")

    logger.info("All tables creation attempted (see errors above if any failed)")
    return True
