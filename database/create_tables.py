# database/create_tables.py
import logging

logger = logging.getLogger(__name__)


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
    db_handler.execute_query(energy_table_sql)
    db_handler.execute_query("CREATE INDEX IF NOT EXISTS idx_energy_hour ON energy_data(timestamp)")
    db_handler.execute_query("CREATE INDEX IF NOT EXISTS idx_energy_area ON energy_data(price_area)")

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
    db_handler.execute_query(production_table_sql)
    db_handler.execute_query("CREATE INDEX IF NOT EXISTS idx_production_ts ON production_data(timestamp)")
    db_handler.execute_query("CREATE INDEX IF NOT EXISTS idx_production_area ON production_data(price_area)")

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
    db_handler.execute_query(consumption_table_sql)
    db_handler.execute_query("CREATE INDEX IF NOT EXISTS idx_consumption_ts ON consumption_data(timestamp)")
    db_handler.execute_query("CREATE INDEX IF NOT EXISTS idx_consumption_area ON consumption_data(price_area)")

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
    db_handler.execute_query(exchange_table_sql)
    db_handler.execute_query("CREATE INDEX IF NOT EXISTS idx_exchange_ts ON exchange_data(timestamp)")
    db_handler.execute_query("CREATE INDEX IF NOT EXISTS idx_exchange_area ON exchange_data(price_area)")

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
    db_handler.execute_query(carbon_table_sql)
    db_handler.execute_query("CREATE INDEX IF NOT EXISTS idx_carbon_ts ON carbon_intensity(timestamp)")
    db_handler.execute_query("CREATE INDEX IF NOT EXISTS idx_carbon_area ON carbon_intensity(price_area)")

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
    db_handler.execute_query(aggregated_production_sql)
    db_handler.execute_query("CREATE INDEX IF NOT EXISTS idx_agg_prod_start ON aggregated_production(period_start)")
    db_handler.execute_query("CREATE INDEX IF NOT EXISTS idx_agg_prod_area ON aggregated_production(price_area)")

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
    db_handler.execute_query(comparison_sql)
    db_handler.execute_query("CREATE INDEX IF NOT EXISTS idx_comparison_area ON comparison_analysis(price_area)")

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
    db_handler.execute_query(consumption_forecast_sql)
    db_handler.execute_query("CREATE INDEX IF NOT EXISTS idx_forecast_date ON consumption_forecast(request_date)")
    db_handler.execute_query("CREATE INDEX IF NOT EXISTS idx_forecast_area ON consumption_forecast(price_area)")

    # Derived data cache weights table
    weights_sql = f"""
    CREATE TABLE IF NOT EXISTS cache_metrics (
        id INTEGER PRIMARY KEY {auto_increment},
        cache_name VARCHAR(50) NOT NULL,
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
    db_handler.execute_query(weights_sql)
    db_handler.execute_query("CREATE INDEX IF NOT EXISTS idx_weights_endpoint ON cache_metrics(endpoint)")
    db_handler.execute_query("CREATE INDEX IF NOT EXISTS idx_weights_hash ON cache_metrics(parameter_hash)")

    # Best models table
    best_models_sql = f"""
    CREATE TABLE IF NOT EXISTS best_models (
        id INTEGER PRIMARY KEY {auto_increment},
        model_name VARCHAR(255) NOT NULL,
        created_at {timestamp_type} DEFAULT CURRENT_TIMESTAMP,
        model_base64 LONGTEXT NOT NULL,
        description TEXT
    )
    """
    db_handler.execute_query(best_models_sql)

    logger.info("All tables created successfully")
    return True

