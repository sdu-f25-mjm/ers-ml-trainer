-- Table: price_area
CREATE TABLE IF NOT EXISTS price_area (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL
);

-- Table: Municipality
CREATE TABLE IF NOT EXISTS municipality (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    price_area_id INT NOT NULL,
    FOREIGN KEY (price_area_id) REFERENCES price_area(id) ON DELETE CASCADE
);

-- Table: Branche
CREATE TABLE IF NOT EXISTS branche (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL
);

-- Table: ConsumptionBranche
CREATE TABLE IF NOT EXISTS consumption (
    id INT AUTO_INCREMENT PRIMARY KEY,
    hourUTC DATETIME NOT NULL,
    municipality_id INT NOT NULL,
    branche_id INT NOT NULL,
    consumption BIGINT NOT NULL,
    FOREIGN KEY (municipality_id) REFERENCES municipality(id) ON DELETE CASCADE,
    FOREIGN KEY (branche_id) REFERENCES branche(id) ON DELETE CASCADE,
    CONSTRAINT unique_hourUTC_municipality_branche UNIQUE (hourUTC, municipality_id, branche_id)
);

-- Table: Production
CREATE TABLE IF NOT EXISTS production (
    id INT AUTO_INCREMENT PRIMARY KEY,
    hourUTC DATETIME NOT NULL,
    price_area_id INT NOT NULL,
    central_power BIGINT NOT NULL,
    local_power BIGINT NOT NULL,
    commercial_power BIGINT NOT NULL,
    commercial_power_self_consumption BIGINT NOT NULL,
    offshore_wind_lt_100mw BIGINT NOT NULL,
    offshore_wind_ge_100mw BIGINT NOT NULL,
    onshore_wind_lt_50kw BIGINT NOT NULL,
    onshore_wind_ge_50kw BIGINT NOT NULL,
    hydro_power BIGINT NOT NULL,
    solar_power_lt_10kw BIGINT NOT NULL,
    solar_power_ge_10kw_lt_40kw BIGINT NOT NULL,
    solar_power_ge_40kw BIGINT NOT NULL,
    solar_power_self_consumption BIGINT NOT NULL,
    unknown_production BIGINT NOT NULL,
    FOREIGN KEY (price_area_id) REFERENCES price_area(id) ON DELETE CASCADE,
    CONSTRAINT unique_hourUTC_price_area UNIQUE (hourUTC, price_area_id)
);

-- Table: Exchange
CREATE TABLE IF NOT EXISTS exchange (
    id INT AUTO_INCREMENT PRIMARY KEY,
    hourUTC DATETIME NOT NULL,
    price_area_id INT NOT NULL,
    exchange_to_norway BIGINT NOT NULL,
    exchange_to_sweden BIGINT NOT NULL,
    exchange_to_germany BIGINT NOT NULL,
    exchange_to_netherlands BIGINT NOT NULL,
    exchange_to_gb BIGINT NOT NULL,
    exchange_over_the_great_belt BIGINT NOT NULL,
    FOREIGN KEY (price_area_id) REFERENCES price_area(id) ON DELETE CASCADE,
    CONSTRAINT unique_hourUTC_price_area_exchange UNIQUE (hourUTC, price_area_id)
);

-- Table: request_cache
CREATE TABLE IF NOT EXISTS request_cache (
    id INT AUTO_INCREMENT PRIMARY KEY,
    request_url VARCHAR(2048) NOT NULL,
    request_time DATETIME NOT NULL,
    response_time DATETIME NOT NULL,
    response_size INT NOT NULL,
    was_in_cache BOOLEAN NOT NULL
    );


-- Table: cache_metrics
CREATE TABLE IF NOT EXISTS cache_metrics
(
    id INT auto_increment primary key,
    cache_key LONGTEXT NOT NULL,
    cache_name LONGTEXT NOT NULL,
    hit_ratio FLOAT null,
    item_count INT null,
    load_time_ms FLOAT null,
    policy_triggered BIT null,
    rl_action_taken VARCHAR(255) null,
    size_bytes BIGINT null,
    timestamp DATETIME(6) null,
    traffic_intensity FLOAT null
);

-- Table: rl_models
CREATE TABLE IF NOT EXISTS rl_models
(
    id INT AUTO_INCREMENT PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    model_base64 LONGTEXT NOT NULL,
    description TEXT,
    model_type VARCHAR(64),
    input_dimension INT,
    algorithm VARCHAR(64),
    device VARCHAR(64),
    cache_size INT,
    batch_size INT,
    learning_rate FLOAT,
    timesteps INT,
    feature_columns TEXT,
    trained_at VARCHAR(64)
);

