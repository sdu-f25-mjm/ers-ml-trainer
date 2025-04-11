import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def simulate_derived_data_weights(host='ers-mariadb', user='cacheuser', password='cachepass', database='cache_db',
                                  update_interval=5, run_duration=None, stop_event=None):
    """
    Simulate cache weights for derived data endpoints like aggregations and forecasts.

    This simulates how users interact with computed/aggregated data differently
    than raw data, reflecting real-world access patterns.
    """
    import time
    from datetime import datetime, timedelta
    import random
    import numpy as np
    import mysql.connector

    logger.info(f"Starting derived data cache weight simulation on {host}")
    start_time = time.time()

    # Define endpoint types with their characteristics
    derived_endpoints = [
        {
            "name": "aggregated_production",
            "volatility_range": (0.4, 0.8),
            "complexity_range": (0.5, 0.9),
            "time_relevance_decay": 0.05,
            "access_probability": 0.3
        },
        {
            "name": "analysis_comparison",
            "volatility_range": (0.2, 0.5),
            "complexity_range": (0.7, 1.0),
            "time_relevance_decay": 0.02,
            "access_probability": 0.2
        },
        {
            "name": "forecast_consumption",
            "volatility_range": (0.6, 1.0),
            "complexity_range": (0.6, 0.9),
            "time_relevance_decay": 0.1,
            "access_probability": 0.4
        },
        {
            "name": "carbon_intensity",
            "volatility_range": (0.3, 0.7),
            "complexity_range": (0.5, 0.8),
            "time_relevance_decay": 0.03,
            "access_probability": 0.5
        },
        {
            "name": "exchange_analysis",
            "volatility_range": (0.4, 0.7),
            "complexity_range": (0.6, 0.8),
            "time_relevance_decay": 0.04,
            "access_probability": 0.25
        },
        {
            "name": "renewable_share",
            "volatility_range": (0.2, 0.6),
            "complexity_range": (0.4, 0.7),
            "time_relevance_decay": 0.03,
            "access_probability": 0.45
        },
        {
            "name": "consumption_patterns",
            "volatility_range": (0.3, 0.5),
            "complexity_range": (0.5, 0.8),
            "time_relevance_decay": 0.02,
            "access_probability": 0.35
        },
        {
            "name": "production_mix",
            "volatility_range": (0.3, 0.6),
            "complexity_range": (0.5, 0.7),
            "time_relevance_decay": 0.04,
            "access_probability": 0.4
        },
        {
            "name": "grid_balance",
            "volatility_range": (0.5, 0.9),
            "complexity_range": (0.7, 0.9),
            "time_relevance_decay": 0.08,
            "access_probability": 0.3
        },
        {
            "name": "carbon_forecast",
            "volatility_range": (0.6, 0.9),
            "complexity_range": (0.7, 1.0),
            "time_relevance_decay": 0.1,
            "access_probability": 0.35
        },
        {
            "name": "peak_demand_analysis",
            "volatility_range": (0.5, 0.8),
            "complexity_range": (0.6, 0.9),
            "time_relevance_decay": 0.06,
            "access_probability": 0.25
        },
        {
            "name": "grid_stability_metrics",
            "volatility_range": (0.7, 0.9),
            "complexity_range": (0.8, 1.0),
            "time_relevance_decay": 0.09,
            "access_probability": 0.2
        },
        {
            "name": "self_consumption_ratio",
            "volatility_range": (0.2, 0.5),
            "complexity_range": (0.4, 0.6),
            "time_relevance_decay": 0.04,
            "access_probability": 0.3
        },
        {
            "name": "price_correlation",
            "volatility_range": (0.4, 0.7),
            "complexity_range": (0.7, 0.9),
            "time_relevance_decay": 0.05,
            "access_probability": 0.4
        },
        {
            "name": "seasonal_patterns",
            "volatility_range": (0.1, 0.4),
            "complexity_range": (0.6, 0.8),
            "time_relevance_decay": 0.01,
            "access_probability": 0.25
        },
        {
            "name": "renewable_forecast",
            "volatility_range": (0.6, 0.9),
            "complexity_range": (0.7, 0.9),
            "time_relevance_decay": 0.08,
            "access_probability": 0.35
        },
        {
            "name": "import_export_balance",
            "volatility_range": (0.3, 0.6),
            "complexity_range": (0.5, 0.7),
            "time_relevance_decay": 0.04,
            "access_probability": 0.3
        },
        {
            "name": "weather_impact_analysis",
            "volatility_range": (0.6, 0.8),
            "complexity_range": (0.7, 0.9),
            "time_relevance_decay": 0.05,
            "access_probability": 0.2
        },
        {
            "name": "load_shifting_potential",
            "volatility_range": (0.5, 0.8),
            "complexity_range": (0.6, 0.9),
            "time_relevance_decay": 0.06,
            "access_probability": 0.25
        },
        {
            "name": "transmission_losses",
            "volatility_range": (0.2, 0.5),
            "complexity_range": (0.5, 0.7),
            "time_relevance_decay": 0.03,
            "access_probability": 0.15
        }
    ]

    try:
        conn = mysql.connector.connect(
            host=host, user=user, password=password, database=database
        )

        # Create table if it doesn't exist
        cursor = conn.cursor()
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
        conn.commit()

        # Generate initial data if table is empty
        cursor.execute("SELECT COUNT(*) FROM derived_data_cache_weights")
        count = cursor.fetchone()[0]

        if count == 0:
            logger.info("Initializing derived data cache weights")
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
            conn.commit()

        # Start simulation loop
        while (run_duration is None or (time.time() - start_time < run_duration)):
            # Check if stop signal received
            if stop_event and stop_event.is_set():
                logger.info("Stopping derived data simulation due to stop signal")
                break

            cursor = conn.cursor(dictionary=True)

            # Simulate access patterns across endpoints
            for endpoint in derived_endpoints:
                if random.random() <= endpoint["access_probability"]:
                    # This endpoint is being accessed this interval

                    # Get all entries for this endpoint
                    cursor.execute(
                        "SELECT * FROM derived_data_cache_weights WHERE endpoint = %s",
                        (endpoint["name"],)
                    )
                    items = cursor.fetchall()

                    if not items:
                        continue

                    # Select a few items to "access"
                    weights = [item['calculated_priority'] + 0.1 for item in items]
                    accessed_items = random.choices(
                        items,
                        weights=weights,
                        k=min(3, len(items))
                    )

                    # Update accessed items
                    now = datetime.now()
                    for item in accessed_items:
                        # Update statistics
                        recency = min(1.0, item['recency'] * 0.7 + 0.3)
                        frequency = min(1.0, item['usage_frequency'] + (1 / (item['access_count'] + 10)))

                        # Time relevance decays based on endpoint characteristics
                        time_relevance = max(0.1, item['time_relevance'] - endpoint["time_relevance_decay"] *
                                             random.random())

                        # Allow slight variation in other parameters
                        volatility = max(0.1, min(1.0, item['volatility'] + np.random.normal(0, 0.02)))
                        complexity = max(0.1, min(1.0, item['complexity'] + np.random.normal(0, 0.01)))

                        weights = {
                            'recency': 0.25, 'access_frequency': 0.20, 'time_relevance': 0.15,
                            'production_importance': 0.15, 'volatility': 0.15, 'complexity': 0.10
                        }

                        priority = calculate_priority(weights, recency, frequency,
                                                      time_relevance, item['production_importance'],
                                                      volatility, complexity)

                        cursor.execute("""
                        UPDATE derived_data_cache_weights SET
                            recency = %s, usage_frequency = %s, time_relevance = %s,
                            volatility = %s, complexity = %s, calculated_priority = %s,
                            last_accessed = %s, access_count = access_count + 1
                        WHERE id = %s
                        """, (
                            recency, frequency, time_relevance, volatility, complexity,
                            priority, now, item['id']
                        ))

            conn.commit()
            logger.info(f"Updated derived data weights at {now}")

            # Decay the items that weren't accessed
            if random.random() < 0.3:
                cursor.execute("""
                UPDATE derived_data_cache_weights 
                SET recency = GREATEST(0.1, recency * 0.95),
                    calculated_priority = (
                        0.25 * GREATEST(0.1, recency * 0.95) +
                        0.20 * usage_frequency +
                        0.15 * time_relevance +
                        0.15 * production_importance +
                        0.15 * volatility +
                        0.10 * complexity
                    )
                """)
                conn.commit()

            time.sleep(update_interval)

        conn.close()
        logger.info(f"Derived data simulation ended after {time.time() - start_time:.1f} seconds")
        return True

    except Exception as e:
        logger.error(f"Error in derived data cache simulation: {e}")
        return False


def generate_random_parameters(endpoint_name):
    """Generate realistic random parameters for the given endpoint"""
    import random
    import json
    from datetime import datetime, timedelta

    base_date = datetime.now()
    price_areas = ["DK1", "DK2"]

    # Common time ranges
    short_range = {
        "from": (base_date - timedelta(days=random.randint(1, 7))).isoformat(),
        "to": base_date.isoformat()
    }
    medium_range = {
        "from": (base_date - timedelta(days=random.randint(7, 30))).isoformat(),
        "to": base_date.isoformat()
    }
    long_range = {
        "from": (base_date - timedelta(days=random.randint(30, 90))).isoformat(),
        "to": base_date.isoformat()
    }

    # Basic parameters
    params = {
        "priceArea": random.choice(price_areas)
    }

    if endpoint_name == "aggregated_production":
        params.update(medium_range)
        params.update({
            "aggregationType": random.choice(["daily", "weekly", "monthly"]),
            "productionType": random.choice(["wind", "solar", "central", "local", "total"])
        })
    elif endpoint_name == "analysis_comparison":
        period1_start = base_date - timedelta(days=random.randint(60, 365))
        period1_end = period1_start + timedelta(days=random.randint(7, 30))
        period2_start = period1_end + timedelta(days=random.randint(1, 30))
        period2_end = period2_start + timedelta(days=(period1_end - period1_start).days)

        params.update({
            "from": period1_start.isoformat(),
            "to": period1_end.isoformat(),
            "compareFrom": period2_start.isoformat(),
            "compareTo": period2_end.isoformat(),
            "comparisonType": random.choice(["production", "consumption", "exchange"])
        })
    elif endpoint_name == "forecast_consumption":
        params.update({
            "from": base_date.isoformat(),
            "to": (base_date + timedelta(days=random.randint(1, 14))).isoformat(),
            "forecastHorizon": random.choice(["24h", "48h", "7d", "14d"])
        })
    elif endpoint_name == "carbon_intensity":
        params.update(medium_range)
        params.update({
            "resolution": random.choice(["hourly", "daily", "weekly"])
        })
    elif endpoint_name == "exchange_analysis":
        params.update(medium_range)
        params.update({
            "country": random.choice(["germany", "norway", "sweden", "netherlands"]),
            "flowType": random.choice(["import", "export", "net"])
        })
    elif endpoint_name == "renewable_share":
        params.update(medium_range)
        params.update({
            "includeTypes": json.dumps(random.sample(["wind", "solar", "hydro", "biomass"], k=random.randint(1, 4))),
            "resolution": random.choice(["hourly", "daily", "weekly"])
        })
    elif endpoint_name == "consumption_patterns":
        params.update(long_range)
        params.update({
            "consumerType": random.choice(["residential", "commercial", "industrial", "all"]),
            "pattern": random.choice(["daily", "weekly", "seasonal"])
        })
    elif endpoint_name == "production_mix":
        params.update(medium_range)
        params.update({
            "aggregation": random.choice(["hourly", "daily", "weekly"]),
            "showPercentage": random.choice([True, False])
        })
    elif endpoint_name == "grid_balance":
        params.update(short_range)
        params.update({
            "includeExchange": random.choice([True, False]),
            "resolution": random.choice(["15min", "hourly"])
        })
    elif endpoint_name == "carbon_forecast":
        params.update({
            "from": base_date.isoformat(),
            "to": (base_date + timedelta(days=random.randint(1, 7))).isoformat(),
            "resolution": random.choice(["hourly", "daily"])
        })
    elif endpoint_name == "peak_demand_analysis":
        params.update(long_range)
        params.update({
            "threshold": random.randint(80, 95),
            "includeWeather": random.choice([True, False])
        })
    elif endpoint_name == "grid_stability_metrics":
        params.update(medium_range)
        params.update({
            "metricTypes": json.dumps(random.sample(["frequency", "voltage", "congestion", "reserves"],
                                                    k=random.randint(1, 4))),
            "criticalThreshold": random.uniform(0.7, 0.9)
        })
    elif endpoint_name == "self_consumption_ratio":
        params.update(medium_range)
        params.update({
            "consumerSize": random.choice(["small", "medium", "large", "all"]),
            "withStorage": random.choice([True, False])
        })
    elif endpoint_name == "price_correlation":
        params.update(long_range)
        params.update({
            "correlationWith": random.choice(["wind", "solar", "demand", "temperature"]),
            "lagHours": random.randint(0, 24)
        })
    elif endpoint_name == "seasonal_patterns":
        params.update({
            "year": random.randint(datetime.now().year - 3, datetime.now().year),
            "patternType": random.choice(["production", "consumption", "prices"]),
            "resolution": random.choice(["daily", "weekly", "monthly"])
        })
    elif endpoint_name == "renewable_forecast":
        params.update({
            "from": base_date.isoformat(),
            "to": (base_date + timedelta(days=random.randint(1, 5))).isoformat(),
            "types": json.dumps(random.sample(["wind", "solar"], k=random.randint(1, 2))),
            "confidence": random.choice([80, 90, 95])
        })
    elif endpoint_name == "import_export_balance":
        params.update(medium_range)
        params.update({
            "countries": json.dumps(random.sample(["germany", "norway", "sweden", "netherlands"],
                                                  k=random.randint(1, 4))),
            "aggregation": random.choice(["hourly", "daily", "weekly"])
        })
    elif endpoint_name == "weather_impact_analysis":
        params.update(medium_range)
        params.update({
            "weatherType": json.dumps(random.sample(["temperature", "wind", "cloud_cover", "precipitation"],
                                                    k=random.randint(1, 4))),
            "impactOn": random.choice(["production", "consumption", "prices"])
        })
    elif endpoint_name == "load_shifting_potential":
        params.update(medium_range)
        params.update({
            "sectorType": random.choice(["residential", "commercial", "industrial", "all"]),
            "shiftHours": random.randint(1, 6),
            "priceThreshold": random.uniform(30, 70)
        })
    elif endpoint_name == "transmission_losses":
        params.update(medium_range)
        params.update({
            "gridLevel": random.choice(["transmission", "distribution", "all"]),
            "includeReactiveLoading": random.choice([True, False])
        })
    else:
        # Fallback for any undefined endpoints
        params.update(medium_range)

    return params


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