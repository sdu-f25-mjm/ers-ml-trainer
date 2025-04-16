# database/simulation.py
import logging
import random
import time
import json
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)


def simulate_derived_data_weights(db_handler, update_interval=5, run_duration=None, stop_event=None):
    """
    Simulate cache weights for derived data endpoints using any database handler.

    Args:
        db_handler: Database handler instance (MySQL, PostgreSQL, or SQLite)
        update_interval: Time between updates in seconds
        run_duration: Total simulation time in seconds (None for indefinite)
        stop_event: Threading event for stopping simulation
    """
    logger.info(f"Starting derived data cache weight simulation")
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
        }
    ]

    try:
        # Create table if it doesn't exist - this is now handled in create_tables.py

        # Check if we need to initialize with data
        result = db_handler.execute_query("SELECT COUNT(*) FROM derived_data_cache_weights")
        count = 0
        if result:
            row = result.fetchone()
            # Handle different result types based on handler
            if hasattr(row, 'values'):
                count = list(row.values())[0]  # Dict-like result
            elif isinstance(row, dict):
                count = row.get(list(row.keys())[0], 0)
            elif isinstance(row, tuple):
                count = row[0]  # Tuple result

        if count == 0:
            logger.info("Initializing derived data cache weights")
            placeholder = db_handler.get_placeholder_symbol()

            # Generate sample requests for each endpoint
            for endpoint in derived_endpoints:
                # Generate multiple parameter combinations per endpoint
                for i in range(10):  # 10 variations per endpoint
                    params = generate_random_parameters(endpoint["name"])
                    param_hash = generate_parameter_hash(params)

                    recency = float(np.random.beta(1.5, 5))
                    usage_frequency = float(np.random.beta(1.2, 6))
                    time_relevance = float(np.random.beta(2, 2))
                    production_importance = float(np.random.beta(1.8, 3))
                    volatility = float(np.random.uniform(*endpoint["volatility_range"]))
                    complexity = float(np.random.uniform(*endpoint["complexity_range"]))

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

                    # Use parameterized query with correct placeholder format
                    query = f"""
                    INSERT INTO derived_data_cache_weights (
                        endpoint, parameter_hash, request_pattern, recency, usage_frequency, 
                        time_relevance, production_importance, volatility, complexity,
                        calculated_priority, last_accessed, access_count
                    ) VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, 
                             {placeholder}, {placeholder}, {placeholder}, {placeholder},
                             {placeholder}, {placeholder}, {placeholder})
                    """

                    db_handler.execute_query(query, (
                        endpoint["name"], param_hash, str(params), recency, usage_frequency,
                        time_relevance, production_importance, volatility, complexity,
                        priority, last_accessed, np.random.randint(1, 50)
                    ))

            db_handler.commit()

        # Start simulation loop
        while (run_duration is None or (time.time() - start_time < run_duration)):
            # Check if stop signal received
            if stop_event and stop_event.is_set():
                logger.info("Stopping derived data simulation due to stop signal")
                break

            placeholder = db_handler.get_placeholder_symbol()

            # Simulate access patterns across endpoints
            now = datetime.now()  # Define now at the very start so it is in scope.
            for endpoint in derived_endpoints:
                if random.random() <= endpoint["access_probability"]:
                    # This endpoint is being accessed this interval

                    # Get all entries for this endpoint
                    query = f"SELECT * FROM derived_data_cache_weights WHERE endpoint = {placeholder}"
                    cursor = db_handler.execute_query(query, (endpoint["name"],))
                    if not cursor:
                        continue

                    items = cursor.fetchall()
                    if not items:
                        continue

                    # Normalize items to dictionaries if they aren't already
                    normalized_items = []
                    for item in items:
                        if isinstance(item, dict):
                            normalized_items.append(item)
                        elif hasattr(item, 'keys'):  # Dict-like object
                            normalized_items.append({k: item[k] for k in item.keys()})
                        elif isinstance(item, tuple) and hasattr(cursor, 'description'):
                            # Convert tuple to dict using cursor description
                            normalized_items.append({
                                cursor.description[i][0]: item[i]
                                for i in range(len(cursor.description))
                            })

                    if not normalized_items:
                        continue

                    # Select a few items to "access"
                    weights = [item.get('calculated_priority', 0.5) + 0.1 for item in normalized_items]
                    accessed_items = random.choices(
                        normalized_items,
                        weights=weights,
                        k=min(3, len(normalized_items))
                    )

                    # Update accessed items
                    for item in accessed_items:
                        # Update statistics
                        recency = min(1.0, item.get('recency', 0.5) * 0.7 + 0.3)
                        frequency = min(1.0,
                                        item.get('usage_frequency', 0.5) + (1 / (item.get('access_count', 10) + 10)))
                        time_relevance = max(0.1, item.get('time_relevance', 0.5) - endpoint[
                            "time_relevance_decay"] * random.random())
                        volatility = max(0.1, min(1.0, item.get('volatility', 0.5) + random.normalvariate(0, 0.02)))
                        complexity = max(0.1, min(1.0, item.get('complexity', 0.5) + random.normalvariate(0, 0.01)))

                        weights = {
                            'recency': 0.25, 'access_frequency': 0.20, 'time_relevance': 0.15,
                            'production_importance': 0.15, 'volatility': 0.15, 'complexity': 0.10
                        }

                        priority = calculate_priority(
                            weights, recency, frequency, time_relevance,
                            item.get('production_importance', 0.5), volatility, complexity
                        )

                        query = f"""
                        UPDATE derived_data_cache_weights SET
                            recency = {placeholder}, 
                            usage_frequency = {placeholder}, 
                            time_relevance = {placeholder},
                            volatility = {placeholder}, 
                            complexity = {placeholder}, 
                            calculated_priority = {placeholder},
                            last_accessed = {placeholder}, 
                            access_count = access_count + 1
                        WHERE id = {placeholder}
                        """
                        db_handler.execute_query(query, (
                            recency, frequency, time_relevance, volatility, complexity,
                            priority, now, item.get('id')
                        ))

            db_handler.commit()
            logger.info(f"Updated derived data weights at {now}")

            time.sleep(update_interval)

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
