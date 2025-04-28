# database/simulation.py
import logging
import random
import time
import json
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)


def simulate_cache_metrics(db_handler, update_interval=5, run_duration=None, stop_event=None):
    """
    Simulate cache metrics for derived data endpoints using any database handler.

    Args:
        db_handler: Database handler instance (MySQL, PostgreSQL, or SQLite)
        update_interval: Time between updates in seconds
        run_duration: Total simulation time in seconds (None for indefinite)
        stop_event: Threading event for stopping simulation
    """
    logger.info(f"Starting cache metrics simulation")
    start_time = time.time()

    # Define cache names for simulation
    cache_names = [
        "aggregated_production_cache",
        "analysis_comparison_cache",
        "forecast_consumption_cache",
        "carbon_intensity_cache"
    ]

    try:
        # Check if we need to initialize with data
        result = db_handler.execute_query("SELECT COUNT(*) FROM cache_metrics")
        count = 0
        if result:
            row = result.fetchone()
            if hasattr(row, 'values'):
                count = list(row.values())[0]
            elif isinstance(row, dict):
                count = row.get(list(row.keys())[0], 0)
            elif isinstance(row, tuple):
                count = row[0]

        if count == 0:
            logger.info("Initializing cache metrics table")
            placeholder = db_handler.get_placeholder_symbol()

            for cache_name in cache_names:
                for i in range(10):  # 10 variations per cache
                    hit_ratio = round(random.uniform(0.7, 0.99), 3)
                    item_count = random.randint(100, 1000)
                    load_time_ms = round(random.uniform(5, 100), 2)
                    policy_triggered = random.choice([0, 1])
                    rl_action_taken = random.choice(["evict", "keep", "promote", "demote"])
                    size_bytes = random.randint(10_000_000, 500_000_000)
                    timestamp = datetime.now() - timedelta(minutes=random.randint(1, 10080))
                    traffic_intensity = round(random.uniform(0.1, 2.0), 3)

                    query = f"""
                    INSERT INTO cache_metrics (
                        cache_name, hit_ratio, item_count, load_time_ms, policy_triggered,
                        rl_action_taken, size_bytes, timestamp, traffic_intensity
                    ) VALUES (
                        {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder}, {placeholder}
                    )
                    """
                    db_handler.execute_query(query, (
                        cache_name, hit_ratio, item_count, load_time_ms, policy_triggered,
                        rl_action_taken, size_bytes, timestamp, traffic_intensity
                    ))

            db_handler.commit()

        # Start simulation loop
        while (run_duration is None or (time.time() - start_time < run_duration)):
            if stop_event and stop_event.is_set():
                logger.info("Stopping cache metrics simulation due to stop signal")
                break

            placeholder = db_handler.get_placeholder_symbol()
            now = datetime.now()

            for cache_name in cache_names:
                # Randomly decide if this cache is accessed/updated
                if random.random() < 0.7:
                    # Get all entries for this cache
                    query = f"SELECT * FROM cache_metrics WHERE cache_name = {placeholder}"
                    cursor = db_handler.execute_query(query, (cache_name,))
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
                        elif hasattr(item, 'keys'):
                            normalized_items.append({k: item[k] for k in item.keys()})
                        elif isinstance(item, tuple) and hasattr(cursor, 'description'):
                            normalized_items.append({
                                cursor.description[i][0]: item[i]
                                for i in range(len(cursor.description))
                            })

                    if not normalized_items:
                        continue

                    # Select a few items to "access/update"
                    accessed_items = random.sample(
                        normalized_items,
                        k=min(3, len(normalized_items))
                    )

                    for item in accessed_items:
                        # Simulate metric changes
                        hit_ratio = min(0.99, max(0.5, item.get('hit_ratio', 0.8) + random.uniform(-0.01, 0.01)))
                        item_count = max(1, item.get('item_count', 100) + random.randint(-5, 5))
                        load_time_ms = max(1.0, item.get('load_time_ms', 50.0) + random.uniform(-2, 2))
                        policy_triggered = random.choice([0, 1])
                        rl_action_taken = random.choice(["evict", "keep", "promote", "demote"])
                        size_bytes = max(1_000_000, item.get('size_bytes', 10_000_000) + random.randint(-100_000, 100_000))
                        traffic_intensity = max(0.01, item.get('traffic_intensity', 1.0) + random.uniform(-0.05, 0.05))

                        query = f"""
                        UPDATE cache_metrics SET
                            hit_ratio = {placeholder},
                            item_count = {placeholder},
                            load_time_ms = {placeholder},
                            policy_triggered = {placeholder},
                            rl_action_taken = {placeholder},
                            size_bytes = {placeholder},
                            timestamp = {placeholder},
                            traffic_intensity = {placeholder}
                        WHERE id = {placeholder}
                        """
                        db_handler.execute_query(query, (
                            hit_ratio, item_count, load_time_ms, policy_triggered,
                            rl_action_taken, size_bytes, now, traffic_intensity, item.get('id')
                        ))

            db_handler.commit()
            logger.info(f"Updated cache metrics at {now}")

            time.sleep(update_interval)

        logger.info(f"Cache metrics simulation ended after {time.time() - start_time:.1f} seconds")
        return True

    except Exception as e:
        logger.error(f"Error in cache metrics simulation: {e}")
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

