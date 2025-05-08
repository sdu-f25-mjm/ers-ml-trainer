import hashlib
import json
import random
import time
import urllib.parse
from datetime import datetime, timedelta

API_ENDPOINTS = [
    "/production",
    "/consumption",
    "/exchange",
]

BRANCHE = ["Erhverv", "Offentligt", "Privat"]
PRICE_AREAS = ["DK1", "DK2"]
PRODUCTION_TYPES = ["WIND", "SOLAR", "HYDRO", "COMMERCIAL_POWER", "CENTRAL_POWER"]
EXCHANGE_COUNTRIES = ["germany", "greatbritain", "netherlands", "norway", "sweden"]
COMPARISON_TYPES = ["PRODUCTION", "CONSUMPTION", "EXCHANGE"]


def random_iso_date(start, end):
    """Return random ISO8601 datetime string between two datetimes."""
    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return (start + timedelta(seconds=random_seconds)).isoformat()


def generate_params(endpoint):
    now = datetime.utcnow()
    week_ago = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)

    if endpoint == "/production":
        return {
            "from": random_iso_date(month_ago, now),
            "to": random_iso_date(week_ago, now),
            "priceArea": random.choice(PRICE_AREAS),
            "productionType": random.choice(PRODUCTION_TYPES)
        }
    elif endpoint == "/consumption":
        return {
            "from": random_iso_date(month_ago, now),
            "to": random_iso_date(week_ago, now),
            "priceArea": random.choice(PRICE_AREAS)
        }
    elif endpoint == "/exchange":
        return {
            "from": random_iso_date(month_ago, now),
            "to": random_iso_date(week_ago, now),
            "priceArea": random.choice(PRICE_AREAS)
        }
    else:
        return {}


def params_hash(params):
    """Create a short hash from parameters for grouping cache entries."""
    params_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()[:8]


def url_hash(url_string):
    """Create a hash from the full URL string."""
    return hashlib.md5(url_string.encode()).hexdigest()


def filter_stable_params(endpoint, params):
    """Remove date/time parameters for cache_name normalization."""
    # Define which params are considered 'stable' for each endpoint
    stable_keys = {
        "/production": ["priceArea", "productionType"],
        "/consumption": ["priceArea"],
        "/exchange": ["priceArea"],
    }
    keys = stable_keys.get(endpoint, [])
    return {k: v for k, v in params.items() if k in keys}


def simulate_visits(
        n=10000,  # Increased default from 100 to 10000 (or any larger number)
        sleep=0,
        db_handler=None,
        update_interval=5,
        run_duration=None,
        stop_event=None
):
    start_time = time.time()
    i = 0
    while (run_duration is None and i < n) or (run_duration is not None and (time.time() - start_time < run_duration)):
        if stop_event and stop_event.is_set():
            break

        endpoint = random.choice(API_ENDPOINTS)
        params = generate_params(endpoint)
        visit_time = datetime.now()

        # --- Normalize cache_name: only stable params, no date/time ---
        stable_params = filter_stable_params(endpoint, params)
        query_string = urllib.parse.urlencode(stable_params)
        cache_name = f"{endpoint}?{query_string}" if query_string else endpoint
        cache_key = url_hash(cache_name)

        hit_ratio = round(random.uniform(0.75, 0.97), 3)
        item_count = random.randint(50, 500)
        load_time_ms = round(random.uniform(40, 180), 2)

        policy_triggered = random.choice([0, 1])
        rl_action_taken = random.choice(["evict", "keep", "promote", "demote"])
        size_bytes = item_count * random.randint(1000, 20000)
        traffic_intensity = round(item_count / max(visit_time.second + 1, 1) * random.uniform(0.5, 1.5), 3)

        if db_handler is not None:
            placeholder = db_handler.get_placeholder_symbol()
            # Update INSERT INTO statements to match the columns in database/01_create_tables.sql
            if endpoint == "/production":
                query = f"""
                    INSERT INTO production (
                        hourUTC, price_area_id, central_power, local_power, commercial_power,
                        commercial_power_self_consumption, offshore_wind_lt_100mw, offshore_wind_ge_100mw,
                        onshore_wind_lt_50kw, onshore_wind_ge_50
