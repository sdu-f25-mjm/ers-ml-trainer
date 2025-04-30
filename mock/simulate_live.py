import hashlib
import json
import random
import time
import urllib.parse
from datetime import datetime, timedelta

API_ENDPOINTS = [
    "/data",
    "/production",
    "/consumption",
    "/exchange",
    "/aggregated-production",
    "/comparison",
    "/forecast",
    "/carbon-intensity"
]

PRICE_AREAS = ["DK1", "DK2"]
PRODUCTION_TYPES = ["WIND", "SOLAR", "HYDRO", "COMMERCIAL_POWER", "CENTRAL_POWER"]
EXCHANGE_COUNTRIES = ["GERMANY", "GREATBRITAIN", "NETHERLANDS", "NORWAY", "SWEDEN"]
AGGREGATION_TYPES = ["HOURLY", "DAILY", "WEEKLY", "MONTHLY", "YEARLY"]
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

    if endpoint == "/data":
        return {
            "from": random_iso_date(month_ago, now),
            "to": random_iso_date(week_ago, now),
            "priceArea": random.choice(PRICE_AREAS)
        }
    elif endpoint == "/production":
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
            "exchangeCountry": random.choice(EXCHANGE_COUNTRIES)
        }
    elif endpoint == "/aggregated-production":
        return {
            "from": random_iso_date(month_ago, now),
            "to": random_iso_date(week_ago, now),
            "priceArea": random.choice(PRICE_AREAS),
            "aggregationType": random.choice(AGGREGATION_TYPES),
            "productionType": random.choice(PRODUCTION_TYPES)
        }
    elif endpoint == "/comparison":
        a = random_iso_date(month_ago, now)
        b = random_iso_date(week_ago, now)
        c = random_iso_date(month_ago, now)
        d = random_iso_date(week_ago, now)
        return {
            "from": a,
            "to": b,
            "compareFrom": c,
            "compareTo": d,
            "priceArea": random.choice(PRICE_AREAS),
            "productionType": random.choice(PRODUCTION_TYPES),
            "comparisonType": random.choice(COMPARISON_TYPES),
            "exchangeCountry": random.choice(EXCHANGE_COUNTRIES)
        }
    elif endpoint == "/forecast":
        return {
            "priceArea": random.choice(PRICE_AREAS),
            "horizon": random.randint(1, 720)
        }
    elif endpoint == "/carbon-intensity":
        return {
            "from": random_iso_date(month_ago, now),
            "to": random_iso_date(week_ago, now),
            "priceArea": random.choice(PRICE_AREAS),
            "productionType": random.choice(PRODUCTION_TYPES),
            "aggregationType": random.choice(AGGREGATION_TYPES)
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

        endpoint = random.choices(
            API_ENDPOINTS,
            weights=[25, 20, 20, 10, 10, 5, 5, 5],
            k=1
        )[0]
        params = generate_params(endpoint)
        visit_time = datetime.now()

        # Construct the full URL with parameters
        query_string = urllib.parse.urlencode(params)
        full_url = f"{endpoint}?{query_string}" if query_string else endpoint

        # Use full URL as cache_name and its hash as cache_key
        cache_name = full_url
        cache_key = url_hash(full_url)

        # Simulate metrics based on endpoint/params
        if endpoint == "/data":
            hit_ratio = round(random.uniform(0.8, 0.98), 3)
            item_count = random.randint(500, 2000)
            load_time_ms = round(random.uniform(30, 100), 2)
        elif endpoint == "/forecast":
            hit_ratio = round(random.uniform(0.6, 0.85), 3)
            item_count = random.randint(10, 100)
            load_time_ms = round(random.uniform(120, 300), 2)
        elif endpoint == "/aggregated-production":
            hit_ratio = round(random.uniform(0.7, 0.95), 3)
            item_count = random.randint(100, 500)
            load_time_ms = round(random.uniform(80, 200), 2)
        else:
            hit_ratio = round(random.uniform(0.75, 0.97), 3)
            item_count = random.randint(50, 500)
            load_time_ms = round(random.uniform(40, 180), 2)

        policy_triggered = random.choice([0, 1])
        rl_action_taken = random.choice(["evict", "keep", "promote", "demote"])
        size_bytes = item_count * random.randint(1000, 20000)
        traffic_intensity = round(item_count / max(visit_time.second + 1, 1) * random.uniform(0.5, 1.5), 3)

        if db_handler is not None:
            placeholder = db_handler.get_placeholder_symbol()
            query = f"""
                INSERT INTO cache_metrics (
                    cache_name, cache_key, hit_ratio, item_count, load_time_ms,
                    policy_triggered, rl_action_taken, size_bytes, timestamp, traffic_intensity
                ) VALUES (
                    {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder},
                    {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}
                )
            """
            db_handler.execute_query(
                query,
                (
                    cache_name,
                    cache_key,
                    hit_ratio,
                    item_count,
                    load_time_ms,
                    policy_triggered,
                    rl_action_taken,
                    size_bytes,
                    visit_time,
                    traffic_intensity
                )
            )
            db_handler.commit()

        if sleep:
            time.sleep(sleep)
        i += 1
