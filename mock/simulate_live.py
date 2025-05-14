import os
import random
import time
import urllib.parse
from datetime import datetime, timedelta

import dotenv
import requests
import yaml

from config import API_URL

dotenv.load_dotenv()




# --- Dynamically extract GET endpoints from ers-api.yaml ---
def extract_api_endpoints(yaml_path, include_only: list = None):
    with open(yaml_path, "r", encoding="utf-8") as f:
        api_spec = yaml.safe_load(f)
    endpoints = []
    paths = api_spec.get("paths", {})
    for path, methods in paths.items():
        if "get" in methods:
            if include_only is None or path in include_only:
                endpoints.append(path)
    return endpoints


# Example usage:
# To include only /production and /consumption endpoints:
ERS_API_YAML = os.path.join(os.path.dirname(__file__), "..", "ers-api.yaml")
API_ENDPOINTS = extract_api_endpoints(ERS_API_YAML,include_only=["/exchange/country","/production/total","/consumption/total"])
# API_ENDPOINTS = extract_api_endpoints(ERS_API_YAML)

# Path to ers-api.yaml (adjust if needed)

PRICE_AREAS = ["DK1", "DK2"]
PRODUCTION_TYPES = ["WIND", "SOLAR", "HYDRO", "COMMERCIAL_POWER", "CENTRAL_POWER"]
EXCHANGE_COUNTRIES = ["germany", "greatbritain", "netherlands", "norway", "sweden"]


def random_iso_date(start, end):
    if start > end:
        start, end = end, start
    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    dt = start + timedelta(seconds=random_seconds)
    # Format as 'YYYY-MM-DDTHH:MM'
    return dt.strftime("%Y-%m-%dT%H:%M")


def generate_params(endpoint):
    now = datetime.utcnow()
    # Only allow dates from 2020-01-01 to now
    start_2021 = datetime(2021, 1, 1)
    end_2021 = datetime(2021, 3, 1)

    def ordered_dates(a, b):
        """Return (earlier, later) as datetime objects."""
        return (a, b) if a <= b else (b, a)

    if endpoint == "/production":
        from_date, to_date = ordered_dates(start_2021, end_2021)
        from_val = random_iso_date(from_date, to_date)
        to_val = random_iso_date(from_date, to_date)
        if from_val > to_val:
            from_val, to_val = to_val, from_val
        return {
            "from": from_val,
            "to": to_val,
            "priceArea": random.choice(PRICE_AREAS),
            "type": random.choice(PRODUCTION_TYPES)
        }
    elif endpoint == "/production/total":
        from_date, to_date = ordered_dates(start_2021, end_2021)
        from_val = random_iso_date(from_date, to_date)
        to_val = random_iso_date(from_date, to_date)
        if from_val > to_val:
            from_val, to_val = to_val, from_val
        return {
            "from": from_val,
            "to": to_val
        }
    elif endpoint == "/consumption":
        from_date, to_date = ordered_dates(start_2021, end_2021)
        from_val = random_iso_date(from_date, to_date)
        to_val = random_iso_date(from_date, to_date)
        if from_val > to_val:
            from_val, to_val = to_val, from_val
        return {
            "from": from_val,
            "to": to_val,
            "priceArea": random.choice(PRICE_AREAS)
        }
    elif endpoint == "/consumption/total":
        from_date, to_date = ordered_dates(start_2021, end_2021)
        from_val = random_iso_date(from_date, to_date)
        to_val = random_iso_date(from_date, to_date)
        if from_val > to_val:
            from_val, to_val = to_val, from_val
        return {
            "from": from_val,
            "to": to_val,
            "priceArea": random.choice(PRICE_AREAS)
        }
    elif endpoint == "/exchange":
        from_date, to_date = ordered_dates(start_2021, end_2021)
        from_val = random_iso_date(from_date, to_date)
        to_val = random_iso_date(from_date, to_date)
        if from_val > to_val:
            from_val, to_val = to_val, from_val
        return {
            "from": from_val,
            "to": to_val,
            "priceArea": random.choice(PRICE_AREAS),
        }
    elif endpoint == "/exchange/country":
        from_date, to_date = ordered_dates(start_2021, end_2021)
        from_val = random_iso_date(from_date, to_date)
        to_val = random_iso_date(from_date, to_date)
        if from_val > to_val:
            from_val, to_val = to_val, from_val
        return {
            "from": from_val,
            "to": to_val,
        }
    else:
        return {}

def simulate_visits(
        n=1000,
        update_interval=0,
        api_url=f"{API_URL}",
        run_duration=None,
        stop_event=None,
        endpoints_to_use: int = None,
):
    """
    Simulate random user visits by making HTTP GET requests to API endpoints.
    The first 10 (or endpoints_to_use if set) unique requests are generated and saved,
    and then only those exact requests are reused for the rest of the simulation.
    """
    # Save the first N unique full URLs (endpoint + params)
    saved_requests = []
    saved_requests_set = set()
    max_saved = endpoints_to_use if endpoints_to_use and endpoints_to_use > 0 else 10

    start_time = time.time()
    i = 0
    # Phase 1: Generate and save the first N unique requests
    while len(saved_requests) < max_saved:
        endpoint = random.choice(API_ENDPOINTS)
        params = generate_params(endpoint)
        query_string = urllib.parse.urlencode(params)
        url = f"{api_url}{endpoint}"
        if query_string:
            url = f"{url}?{query_string}"
        if url not in saved_requests_set:
            saved_requests.append(url)
            saved_requests_set.add(url)
            try:
                response = requests.get(url)
                print(f"[{datetime.now().isoformat()}] GET {url} -> {response.status_code}")
            except requests.exceptions.SSLError as ssl_err:
                print(f"[{datetime.now().isoformat()}] SSL ERROR requesting {url}: {ssl_err}")
            except Exception as e:
                print(f"[{datetime.now().isoformat()}] ERROR requesting {url}: {e}")
            if update_interval:
                time.sleep(update_interval)
            i += 1

    # Phase 2: Reuse only the saved requests for the rest of the simulation
    while ((run_duration is None and i < n) or
           (run_duration is not None and (time.time() - start_time < run_duration))):
        if stop_event and stop_event.is_set():
            break

        url = random.choice(saved_requests)
        try:
            response = requests.get(url)
            print(f"[{datetime.now().isoformat()}] GET {url} -> {response.status_code}")
        except requests.exceptions.SSLError as ssl_err:
            print(f"[{datetime.now().isoformat()}] SSL ERROR requesting {url}: {ssl_err}")
        except Exception as e:
            print(f"[{datetime.now().isoformat()}] ERROR requesting {url}: {e}")

        if update_interval:
            time.sleep(update_interval)
        i += 1
