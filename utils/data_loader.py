"""
utils/data_loader.py
All data sources: synthetic, Kaggle CSV, live APIs.
Fixes applied: #5 #6 #7 #8 #9 #10 #11 #16 #18 #21 #22 #23 #24 #38
"""

import os
import numpy as np
import pandas as pd
import requests
import time
from datetime import datetime, timedelta

# Load .env file so API keys are available
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))
except ImportError:
    pass  # dotenv not installed — rely on system env vars

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AIRPORTS = {
    'VOMM': {'name': 'Chennai (MAA)',    'lat': 12.9941, 'lon': 80.1709, 'runways': 2},
    'VIDP': {'name': 'Delhi (DEL)',      'lat': 28.5562, 'lon': 77.1000, 'runways': 3},
    'VABB': {'name': 'Mumbai (BOM)',     'lat': 19.0896, 'lon': 72.8656, 'runways': 2},
    'VOBL': {'name': 'Bangalore (BLR)', 'lat': 13.1979, 'lon': 77.7063, 'runways': 2},
    'VOCI': {'name': 'Kochi (COK)',      'lat': 10.1520, 'lon': 76.4019, 'runways': 1},
}


# ─────────────────────────────────────────
#  SYNTHETIC DATA GENERATOR
# ─────────────────────────────────────────

def generate_synthetic_data(n_days=30, save=True):
    """Generate realistic synthetic ATC data for all airports."""
    records = []
    base = datetime.now() - timedelta(days=n_days)

    for icao, info in AIRPORTS.items():
        for day in range(n_days):
            for hour in range(24):
                # FIX #23: single timestamp per record, not multiple datetime.now() calls
                ts = base + timedelta(days=day, hours=hour)

                # Peak-hour traffic patterns
                if 6 <= hour <= 10 or 17 <= hour <= 21:
                    fph = np.random.randint(25, 48)
                    delay = np.random.uniform(10, 45)
                elif 22 <= hour or hour <= 5:
                    fph = np.random.randint(3, 15)
                    delay = np.random.uniform(0, 8)
                else:
                    fph = np.random.randint(12, 30)
                    delay = np.random.uniform(5, 20)

                wind = np.random.uniform(5, 60)
                vis = np.random.uniform(1000, 10000)
                rain = max(0, np.random.exponential(3))
                # FIX #21: guard cloud_cover with max(0, ...)
                cloud = max(0.0, min(1.0, np.random.beta(2, 5) + np.random.normal(0, 0.05)))
                active_rwy = max(1, info['runways'] - np.random.choice([0, 0, 0, 1]))
                notam = np.random.choice([0, 0, 0, 0, 1])
                notam_sev = notam * np.random.uniform(0.3, 1.0)

                # FIX #22: actually include runway_util in the record
                runway_util = round(min(1.0, fph / (active_rwy * 18)), 3)

                records.append({
                    'timestamp': ts,
                    'airport_icao': icao,
                    'lat': info['lat'] + np.random.uniform(-0.01, 0.01),
                    'lon': info['lon'] + np.random.uniform(-0.01, 0.01),
                    'flights_per_hour': fph,
                    'avg_delay_min': round(delay, 1),
                    'wind_speed_kmh': round(wind, 1),
                    'visibility_m': round(vis),
                    'precipitation_mm': round(rain, 2),
                    'cloud_cover': round(cloud, 3),
                    'active_runways': active_rwy,
                    'total_runways': info['runways'],
                    'restriction_flag': notam,
                    'notam_severity': round(notam_sev, 3),
                    'runway_util_ratio': runway_util,
                })

    df = pd.DataFrame(records)
    df.sort_values(['airport_icao', 'timestamp'], inplace=True)

    if save:
        # FIX #5: create data/ directory before saving
        save_dir = os.path.join(BASE_DIR, "data")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "synthetic_atc_data.csv")
        df.to_csv(save_path, index=False)
        print(f"[data] Saved synthetic data: {len(df):,} rows → {save_path}")

    return df


# ─────────────────────────────────────────
#  LIVE WEATHER — Open-Meteo (free, no key)
# ─────────────────────────────────────────

def fetch_weather_live(airport_info):
    """Fetch current weather from Open-Meteo.
    FIX #6: Use hardcoded URL, not broken env var.
    FIX #7: wind_speed_10m (not windspeed_10m).
    FIX #8: cloud_cover (not cloudcover).
    FIX #9: Remove visibility — not available in current weather free tier.
    """
    try:
        lat, lon = airport_info['lat'], airport_info['lon']
        # FIX #6: hardcode the URL — OPEN_METEO_BASE env var was never reliably set
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': lat,
            'longitude': lon,
            # FIX #7 #8: correct field names
            'current': 'temperature_2m,wind_speed_10m,wind_gusts_10m,rain,cloud_cover',
            'timezone': 'auto',
        }
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json().get('current', {})
            return {
                'wind_speed_kmh': data.get('wind_speed_10m', 10.0),
                # FIX #9: estimate visibility from cloud cover + rain (not available directly)
                'visibility_m': max(1000, 10000 - data.get('rain', 0) * 500
                                    - data.get('cloud_cover', 20) * 30),
                'precipitation_mm': data.get('rain', 0.0),
                'cloud_cover': data.get('cloud_cover', 20.0) / 100.0,
                'temperature_c': data.get('temperature_2m', 25.0),
                'source': 'LIVE_OPENMETEO',
            }
        return None
    except Exception as e:
        print(f"[weather] API error: {e}")
        return None


def fetch_live_weather_all():
    """Fetch live weather for all airports. Returns {icao: weather_dict}."""
    weather_map = {}
    for icao, info in AIRPORTS.items():
        w = fetch_weather_live(info)
        if w:
            weather_map[icao] = w
        time.sleep(0.3)
    return weather_map


# ─────────────────────────────────────────
#  LIVE FLIGHTS — OpenSky (OAuth2 since March 18, 2026)
# ─────────────────────────────────────────

_opensky_token_cache = {'token': None, 'expires': 0}

def _get_opensky_token():
    """Get OAuth2 Bearer token using client_id + client_secret.
    Basic auth was deprecated on March 18, 2026.
    Token is cached for 25 minutes (they expire at 30)."""
    import time as _t
    if _opensky_token_cache['token'] and _t.time() < _opensky_token_cache['expires']:
        return _opensky_token_cache['token']

    client_id = os.getenv("OPENSKY_CLIENT_ID", "")
    client_secret = os.getenv("OPENSKY_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        return None  # Fall back to anonymous

    try:
        resp = requests.post(
            "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token",
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            token = data.get("access_token")
            _opensky_token_cache['token'] = token
            _opensky_token_cache['expires'] = _t.time() + 25 * 60  # cache 25 min
            print(f"[opensky] OAuth2 token acquired (expires in 25 min)")
            return token
        else:
            print(f"[opensky] Token request failed: {resp.status_code}")
            return None
    except Exception as e:
        print(f"[opensky] OAuth2 error: {e}")
        return None


def fetch_opensky_live(airport_info, radius_km=150):
    """Fetch live flight count around an airport.
    Uses OAuth2 Bearer token if credentials exist, otherwise anonymous (400 req/day)."""
    try:
        lat, lon = airport_info['lat'], airport_info['lon']
        delta = radius_km / 111.0
        params = {
            'lamin': lat - delta, 'lamax': lat + delta,
            'lomin': lon - delta, 'lomax': lon + delta,
        }

        headers = {}
        token = _get_opensky_token()
        if token:
            headers['Authorization'] = f'Bearer {token}'

        resp = requests.get("https://opensky-network.org/api/states/all",
                            params=params, headers=headers, timeout=15)
        if resp.status_code == 200:
            states = resp.json().get('states', [])
            return len(states) if states else 0
        elif resp.status_code == 429:
            print(f"[opensky] Rate limited — using simulated data")
            return None
        return None
    except Exception as e:
        print(f"[opensky] API error: {e}")
        return None


# ─────────────────────────────────────────
#  LIVE FLIGHTS — AviationStack (free tier: 100 req/month)
# ─────────────────────────────────────────

def fetch_aviationstack(airport_iata):
    """Fetch live flight data from AviationStack.
    Free tier: 100 requests/month, no credit card.
    Returns flight count + avg delay for an airport."""
    api_key = os.getenv("AVIATIONSTACK_KEY", "")
    if not api_key:
        return None

    try:
        # Get arrivals at this airport
        resp = requests.get(
            "http://api.aviationstack.com/v1/flights",
            params={
                'access_key': api_key,
                'arr_iata': airport_iata,
                'flight_status': 'active',
                'limit': 50,
            },
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            flights = data.get('data', [])
            if not flights:
                return None

            delays = []
            for f in flights:
                dep_delay = f.get('departure', {}).get('delay')
                arr_delay = f.get('arrival', {}).get('delay')
                d = arr_delay or dep_delay
                if d is not None:
                    delays.append(float(d))

            return {
                'flight_count': len(flights),
                'avg_delay_min': round(np.mean(delays), 1) if delays else 0.0,
                'source': 'LIVE_AVIATIONSTACK',
            }
        else:
            err = resp.json().get('error', {}).get('message', resp.status_code)
            print(f"[aviationstack] Error: {err}")
            return None
    except Exception as e:
        print(f"[aviationstack] API error: {e}")
        return None


# IATA codes for our Indian airports (AviationStack uses IATA not ICAO)
ICAO_TO_IATA = {
    'VOMM': 'MAA', 'VIDP': 'DEL', 'VABB': 'BOM',
    'VOBL': 'BLR', 'VOCI': 'COK',
}


# ─────────────────────────────────────────
#  LIVE SNAPSHOT BUILDER
# ─────────────────────────────────────────

def _build_live_snapshot():
    """Build a single-timestamp snapshot from live APIs.
    Priority: OpenSky (flights) + AviationStack (delays) + Open-Meteo (weather).
    Falls back to simulated data for any API that fails."""
    now = datetime.now()

    records = []
    weather_map = fetch_live_weather_all()
    any_live = False

    for icao, info in AIRPORTS.items():
        # 1. Try OpenSky for flight count
        flight_count = fetch_opensky_live(info)
        if flight_count is not None:
            fph = max(1, flight_count)
            flight_src = 'LIVE_OPENSKY'
            any_live = True
        else:
            fph = np.random.poisson(22)
            flight_src = 'SIMULATED'

        # 2. Try AviationStack for delay data
        iata = ICAO_TO_IATA.get(icao)
        avg_delay = None
        if iata:
            av_data = fetch_aviationstack(iata)
            if av_data:
                # Use AviationStack flight count if OpenSky failed
                if flight_src == 'SIMULATED':
                    fph = av_data['flight_count']
                    flight_src = 'LIVE_AVIATIONSTACK'
                avg_delay = av_data['avg_delay_min']
                any_live = True

        if avg_delay is None:
            avg_delay = round(max(0, np.random.normal(15, 12)), 1)

        # 3. Weather from Open-Meteo (already fetched above)
        weather = weather_map.get(icao)
        if weather:
            wind = weather['wind_speed_kmh']
            vis = weather['visibility_m']
            rain = weather['precipitation_mm']
            cloud = weather['cloud_cover']
            any_live = True
        else:
            wind = np.random.uniform(10, 40)
            vis = np.random.uniform(3000, 10000)
            rain = max(0, np.random.exponential(2))
            cloud = max(0.0, np.random.beta(2, 5))

        active_rwy = max(1, info['runways'] - np.random.choice([0, 0, 0, 1]))

        records.append({
            'timestamp': now,
            'airport_icao': icao,
            'lat': info['lat'],
            'lon': info['lon'],
            'flights_per_hour': fph,
            'avg_delay_min': avg_delay,
            'wind_speed_kmh': round(wind, 1),
            'visibility_m': round(vis),
            'precipitation_mm': round(rain, 2),
            'cloud_cover': round(cloud, 3),
            'active_runways': active_rwy,
            'total_runways': info['runways'],
            'restriction_flag': 0,
            'notam_severity': 0.0,
            'hour_of_day': now.hour,
            'day_of_week': now.weekday(),
            'is_peak_hour': 1 if (6 <= now.hour <= 9 or 17 <= now.hour <= 21) else 0,
            'is_weekend': 1 if now.weekday() >= 5 else 0,
            'data_source': flight_src,
        })
        time.sleep(0.5)

    # FIX #16: only return None if we got absolutely nothing useful
    if not any_live and not records:
        return None

    return pd.DataFrame(records)


# ─────────────────────────────────────────
#  KAGGLE LOADER
# ─────────────────────────────────────────

def _reshape_kaggle(kaggle_path=None):
    """Load Kaggle flight delay CSV and reshape to our schema.
    FIX #18: Map to all 5 airports, not just VOMM.
    FIX #24: Handle SCHEDULED_DEPARTURE float→time conversion properly.
    """
    if kaggle_path is None:
        kaggle_path = os.path.join(BASE_DIR, "data", "flights_2015.csv")

    if not os.path.exists(kaggle_path):
        print(f"[data] Kaggle file not found: {kaggle_path}")
        return None

    cols = ['YEAR', 'MONTH', 'DAY', 'SCHEDULED_DEPARTURE', 'DEPARTURE_DELAY',
            'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']
    df = pd.read_csv(kaggle_path, usecols=cols, nrows=200_000)

    # FIX #24: Properly convert float departure time to hour
    # SCHEDULED_DEPARTURE is stored as float: 830.0 = 08:30, 1430.0 = 14:30
    # Convert to int first, then extract hour
    df['SCHEDULED_DEPARTURE'] = pd.to_numeric(df['SCHEDULED_DEPARTURE'], errors='coerce')
    df = df.dropna(subset=['SCHEDULED_DEPARTURE'])
    df['hour'] = (df['SCHEDULED_DEPARTURE'] // 100).astype(int).clip(0, 23)

    # FIX #18: Map top US airports to all 5 Indian airports (not just VOMM)
    us_to_india = {
        'ATL': 'VABB', 'ORD': 'VIDP', 'DFW': 'VOMM', 'DEN': 'VOBL', 'LAX': 'VOCI',
        'SFO': 'VABB', 'JFK': 'VIDP', 'SEA': 'VOMM', 'CLT': 'VOBL', 'LAS': 'VOCI',
        'PHX': 'VABB', 'MIA': 'VIDP', 'IAH': 'VOMM', 'MCO': 'VOBL', 'EWR': 'VOCI',
        'MSP': 'VABB', 'BOS': 'VIDP', 'DTW': 'VOMM', 'FLL': 'VOBL', 'PHL': 'VOCI',
    }
    df['airport_icao'] = df['ORIGIN_AIRPORT'].map(us_to_india)
    df = df.dropna(subset=['airport_icao'])

    # Build timestamp
    df['timestamp'] = pd.to_datetime(
        df[['YEAR', 'MONTH', 'DAY']].assign(HOUR=df['hour']),
        format='%Y%m%d%H', errors='coerce'
    )
    # If that fails, build manually
    if df['timestamp'].isna().all():
        df['timestamp'] = pd.to_datetime(
            df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str).str.zfill(2) + '-' +
            df['DAY'].astype(str).str.zfill(2) + ' ' + df['hour'].astype(str).str.zfill(2) + ':00:00',
            errors='coerce'
        )
    df = df.dropna(subset=['timestamp'])

    # Aggregate to hourly per airport
    agg = df.groupby(['airport_icao', pd.Grouper(key='timestamp', freq='h')]).agg(
        flights_per_hour=('ORIGIN_AIRPORT', 'count'),
        avg_delay_min=('DEPARTURE_DELAY', 'mean'),
    ).reset_index()

    # Add airport metadata
    for icao, info in AIRPORTS.items():
        mask = agg['airport_icao'] == icao
        agg.loc[mask, 'lat'] = info['lat']
        agg.loc[mask, 'lon'] = info['lon']
        agg.loc[mask, 'active_runways'] = info['runways']
        agg.loc[mask, 'total_runways'] = info['runways']

    # Fill weather with reasonable defaults (Kaggle doesn't have weather)
    agg['wind_speed_kmh'] = np.random.uniform(10, 40, len(agg))
    agg['visibility_m'] = np.random.uniform(3000, 10000, len(agg))
    agg['precipitation_mm'] = np.random.exponential(2, len(agg)).clip(0)
    agg['cloud_cover'] = np.random.beta(2, 5, len(agg)).clip(0, 1)
    agg['restriction_flag'] = 0
    agg['notam_severity'] = 0.0
    agg['avg_delay_min'] = agg['avg_delay_min'].fillna(0).clip(lower=0)

    return agg


# ─────────────────────────────────────────
#  MAIN LOADER
# ─────────────────────────────────────────

def load_data(source="synthetic"):
    """
    Load data based on source:
      - 'synthetic': generate synthetic data
      - 'kaggle': load from Kaggle CSV
      - 'auto' / 'live': try live APIs → Kaggle → synthetic fallback
    FIX #38: All paths use BASE_DIR for portability.
    """
    if source == "synthetic":
        csv_path = os.path.join(BASE_DIR, "data", "synthetic_atc_data.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, parse_dates=['timestamp'])
            print(f"[data] Loaded synthetic CSV: {len(df):,} rows")
            return df
        return generate_synthetic_data(n_days=30)

    elif source == "kaggle":
        df = _reshape_kaggle()
        if df is not None and len(df) > 0:
            return df
        print("[data] Kaggle load failed — falling back to synthetic")
        return generate_synthetic_data(n_days=30)

    elif source in ("auto", "live"):
        # Try live first
        df = _build_live_snapshot()
        if df is not None and len(df) > 0:
            return df
        # Fallback to Kaggle
        df = _reshape_kaggle()
        if df is not None and len(df) > 0:
            return df
        # Final fallback
        print("[data] All live sources failed — falling back to synthetic")
        return generate_synthetic_data(n_days=30)

    else:
        return generate_synthetic_data(n_days=30)
