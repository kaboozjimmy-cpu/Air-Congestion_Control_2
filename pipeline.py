"""
pipeline.py
Feature engineering, label creation, and train/test split.
FIX #15: build_sequences no longer calls run_pipeline() internally.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURE_COLS = [
    'flights_per_hour',
    'avg_delay_min',
    'weather_severity',
    'runway_util_ratio',
    'restriction_flag',
    'notam_severity',
    'hour_of_day',
    'day_of_week',
    'is_peak_hour',
    'is_weekend',
    'cloud_cover',
]

LABEL_MAP = {0: 'LOW', 1: 'MODERATE', 2: 'CRITICAL'}
COLOR_MAP = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}


# ─────────────────────────────────────────
#  WEATHER SEVERITY SCORE
# ─────────────────────────────────────────

def compute_weather_severity(row):
    wind_score = min(row['wind_speed_kmh'] / 80.0, 1.0)
    vis_score  = 1.0 - min(row['visibility_m'] / 10000.0, 1.0)
    rain_score = min(row['precipitation_mm'] / 20.0, 1.0)
    cloud_score = max(0.0, min(1.0, row.get('cloud_cover', 0.2)))
    return round(
        0.35 * wind_score +
        0.40 * vis_score  +
        0.15 * rain_score +
        0.10 * cloud_score,
        3
    )


# ─────────────────────────────────────────
#  RUNWAY UTILIZATION
# ─────────────────────────────────────────

def compute_runway_util(row):
    denom = row['active_runways'] * 18
    if denom == 0:
        return 1.0
    return round(min(1.0, row['flights_per_hour'] / denom), 3)


# ─────────────────────────────────────────
#  CONGESTION LABEL
# ─────────────────────────────────────────

def assign_congestion_label(row):
    fph   = row['flights_per_hour']
    delay = row['avg_delay_min']
    wx    = row['weather_severity']
    rwy   = row['runway_util_ratio']
    notam = row['restriction_flag']

    if (fph > 35 or delay > 30 or wx > 0.70 or
            (rwy >= 1.0 and fph > 25) or notam == 1):
        return 2
    if fph > 20 or delay > 15 or wx > 0.40 or rwy > 0.75:
        return 1
    return 0


# ─────────────────────────────────────────
#  FULL PIPELINE
# ─────────────────────────────────────────

def run_pipeline(df):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Derived features
    df['weather_severity']  = df.apply(compute_weather_severity, axis=1)
    df['runway_util_ratio'] = df.apply(compute_runway_util, axis=1)

    # Temporal
    df['hour_of_day']  = df['timestamp'].dt.hour
    df['day_of_week']  = df['timestamp'].dt.dayofweek
    df['is_peak_hour'] = df['hour_of_day'].apply(
        lambda h: 1 if (6 <= h <= 9 or 17 <= h <= 21) else 0
    )
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Label
    df['congestion_label'] = df.apply(assign_congestion_label, axis=1)
    df['congestion_name']  = df['congestion_label'].map(LABEL_MAP)
    df['congestion_color'] = df['congestion_label'].map(COLOR_MAP)

    return df


# ─────────────────────────────────────────
#  TRAIN / TEST SPLIT + SCALING
# ─────────────────────────────────────────

def prepare_xy(df, test_size=0.2):
    df = run_pipeline(df)

    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(0)
    y = df['congestion_label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))

    print(f"[pipeline] Train: {len(X_train):,}  Test: {len(X_test):,}")
    print("[pipeline] Label distribution (train):")
    for k, v in y_train.value_counts(normalize=True).sort_index().items():
        print(f"  {LABEL_MAP[k]:10s}: {v*100:.1f}%")

    return X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, scaler, available


# ─────────────────────────────────────────
#  LSTM SEQUENCE BUILDER
# ─────────────────────────────────────────

def build_sequences(df, airport_icao, window=12):
    """Build LSTM sequences.
    FIX #15: Does NOT call run_pipeline() again.
    Expects df that already has pipeline features computed.
    This preserves sidebar overrides (weather slider, runway slider).
    FIX #25: Applies StandardScaler to LSTM features.
    """
    adf = df[df['airport_icao'] == airport_icao].sort_values('timestamp')
    available = [c for c in FEATURE_COLS if c in adf.columns]
    vals_raw = adf[available].fillna(0).values

    # FIX #25: Scale LSTM input features (previously raw unscaled values)
    scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        try:
            vals = scaler.transform(vals_raw)
        except ValueError:
            # Feature count mismatch — fall back to raw
            vals = vals_raw
    else:
        vals = vals_raw

    labels = adf['congestion_label'].values if 'congestion_label' in adf.columns else np.zeros(len(adf))

    X_seq, y_seq = [], []
    for i in range(window, len(vals)):
        X_seq.append(vals[i-window:i])
        y_seq.append(labels[i])

    return np.array(X_seq) if X_seq else np.array([]), np.array(y_seq) if y_seq else np.array([])


# ─────────────────────────────────────────
#  VALIDATE LABEL BALANCE
# ─────────────────────────────────────────

def validate_labels(df):
    if 'congestion_label' not in df.columns:
        df = run_pipeline(df)
    dist = df['congestion_label'].value_counts(normalize=True).sort_index()
    print("\n=== Label distribution ===")
    for k, v in dist.items():
        print(f"  {LABEL_MAP[k]:10s}: {v*100:.1f}%")
    if dist.get(2, 0) < 0.05:
        print("\n  WARNING: CRITICAL < 5% — use class_weight in XGBoost")
    return dist
