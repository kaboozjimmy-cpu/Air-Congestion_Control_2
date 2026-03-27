"""
train.py
Trains XGBoost classifier + LSTM forecaster, saves models + SHAP values.
Run: python train.py

Fixes applied: #1 #14 #17 #30 #31 #32 #35
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from utils.data_loader import load_data, AIRPORTS
from pipeline import prepare_xy, build_sequences, run_pipeline, FEATURE_COLS, LABEL_MAP


# ─────────────────────────────────────────
#  1. LOAD DATA
# ─────────────────────────────────────────

print("\n" + "="*50)
print("  ATC Congestion Prediction — Training")
print("="*50)

source = os.getenv("DATA_SOURCE", "synthetic")
print(f"\n[train] Data source: {source}")
df = load_data(source=source)
print(f"[train] Loaded {len(df):,} rows")


# ─────────────────────────────────────────
#  2. FEATURE ENGINEERING + SPLIT
# ─────────────────────────────────────────

# FIX #14: prepare_xy now returns `available` feature list
X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, scaler, available_features = prepare_xy(df)


# ─────────────────────────────────────────
#  3. XGBOOST CLASSIFIER
# ─────────────────────────────────────────

print("\n[xgb] Training XGBoost classifier...")
from xgboost import XGBClassifier
# FIX #31: only import what we use
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight

# FIX #17: Use sample_weight for multi-class instead of scale_pos_weight
sample_weights = compute_sample_weight('balanced', y_train)

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    # FIX #1: removed use_label_encoder=False (removed in XGBoost 2.0+)
    # FIX #17: removed scale_pos_weight=2 (binary-only param)
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1,
)
xgb_model.fit(
    X_train_sc, y_train,
    sample_weight=sample_weights,
    eval_set=[(X_test_sc, y_test)],
    verbose=False,
)

y_pred = xgb_model.predict(X_test_sc)
print("\n[xgb] Classification report:")
n_classes = len(np.unique(y_train))
label_names = [LABEL_MAP[i] for i in range(n_classes)]
print(classification_report(y_test, y_pred, target_names=label_names,
                            labels=list(range(n_classes))))

models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)
joblib.dump(xgb_model, os.path.join(models_dir, "xgboost_model.pkl"))
print("[xgb] Saved → models/xgboost_model.pkl")


# ─────────────────────────────────────────
#  4. SHAP VALUES
# ─────────────────────────────────────────

print("\n[shap] Computing SHAP values...")
import shap

# FIX #14: Use available_features from prepare_xy (post-pipeline),
# not from raw df which lacks computed features
explainer   = shap.TreeExplainer(xgb_model)
shap_sample = X_test_sc[:200]
shap_values = explainer.shap_values(shap_sample)

shap_data = {
    # FIX #14: feature names now match the actual scaled features
    "feature_names": available_features,
    "shap_values":   shap_values.tolist() if hasattr(shap_values, 'tolist')
                     else [sv.tolist() for sv in shap_values],
    "X_sample":      shap_sample.tolist(),
}
with open(os.path.join(models_dir, "shap_data.json"), "w") as f:
    json.dump(shap_data, f)
print("[shap] Saved → models/shap_data.json")


# ─────────────────────────────────────────
#  5. LSTM FORECASTER
# ─────────────────────────────────────────

print("\n[lstm] Training LSTM forecaster...")
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    WINDOW = 12

    # FIX #30: Build sequences per airport, split chronologically within each,
    # THEN stack. This prevents train/test temporal bleed.
    df_processed = run_pipeline(df)

    all_Xtr, all_ytr = [], []
    all_Xte, all_yte = [], []

    for icao in AIRPORTS.keys():
        Xs, ys = build_sequences(df_processed, icao, window=WINDOW)
        if len(Xs) < 20:
            continue
        # Chronological split per airport
        split = int(len(Xs) * 0.8)
        all_Xtr.append(Xs[:split])
        all_ytr.append(ys[:split])
        all_Xte.append(Xs[split:])
        all_yte.append(ys[split:])

    if all_Xtr:
        X_lstm_tr = np.vstack(all_Xtr)
        y_lstm_tr = np.concatenate(all_ytr)
        X_lstm_te = np.vstack(all_Xte)
        y_lstm_te = np.concatenate(all_yte)

        n_feat = X_lstm_tr.shape[2]

        model_lstm = Sequential([
            LSTM(64, return_sequences=True, input_shape=(WINDOW, n_feat)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(3, activation='softmax'),
        ])
        model_lstm.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        es = EarlyStopping(patience=5, restore_best_weights=True)
        model_lstm.fit(
            X_lstm_tr, y_lstm_tr,
            epochs=30,
            batch_size=64,
            validation_data=(X_lstm_te, y_lstm_te),
            callbacks=[es],
            verbose=0,
        )

        loss, acc = model_lstm.evaluate(X_lstm_te, y_lstm_te, verbose=0)
        print(f"[lstm] Test accuracy: {acc*100:.1f}%")

        # FIX #35: Save as .keras (modern format) instead of .h5 (deprecated)
        lstm_path = os.path.join(models_dir, "lstm_model.keras")
        model_lstm.save(lstm_path)
        print(f"[lstm] Saved → {lstm_path}")
    else:
        print("[lstm] Not enough sequence data — skipping")

except Exception as e:
    print(f"[lstm] Training failed ({e}) — skipping LSTM")


# ─────────────────────────────────────────
#  6. SAVE METADATA
# ─────────────────────────────────────────

df_for_meta = run_pipeline(df) if 'congestion_label' not in df.columns else df
# FIX #32: Convert int keys to string keys explicitly
label_dist = {str(k): int(v) for k, v in
              df_for_meta['congestion_label'].value_counts().to_dict().items()}

meta = {
    "trained_at":   pd.Timestamp.now().isoformat(),
    "data_source":  source,
    "n_samples":    len(df),
    "features":     available_features,
    "label_dist":   label_dist,
}
with open(os.path.join(models_dir, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2, default=str)

print("\n[train] Done. All models saved to /models/")
print("="*50)
