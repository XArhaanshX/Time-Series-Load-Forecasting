import os
import pandas as pd
import numpy as np
import joblib
import json
import torch
from . import config

def load_lgbm_model():
    print(f"Loading LightGBM model from {config.LGBM_MODEL_PATH}...")
    model = joblib.load(config.LGBM_MODEL_PATH)
    return model

def compute_residuals(model):
    # Load canonical datasets to get the features used for LGBM
    print(f"Loading canonical datasets for LGBM predictions...")
    train_df = pd.read_parquet(config.LGBM_TRAIN_DATA)
    test_df = pd.read_parquet(config.LGBM_TEST_DATA)
    
    # Concatenate to get all possible points
    # (Optional, depends on if dataset_v3 covers more than these two)
    full_df = pd.concat([train_df, test_df], axis=0)
    
    if full_df.index.name != 'timestamp' and 'timestamp' in full_df.columns:
        full_df = full_df.set_index('timestamp')
    elif full_df.index.name != 'timestamp':
        # Many datasets have it as index
        full_df.index.name = 'timestamp'
        
    full_df = full_df.sort_index()
    
    # Enforce feature consistency using model's internal feature names
    print("Enforcing feature schema consistency for LightGBM...")
    feature_names = model.feature_name_
    X = full_df[feature_names]
    
    # Generate predictions
    print(f"Generating LightGBM predictions for {len(X)} points...")
    y_base = model.predict(X)
    
    # Compute residuals: actual - predicted
    actuals = full_df["load_MW"]
    residuals = actuals - y_base
    
    # Store in a Series for timestamp-based lookup
    residual_series = pd.Series(residuals, index=full_df.index)
    
    return residual_series, pd.Series(y_base, index=full_df.index)

def align_residuals(residual_series, mode="train"):
    ts_path = os.path.join(config.V3_DATASET_PATH, f"timestamps_{mode}.npy")
    print(f"Loading {mode} timestamps from {ts_path}...")
    ts = np.load(ts_path, allow_pickle=True)
    
    # Create residual targets using timestamp-based lookup
    ts_idx = pd.to_datetime(ts)
    
    try:
        aligned = residual_series.loc[ts_idx]
        residual_targets = aligned.values
    except KeyError:
        # Some timestamps might be missing if they were filtered during canonical dataset creation
        # but present in dataset_v3? Unlikely if they use the same source.
        # Fallback: reindex with ffill or handle missing
        print(f"WARNING: Some timestamps in {mode} set not found in residual series. Reindexing...")
        aligned = residual_series.reindex(ts_idx).ffill().bfill()
        residual_targets = aligned.values
        
    return residual_targets, ts

def log_diagnostics(residuals, mode="train"):
    res_mean = np.mean(residuals)
    res_std = np.std(residuals)
    res_max_abs = np.max(np.abs(residuals))
    res_p95 = np.percentile(np.abs(residuals), 95)
    
    print(f"\n--- Residual Diagnostics ({mode}) ---")
    print(f"Mean: {res_mean:.4f}")
    print(f"Std: {res_std:.4f}")
    print(f"Max Abs: {res_max_abs:.4f}")
    print(f"95th Percentile Abs: {res_p95:.4f}")
    print("--------------------------------------")
    
    return {
        "mean": res_mean,
        "std": res_std,
        "max_abs": res_max_abs,
        "p95_abs": res_p95
    }

def check_leakage(y_base, residuals):
    correlation = np.corrcoef(y_base, residuals)[0, 1]
    print(f"Correlation between LightGBM predictions and residuals: {correlation:.4f}")
    if abs(correlation) > 0.5:
        print("WARNING: Unusually high correlation detected. Potential data leakage or bias!")
    return correlation

def build_residual_targets():
    model = load_lgbm_model()
    
    residual_series, y_base_series = compute_residuals(model)
    
    res_train, ts_train = align_residuals(residual_series, mode="train")
    res_test, ts_test = align_residuals(residual_series, mode="test")
    
    # Final Integrity Checks
    assert np.isfinite(res_train).all(), "NaN or Inf detected in train residuals!"
    assert np.isfinite(res_test).all(), "NaN or Inf detected in test residuals!"
    
    # Chronological check
    assert pd.to_datetime(ts_train).is_monotonic_increasing, "Train timestamps not chronological!"
    assert pd.to_datetime(ts_test).is_monotonic_increasing, "Test timestamps not chronological!"
    
    # Diagnostics
    log_diagnostics(res_train, mode="train")
    log_diagnostics(res_test, mode="test")
    
    # Leakage check (on train set alignment)
    y_base_train = y_base_series.loc[pd.to_datetime(ts_train)].values
    check_leakage(y_base_train, res_train)
    
    return res_train, res_test

if __name__ == "__main__":
    res_train, res_test = build_residual_targets()
    print("Residual targets successfully built and aligned.")
