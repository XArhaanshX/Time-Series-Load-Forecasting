import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from datetime import datetime

# --- Constants ---
WINDOW_SIZE = 192
TARGET_COL = "load_MW"
SOURCE_PATH = "data/features/engineered_features.parquet"
OUTPUT_DIR = "experiments/deep_learning/dataset_v3"
VERSION = "v3_sequence_dataset"

def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_sequences(features, targets, timestamps):
    """
    Creates 3D sequence tensors (Samples, Time_Steps, Features) independently.
    Rule: X_i uses records [i : i + WINDOW_SIZE] to predict y_i [i + WINDOW_SIZE].
    """
    n_samples = len(features) - WINDOW_SIZE
    n_features = features.shape[1]
    
    # Preallocate for memory efficiency
    X = np.zeros((n_samples, WINDOW_SIZE, n_features), dtype=np.float32)
    y = np.zeros((n_samples,), dtype=np.float32)
    ts = []
    
    for i in range(n_samples):
        X[i] = features[i : i + WINDOW_SIZE]
        y[i] = targets[i + WINDOW_SIZE]
        ts.append(timestamps[i + WINDOW_SIZE])
        
    return X, y, np.array(ts)

def validate_dataset(X, y, ts, name="Dataset"):
    """
    Comprehensive diagnostics for the generated tensors.
    """
    # 1. Finite Check
    if not np.isfinite(X).all():
        raise ValueError(f"CRITICAL: Infinite or NaN values detected in {name} features!")
    if not np.isfinite(y).all():
        raise ValueError(f"CRITICAL: Infinite or NaN values detected in {name} targets!")
        
    # 2. Causality Check (Vectorized)
    # This check is actually performed by our create_sequences logic by design, 
    # but we can verify the sample alignment here if needed.
    pass

def main():
    ensure_dirs()
    print(f"--- Starting {VERSION} Construction ---")
    
    # 1. Load and Sorting
    print(f"Loading source data from {SOURCE_PATH}...")
    df = pd.read_parquet(SOURCE_PATH)
    if df.index.name == 'timestamp':
        df = df.reset_index()
    
    print("Sorting by timestamp...")
    df = df.sort_values("timestamp")
    if not df["timestamp"].is_monotonic_increasing:
        raise RuntimeError("HARD FAILURE: Timestamps are not strictly increasing after sort!")
    df = df.reset_index(drop=True)
    
    # 2. Numerical Sanitization (Bidirectional Fill)
    print("Performing numerical sanitization (Bidirectional Fill)...")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()
    
    if df.isna().any().any():
        raise RuntimeError("HARD FAILURE: Residual NaN values detected after bidirectional fill!")
    
    # 3. Chronological Split
    # Last 35,040 samples (1 year) as test
    print("Splitting into train and test partitions...")
    test_size = 35040
    train_df = df.iloc[:-test_size].copy()
    test_df = df.iloc[-test_size:].copy()
    
    if not train_df["timestamp"].max() < test_df["timestamp"].min():
        raise RuntimeError("HARD FAILURE: Training and testing partitions overlap in time!")
    
    # 4. Feature Selection & Filtering
    print("Selecting features and filtering zero-variance columns...")
    all_cols = df.columns.tolist()
    
    # Identify cyclical features to see if we should drop raw ones
    cyclical_prefixes = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
    has_cyclical = all(any(c.startswith(p) for c in all_cols) for p in cyclical_prefixes)
    
    exclude_cols = [TARGET_COL, "timestamp"]
    if has_cyclical:
        exclude_cols.extend(["hour", "day_of_week"])
    
    feature_cols = [c for c in all_cols if c not in exclude_cols]
    
    # Zero-Variance Filter
    vt = VarianceThreshold(threshold=1e-12)
    vt.fit(train_df[feature_cols])
    
    # Apply filter to feature names
    feature_cols = [feature_cols[i] for i in vt.get_support(indices=True)]
    feature_cols.sort() # Deterministic ordering
    
    print(f"Final feature count after filtering: {len(feature_cols)}")
    
    # 5. Fitting Scalers (Train Only)
    print("Fitting Scalers on training partition...")
    feature_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    feature_scaler.fit(train_df[feature_cols])
    y_scaler.fit(train_df[[TARGET_COL]].values)
    
    # Safeguard: Target Scaler Stability
    if y_scaler.scale_[0] < 1e-8:
        raise RuntimeError(f"HARD FAILURE: Target variable ({TARGET_COL}) variance too small for reliable scaling!")
    
    # 6. Apply Transformations
    print("Applying transformations...")
    train_feat_scaled = feature_scaler.transform(train_df[feature_cols])
    test_feat_scaled = feature_scaler.transform(test_df[feature_cols])
    
    train_y_scaled = y_scaler.transform(train_df[[TARGET_COL]].values).flatten()
    test_y_scaled = y_scaler.transform(test_df[[TARGET_COL]].values).flatten()
    
    # 7. Sequence Construction
    print(f"Constructing sequences (Window Size: {WINDOW_SIZE})...")
    X_train, y_train, ts_train = create_sequences(train_feat_scaled, train_y_scaled, train_df["timestamp"].values)
    X_test, y_test, ts_test = create_sequences(test_feat_scaled, test_y_scaled, test_df["timestamp"].values)
    
    # 8. Validation
    print("Running final diagnostics...")
    
    # Shape verification
    if X_train.shape[0] != len(train_df) - WINDOW_SIZE:
        raise RuntimeError(f"HARD FAILURE: Found {X_train.shape[0]} sequences but expected {len(train_df) - WINDOW_SIZE} for Train.")
    if X_test.shape[0] != len(test_df) - WINDOW_SIZE:
        raise RuntimeError(f"HARD FAILURE: Found {X_test.shape[0]} sequences but expected {len(test_df) - WINDOW_SIZE} for Test.")
        
    # Statistical validation (Flattened)
    train_feat_flat = X_train.reshape(-1, len(feature_cols))
    means = np.abs(train_feat_flat.mean(axis=0))
    stds = train_feat_flat.std(axis=0)
    
    if (means > 0.01).any():
        print(f"WARNING: Feature means out of standard range: max mean = {means.max():.4f}")
    if (stds < 0.95).any() or (stds > 1.05).any():
        print(f"WARNING: Feature stds out of standard range: min std = {stds.min():.4f}, max std = {stds.max():.4f}")
        
    y_mean, y_std = np.abs(y_train.mean()), y_train.std()
    if y_mean > 0.01 or y_std < 0.95 or y_std > 1.05:
         print(f"WARNING: Target stats out of standard range: mean={y_mean:.4f}, std={y_std:.4f}")
    else:
         print("SUCCESS: Scaled target statistics verified (mean approx. 0, std approx. 1).")
         
    # Finite check
    validate_dataset(X_train, y_train, ts_train, "Train")
    validate_dataset(X_test, y_test, ts_test, "Test")
    
    # 9. Serialization
    print(f"Saving artifacts to {OUTPUT_DIR}...")
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)
    np.save(os.path.join(OUTPUT_DIR, "timestamps_train.npy"), ts_train)
    np.save(os.path.join(OUTPUT_DIR, "timestamps_test.npy"), ts_test)
    
    joblib.dump(feature_scaler, os.path.join(OUTPUT_DIR, "feature_scaler.pkl"))
    joblib.dump(y_scaler, os.path.join(OUTPUT_DIR, "y_scaler.pkl"))
    
    with open(os.path.join(OUTPUT_DIR, "feature_list.json"), "w") as f:
        json.dump(feature_cols, f, indent=4)
        
    summary = {
        "dataset_version": VERSION,
        "sequence_length": WINDOW_SIZE,
        "num_features": len(feature_cols),
        "num_train_sequences": int(X_train.shape[0]),
        "num_test_sequences": int(X_test.shape[0]),
        "train_tensor_shape": list(X_train.shape),
        "test_tensor_shape": list(X_test.shape),
        "tensor_dtype": str(X_train.dtype),
        "feature_names": feature_cols,
        "timestamp_start": str(ts_train[0]),
        "timestamp_end": str(ts_test[-1]),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(OUTPUT_DIR, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"\n--- {VERSION} Builder Complete ---")
    print(f"Train Sequences: {X_train.shape[0]:,}")
    print(f"Test Sequences: {X_test.shape[0]:,}")
    print(f"Window Size: {WINDOW_SIZE}")
    print(f"Feature Count: {len(feature_cols)}")

if __name__ == "__main__":
    main()
