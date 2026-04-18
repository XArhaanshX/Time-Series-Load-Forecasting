import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# --- Constants ---
WINDOW_SIZE = 96
TARGET_COL = "load_MW"
SOURCE_PATH = "data/features/engineered_features.parquet"
OUTPUT_DIR = "experiments/deep_learning/dataset"
VERSION = "v2_sequence_dataset"

def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_sequences(features, targets, timestamps):
    """
    Creates 3D sequence tensors (Samples, Time_Steps, Features).
    Enforces the rule: X_t uses records [t-96 ... t-1] to predict y_t.
    """
    X, y, ts = [], [], []
    
    # Range: from WINDOW_SIZE to the end
    # Example: if WINDOW_SIZE=96, first index is 96.
    # X[0] will be features[0:96], y[0] will be targets[96].
    for i in range(len(features) - WINDOW_SIZE):
        X.append(features[i : i + WINDOW_SIZE])
        y.append(targets[i + WINDOW_SIZE])
        ts.append(timestamps[i + WINDOW_SIZE])
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(ts)

def validate_leakage(X_timestamps, y_timestamps):
    """
    Ensures max(input_timestamp) < target_timestamp for every sample.
    X_timestamps should have shape (N, WINDOW_SIZE)
    y_timestamps should have shape (N)
    """
    # For every sample i, we check if the last timestamp in the window is before the target
    # Since we used slicing [i:i+96], the last timestamp in X[i] is at index i+95.
    # y[i] is at index i+96.
    # We can perform a vectorized check.
    
    # We need the full timestamp matrix for X to be 100% sure, 
    # but since we processed them chronologically, checking the last element is sufficient.
    last_input_ts = X_timestamps[:, -1]
    violations = np.sum(last_input_ts >= y_timestamps)
    
    if violations > 0:
        raise ValueError(f"CRITICAL: Temporal leakage detected in {violations} samples!")
    
    print("SUCCESS: Temporal integrity verified: max(input_time) < target_time for all samples.")

def main():
    ensure_dirs()
    print(f"--- Starting {VERSION} Construction ---")
    
    # 1. Load and Sort
    print(f"Loading source data from {SOURCE_PATH}...")
    df = pd.read_parquet(SOURCE_PATH)
    if df.index.name == 'timestamp':
        df = df.reset_index()
    
    print("Sorting by timestamp...")
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # 2. Feature Selection
    # Target is load_MW. 
    # Features exclude target and raw integer time variables if cyclical exist.
    all_cols = df.columns.tolist()
    
    # Identify cyclical features to see if we should drop raw ones
    cyclical_prefixes = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
    has_cyclical = all(any(c.startswith(p) for c in all_cols) for p in cyclical_prefixes)
    
    exclude_cols = [TARGET_COL, "timestamp"]
    if has_cyclical:
        exclude_cols.extend(["hour", "day_of_week"])
    
    feature_cols = [c for c in all_cols if c not in exclude_cols]
    feature_cols.sort() # Deterministic ordering
    
    print(f"Detected {len(feature_cols)} features for sequence construction.")
    print(f"Feature list: {feature_cols}")
    
    # 3. Chronological Split
    # Last year (35040 samples) as test
    test_size = 35040
    train_df = df.iloc[:-test_size].copy()
    test_df = df.iloc[-test_size:].copy()
    
    # 4. Scaling
    print("Fitting StandardScaler on training features...")
    scaler = StandardScaler()
    
    # Fit on training features only
    scaler.fit(train_df[feature_cols])
    
    # Transform both
    train_features_scaled = scaler.transform(train_df[feature_cols])
    test_features_scaled = scaler.transform(test_df[feature_cols])
    
    train_targets = train_df[TARGET_COL].values
    test_targets = test_df[TARGET_COL].values
    
    train_timestamps = train_df["timestamp"].values
    test_timestamps = test_df["timestamp"].values
    
    # 5. Sequence Construction (Independent within partitions)
    print("Constructing training sequences...")
    X_train, y_train, ts_train = create_sequences(train_features_scaled, train_targets, train_timestamps)
    
    print("Constructing testing sequences...")
    X_test, y_test, ts_test = create_sequences(test_features_scaled, test_targets, test_timestamps)
    
    # 6. Leakage Validation
    # Build a full timestamp window for X to satisfy the strict verification requirement
    # For training
    print("Performing strict leakage validation...")
    X_ts_train = []
    for i in range(len(train_timestamps) - WINDOW_SIZE):
        X_ts_train.append(train_timestamps[i : i + WINDOW_SIZE])
    X_ts_train = np.array(X_ts_train)
    validate_leakage(X_ts_train, ts_train)
    
    # 7. Diagnostics
    print("\n--- Dataset Summary ---")
    print(f"Total Source Rows: {len(df):,}")
    print(f"Sequence Length: {WINDOW_SIZE}")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Target scale check
    print(f"Target mean (Train): {y_train.mean():.2f} MW")
    if y_train.mean() < 10.0: # Sanity check for normalization
        print("WARNING: y_train mean is very low. Possible accidental normalization!")
    else:
        print("SUCCESS: Target variable scaling verified (MW scale preserved).")
        
    # NaN check
    for name, arr in [("X_train", X_train), ("X_test", X_test), ("y_train", y_train), ("y_test", y_test)]:
        if np.isnan(arr).any():
            print(f"CRITICAL: NaNs detected in {name}!")
        else:
            print(f"SUCCESS: No NaNs in {name}")

    # 8. Serialization
    print(f"\nSaving artifacts to {OUTPUT_DIR}...")
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)
    np.save(os.path.join(OUTPUT_DIR, "timestamps_train.npy"), ts_train)
    np.save(os.path.join(OUTPUT_DIR, "timestamps_test.npy"), ts_test)
    
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
    
    with open(os.path.join(OUTPUT_DIR, "feature_list.json"), "w") as f:
        json.dump(feature_cols, f, indent=4)
        
    summary = {
        "dataset_version": VERSION,
        "num_train_samples": int(X_train.shape[0]),
        "num_test_samples": int(X_test.shape[0]),
        "sequence_length": WINDOW_SIZE,
        "num_features": int(X_train.shape[2]),
        "feature_names": feature_cols,
        "scaler_type": "StandardScaler",
        "tensor_dtype": str(X_train.dtype),
        "timestamp_generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(OUTPUT_DIR, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"--- Dataset {VERSION} Builder Complete ---")

if __name__ == "__main__":
    main()
