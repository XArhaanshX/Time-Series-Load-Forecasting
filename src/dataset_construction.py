import pandas as pd
import numpy as np
import os
import logging
from numpy.lib.stride_tricks import sliding_window_view

logger = logging.getLogger(__name__)

def create_sliding_window_fast(df, selected_features, target_col, window_size):
    """
    Optimized version of sliding window construction using numpy strides.
    """
    logger.info(f"Creating sliding window dataset with window_size={window_size} (Optimized)")
    
    # Isolate relevant data
    X_raw = df[selected_features].values
    y_raw = df[target_col].values
    
    # Create windows for X using sliding_window_view
    # Input X_raw shape: (N, num_features)
    # Output windows shape: (N - window_size + 1, window_size, num_features)
    windows = sliding_window_view(X_raw, window_shape=(window_size, X_raw.shape[1]), axis=(0, 1)).squeeze()
    
    # We want features from t-window_size to t-1
    # So we take windows starting from 0 up to (len-1 - window_size + 1)
    # And the target is y[window_size:]
    
    # Actually, sliding_window_view with window_size=W gives:
    # index 0: [0:W]
    # index 1: [1:W+1]
    # ...
    # index N-W: [N-W : N]
    
    # If the target for [0:W] is at index W:
    # Then we use all windows except the very last one if it goes up to N.
    # Wait, if we have N observations [0...N-1].
    # Window [0...W-1] (length W) -> Target y[W]
    # Window [N-W-1 ... N-2] -> Target y[N-1]
    
    # So we need windows from index 0 to index N-W-1.
    X_windows = windows[:-1] # Remove the last window which ends at N-1 (since we need its target at N, which doesn't exist)
    # Actually, let's check:
    # total length N=245k. W=96.
    # X_raw indices: 0 to N-1.
    # windows shape: (N-W+1, W, num_features)
    # Target y[W] is for windows[0].
    # Target y[N-1] is for windows[N-W-1].
    
    X_final = X_windows.reshape(len(X_windows), -1) # Flatten each window
    y_final = y_raw[window_size:]
    timestamps = df.index[window_size:]
    
    # Create column names: feature_name_t-k
    col_names = []
    for lag in range(window_size, 0, -1):
        for feat in selected_features:
            col_names.append(f"{feat}_t-{lag}")
            
    final_df = pd.DataFrame(X_final, columns=col_names, index=timestamps)
    final_df[target_col] = y_final
    
    return final_df

def run_dataset_construction(df, selected_features, optimal_lag, config):
    logger.info("Starting dataset construction (sliding window).")
    
    target_col = config['data']['target_col']
    
    final_matrix = create_sliding_window_fast(df, selected_features, target_col, optimal_lag)
    
    # Verification of shape
    expected_rows = len(df) - optimal_lag
    if len(final_matrix) != expected_rows:
        logger.warning(f"Row count mismatch. Expected: {expected_rows}, Actual: {len(final_matrix)}")
        
    # Save final matrix
    output_path = os.path.join(config['paths']['feature_data'], 'final_feature_matrix.parquet')
    final_matrix.to_parquet(output_path)
    
    logger.info(f"Final feature matrix saved to {output_path}. Shape: {final_matrix.shape}")
    return final_matrix
