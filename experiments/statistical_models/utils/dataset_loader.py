import pandas as pd
import os
from .. import config

def load_statistical_data():
    """
    Loads canonical dataset, extracts target column, and applies truncation.
    Returns:
        train_series (pd.Series): Truncated training series.
        test_series (pd.Series): Full test series.
    """
    print(f"Loading datasets from {config.DATASET_PATH}...")
    
    # Load parquet files
    train_df = pd.read_parquet(config.TRAIN_PATH)
    test_df = pd.read_parquet(config.TEST_PATH)
    
    # Extract only the target series
    if config.TARGET_COLUMN not in train_df.columns:
        raise ValueError(f"Target column '{config.TARGET_COLUMN}' not found in train dataset.")
    
    # Statistical models operate on the raw series. 
    # Timestamps are typically preserved in the index.
    train_series = train_df[config.TARGET_COLUMN]
    test_series = test_df[config.TARGET_COLUMN]
    
    # Truncate training set for computational feasibility
    actual_full_size = len(train_series)
    train_series = train_series.tail(config.MAX_TRAIN_POINTS)
    
    print(f"Training set truncated from {actual_full_size:,} to {len(train_series):,} samples.")
    print(f"Test set size: {len(test_series):,} samples.")
    
    return train_series, test_series
