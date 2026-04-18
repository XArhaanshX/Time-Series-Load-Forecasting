import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

def run_feature_engineering(df, config):
    logger.info("Starting feature engineering stage.")
    
    # Work on a copy to preserve the original load column for reference if needed
    # but the instructions say predicting raw load_t is the goal.
    feat_df = df.copy()
    target_col = config['data']['target_col']
    eps = float(config['features']['epsilon'])
    
    # 1. Calendar Features (Extracted from index)
    feat_df['hour'] = feat_df.index.hour
    feat_df['day_of_week'] = feat_df.index.dayofweek
    feat_df['month'] = feat_df.index.month
    
    # 2. Holiday Schema (Placeholders)
    feat_df['is_weekend'] = feat_df['day_of_week'].isin([5, 6]).astype(int)
    feat_df['is_holiday'] = 0 # Placeholder
    feat_df['is_festival'] = 0 # Placeholder
    
    # 3. Cyclical Encoding
    feat_df['hour_sin'] = np.sin(2 * np.pi * feat_df['hour'] / 24)
    feat_df['hour_cos'] = np.cos(2 * np.pi * feat_df['hour'] / 24)
    feat_df['dow_sin'] = np.sin(2 * np.pi * feat_df['day_of_week'] / 7)
    feat_df['dow_cos'] = np.cos(2 * np.pi * feat_df['day_of_week'] / 7)
    
    # 4. Fourier Features (Capturing periodic cycles)
    # Harmonics for hourly (96 steps) and weekly (672 steps) cycles
    for h in range(1, config['features']['fourier_harmonics'] + 1):
        # Daily cycle (96 intervals)
        feat_df[f'fourier_daily_sin_{h}'] = np.sin(2 * np.pi * h * np.arange(len(feat_df)) / 96)
        feat_df[f'fourier_daily_cos_{h}'] = np.cos(2 * np.pi * h * np.arange(len(feat_df)) / 96)
        # Weekly cycle (672 intervals)
        feat_df[f'fourier_weekly_sin_{h}'] = np.sin(2 * np.pi * h * np.arange(len(feat_df)) / 672)
        feat_df[f'fourier_weekly_cos_{h}'] = np.cos(2 * np.pi * h * np.arange(len(feat_df)) / 672)
        
    # 5. Lagged Load Lags
    for lag in config['features']['seasonal_lags']:
        feat_df[f'lag_{lag}'] = feat_df[target_col].shift(lag)
        
    # 6. Lagged Rolling Statistics
    # rolling_mean_96(t) = aggregate(load[t-96 : t-1])
    for window in config['features']['rolling_windows']:
        # Shift(1) before rolling ensures zero leakage
        shifted_load = feat_df[target_col].shift(1)
        feat_df[f'rolling_mean_{window}'] = shifted_load.rolling(window=window).mean()
        feat_df[f'rolling_std_{window}'] = shifted_load.rolling(window=window).std()
        feat_df[f'rolling_min_{window}'] = shifted_load.rolling(window=window).min()
        feat_df[f'rolling_max_{window}'] = shifted_load.rolling(window=window).max()
        
    # 7. Differences (Momentum)
    # Using shift(1) implicitly by subtracting lagged values
    feat_df['load_diff_1'] = feat_df[target_col].shift(1) - feat_df[target_col].shift(2)
    feat_df['load_diff_96'] = feat_df[target_col].shift(1) - feat_df[target_col].shift(97)
    feat_df['load_diff_672'] = feat_df[target_col].shift(1) - feat_df[target_col].shift(673)
    
    # 8. Ratio Feature with Epsilon Protection
    # load_ratio_96 = load_{t-1} / (load_{t-96} + eps)
    feat_df['load_ratio_96'] = feat_df[target_col].shift(1) / (feat_df[target_col].shift(96) + eps)
    
    # Drop rows with NaNs created by lags/rolling
    # Max lag is 673 for diff_672 or 672 for lags
    feat_df = feat_df.dropna()
    
    logger.info(f"Feature engineering complete. Generated {len(feat_df.columns)} columns.")
    
    # Save to features directory
    os.makedirs(config['paths']['feature_data'], exist_ok=True)
    feat_df.to_parquet(os.path.join(config['paths']['feature_data'], 'engineered_features.parquet'))
    
    return feat_df
