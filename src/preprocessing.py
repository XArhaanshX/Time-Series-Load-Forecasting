import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

def apply_winsorization(series, threshold=4.0):
    """
    Apply winsorization by clipping values beyond N standard deviations.
    """
    mu = series.mean()
    sigma = series.std()
    
    lower_bound = mu - threshold * sigma
    upper_bound = mu + threshold * sigma
    
    logger.info(f"Winsorization bounds: [{lower_bound:.2f}, {upper_bound:.2f}] (Threshold: {threshold} sigma)")
    return series.clip(lower=lower_bound, upper=upper_bound)

def run_preprocessing(df, config):
    logger.info("Starting preprocessing stage.")
    
    # Ensure datetime index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    
    # 1. Enforce strict 15-minute frequency
    initial_len = len(df)
    df = df.asfreq(config['data']['frequency'])
    after_freq_len = len(df)
    missing_intervals = df['load_MW'].isna().sum()
    
    logger.info(f"Frequency enforcement: {missing_intervals} missing intervals identified.")
    
    # 2. Forward fill short gaps
    # For short term forecasting, ffilling small gaps is acceptable
    df['load_MW'] = df['load_MW'].ffill(limit=4) # Limit 1 hour
    
    # 3. Winsorization (Z-score clipping at 4 standard deviations)
    df['load_MW'] = apply_winsorization(df['load_MW'], threshold=config['analysis']['z_score_threshold'])
    
    # 4. Final safety check for any remaining NaNs (e.g. at start or long gaps)
    if df['load_MW'].isna().any():
        logger.warning("Remaining NaNs after preprocessing. Using linear interpolation as fallback.")
        df['load_MW'] = df['load_MW'].interpolate(method='linear').bfill()

    # Save cleaned series to parquet
    output_path = os.path.join(config['paths']['processed_data'], 'cleaned_load.parquet')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path)
    
    logger.info(f"Preprocessing complete. Cleaned data saved to {output_path}")
    
    report = {
        "initial_rows": initial_len,
        "final_rows": len(df),
        "missing_intervals": int(missing_intervals),
        "status": "Success"
    }
    
    return df, report
