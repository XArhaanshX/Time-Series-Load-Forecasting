import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import pacf
import os
import json
import logging

logger = logging.getLogger(__name__)

def run_lag_analysis(df, config):
    logger.info("Starting lag analysis using PACF.")
    nlags = config['analysis']['pacf_lags']
    
    # Compute PACF
    pacf_values = pacf(df['load_MW'], nlags=nlags)
    
    # Statistical significance threshold (approximate 95% confidence interval)
    n = len(df)
    threshold = 1.96 / np.sqrt(n)
    
    # Find the largest contiguous block of significant PACF values near the origin
    short_term_lag = 0
    for i in range(1, len(pacf_values)):
        if abs(pacf_values[i]) > threshold:
            short_term_lag = i
        else:
            # First non-significant lag stops the "contiguous" block from origin
            break
            
    # Check for strong seasonal lag at index 96
    seasonal_lag = 0
    if nlags >= 96:
        # Check window around 96
        window_96 = pacf_values[94:99]
        if np.max(np.abs(window_96)) > threshold:
            seasonal_lag = 96
            
    # Optimal lag = max(contiguous short-term, seasonal lag)
    optimal_lag = max(short_term_lag, seasonal_lag)
    
    logger.info(f"Discovered contiguous short-term lag: {short_term_lag}")
    logger.info(f"Discovered seasonal lag: {seasonal_lag}")
    logger.info(f"Final Optimal Lag determined as: {optimal_lag}")
    
    # Save to file
    output_path = os.path.join(config['paths']['processed_data'], 'optimal_lag.json')
    with open(output_path, 'w') as f:
        json.dump({"optimal_lag": int(optimal_lag)}, f)
        
    return optimal_lag
