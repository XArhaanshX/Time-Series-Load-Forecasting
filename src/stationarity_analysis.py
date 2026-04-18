import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.tsa.seasonal import STL
import os
import logging

logger = logging.getLogger(__name__)

def run_stationarity_analysis(df, config):
    logger.info("Starting stationarity and seasonality analysis.")
    results_dir = config['paths']['results']
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Augmented Dickey-Fuller Test
    logger.info("Running ADF Test...")
    adf_result = adfuller(df['load_MW'])
    adf_report = {
        'Test Statistic': adf_result[0],
        'p-value': adf_result[1],
        'Stationary': adf_result[1] < 0.05
    }
    
    # 2. ACF Plot (up to lag 700)
    logger.info("Generating ACF plot...")
    lags = config['analysis']['acf_lags']
    acf_values = acf(df['load_MW'], nlags=lags)
    
    plt.figure(figsize=(12, 6))
    plt.stem(range(len(acf_values)), acf_values)
    plt.title(f'Autocorrelation Function (up to lag {lags})')
    plt.xlabel('Lag (15-min intervals)')
    plt.ylabel('ACF')
    plt.axhline(0, color='black', linestyle='--')
    
    # Annotate key seasonal lags
    if lags >= 96:
        plt.axvline(96, color='r', linestyle='--', alpha=0.5, label='Daily (96)')
    if lags >= 672:
        plt.axvline(672, color='g', linestyle='--', alpha=0.5, label='Weekly (672)')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'acf_plot.png'))
    plt.close()
    
    # 3. STL Decomposition
    logger.info("Performing STL decomposition...")
    period = config['analysis']['stl_period']
    res = STL(df['load_MW'], period=period).fit()
    
    fig = res.plot()
    fig.suptitle('STL Decomposition (Trend, Seasonal, Residual)')
    plt.savefig(os.path.join(results_dir, 'stl_decomposition.png'))
    plt.close()
    
    logger.info("Analysis plots saved to results directory.")
    return adf_report
