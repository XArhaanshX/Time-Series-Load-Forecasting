import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .. import config

def generate_cyclical_features(index):
    """
    Generates sin/cos cyclical features for hour of day and day of week from a timestamp index.
    """
    hours = index.hour
    days = index.dayofweek # Monday=0, Sunday=6
    
    # Hour of day (24h cycle)
    sin_hour = np.sin(2 * np.pi * hours / 24)
    cos_hour = np.cos(2 * np.pi * hours / 24)
    
    # Day of week (7d cycle)
    sin_day = np.sin(2 * np.pi * days / 7)
    cos_day = np.cos(2 * np.pi * days / 7)
    
    exog = pd.DataFrame({
        'sin_hour': sin_hour,
        'cos_hour': cos_hour,
        'sin_day': sin_day,
        'cos_day': cos_day
    }, index=index)
    
    return exog

def train_and_forecast_sarimax(train_series, test_index):
    """
    Fits a SARIMAX(2,1,2)(1,1,1,96) model with sin/cos features and returns a forecast.
    Falls back to (1,1,1)(1,1,1,96) if convergence fails.
    """
    # Generate exogenous features internally
    train_exog = generate_cyclical_features(train_series.index)
    test_exog = generate_cyclical_features(test_index)
    
    configs = [
        {"order": (2, 1, 2), "seasonal_order": (1, 1, 1, config.SEASONAL_PERIOD)},
        {"order": (1, 1, 1), "seasonal_order": (1, 1, 1, config.SEASONAL_PERIOD)}
    ]
    
    for cfg in configs:
        order = cfg["order"]
        seasonal_order = cfg["seasonal_order"]
        
        print(f"Attempting SARIMAX with order={order}, seasonal_order={seasonal_order}...")
        
        try:
            model = SARIMAX(
                train_series,
                exog=train_exog,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            # maxiter=50 prevents infinite likelihood optimization loops
            model_fit = model.fit(maxiter=50, disp=False)
            
            print(f"SARIMAX converged with order {order}.")
            # Forecast needs exog for the steps
            forecast = model_fit.forecast(steps=len(test_index), exog=test_exog)
            return forecast, order
            
        except Exception as e:
            print(f"SARIMAX failed to converge or errored with order {order}: {e}")
            continue
            
    raise RuntimeError("All SARIMAX configurations failed to converge.")
