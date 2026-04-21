import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .. import config

def train_and_forecast_sarima(train_series, forecast_steps):
    """
    Fits a SARIMA(2,1,2)(1,1,1,96) model and returns a forecast.
    Falls back to (1,1,1)(1,1,1,96) if convergence fails.
    """
    configs = [
        {"order": (2, 1, 2), "seasonal_order": (1, 1, 1, config.SEASONAL_PERIOD)},
        {"order": (1, 1, 1), "seasonal_order": (1, 1, 1, config.SEASONAL_PERIOD)}
    ]
    
    for cfg in configs:
        order = cfg["order"]
        seasonal_order = cfg["seasonal_order"]
        
        print(f"Attempting SARIMA with order={order}, seasonal_order={seasonal_order}...")
        
        try:
            model = SARIMAX(
                train_series,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            # maxiter=50 prevents infinite likelihood optimization loops
            model_fit = model.fit(maxiter=50, disp=False)
            
            print(f"SARIMA converged with order {order}.")
            forecast = model_fit.forecast(steps=forecast_steps)
            return forecast, order
            
        except Exception as e:
            print(f"SARIMA failed to converge or errored with order {order}: {e}")
            continue
            
    raise RuntimeError("All SARIMA configurations failed to converge.")
