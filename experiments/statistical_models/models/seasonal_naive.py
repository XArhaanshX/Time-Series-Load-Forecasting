import pandas as pd
from .. import config

class SeasonalNaiveModel:
    """
    Seasonal Naive Baseline Predictor: y_hat(t) = y(t - S)
    where S is the seasonal period (e.g., 96 for 15-min daily data).
    """
    def __init__(self, seasonal_period=config.SEASONAL_PERIOD):
        self.seasonal_period = seasonal_period
        self.train_series = None

    def fit(self, train_series):
        """
        Store the training series to extract the last seasonal period for initial forecasts.
        """
        self.train_series = train_series

    def predict(self, horizon_length):
        """
        Forecast by repeating the last observed seasonal cycle.
        """
        # For seasonal naive, we essentially take the last 'seasonal_period' values 
        # from the training set and tile them to cover the forecast horizon.
        last_cycle = self.train_series.tail(self.seasonal_period).values
        
        # Calculate how many full cycles and partial cycles are needed
        num_full_cycles = horizon_length // self.seasonal_period
        remainder = horizon_length % self.seasonal_period
        
        predictions = []
        for _ in range(num_full_cycles):
            predictions.extend(last_cycle)
        if remainder > 0:
            predictions.extend(last_cycle[:remainder])
            
        return predictions
