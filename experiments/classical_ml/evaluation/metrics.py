import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Computes MAPE. Handles zero values by adding epsilon if necessary,
    though load values are typically > 0.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Mask zeros to avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def compute_metrics(y_true, y_pred):
    """
    Returns a dictionary of MAE, RMSE, and MAPE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    return mae, rmse, mape
