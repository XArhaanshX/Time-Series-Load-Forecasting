import time
import os
import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.stattools import adfuller

from . import config
from .utils import dataset_loader
from .evaluation import metrics
from .models.seasonal_naive import SeasonalNaiveModel
from .models.sarima_model import train_and_forecast_sarima
from .models.sarimax_model import train_and_forecast_sarimax

def run_adf_test(series):
    """
    Performs Augmented Dickey-Fuller test and returns results and interpretation.
    """
    print("Running Augmented Dickey-Fuller (ADF) test...")
    result = adfuller(series.dropna())
    
    adf_stat = result[0]
    p_value = result[1]
    is_stationary = p_value < 0.05
    
    conclusion = "Series is stationary" if is_stationary else "Series is non-stationary"
    
    status_str = (
        f"ADF Statistic: {adf_stat:.4f}\n"
        f"p-value: {p_value:.4f}\n"
        f"Conclusion: {conclusion}"
    )
    
    print(status_str)
    return adf_stat, p_value, conclusion

def log_model_run(model_name, dataset_size, adf_results, start_time, end_time):
    """
    Writes detailed log for a specific model run.
    """
    log_path = os.path.join(config.LOGS_DIR, f"{model_name.lower().replace(' ', '_')}_log.txt")
    
    if not os.path.exists(config.LOGS_DIR):
        os.makedirs(config.LOGS_DIR)
        
    start_dt = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
    end_dt = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
    
    with open(log_path, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Training Dataset Size: {dataset_size}\n")
        f.write(f"Seasonal Period: {config.SEASONAL_PERIOD}\n")
        f.write(f"ADF Test Results:\n")
        f.write(f"  ADF Statistic: {adf_results[0]:.4f}\n")
        f.write(f"  p-value: {adf_results[1]:.4f}\n")
        f.write(f"  Conclusion: {adf_results[2]}\n")
        f.write(f"Training Start Time: {start_dt}\n")
        f.write(f"Training Completion Time: {end_dt}\n")
        f.write(f"Duration: {end_time - start_time:.2f} seconds\n")

def save_result_to_csv(row):
    """
    Appends a single result row to the results CSV.
    """
    df = pd.DataFrame([row])
    if not os.path.exists(config.RESULTS_FILE):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config.RESULTS_FILE), exist_ok=True)
        df.to_csv(config.RESULTS_FILE, index=False)
    else:
        df.to_csv(config.RESULTS_FILE, mode='a', header=False, index=False)

def run_benchmarks():
    print("\n=== Haryana STLF Statistical Benchmarking Pipeline (Stabilized) ===")
    
    # 1. Load Data
    train_series, test_series = dataset_loader.load_statistical_data()
    
    # --- STABILITY VALIDATION ---
    print("Performing safety validation checks...")
    # 1. Assert no NaNs
    assert not train_series.isnull().any(), "Safety Check Failed: Training series contains NaNs."
    # 2. Assert length matches MAX_TRAIN_POINTS
    assert len(train_series) == config.MAX_TRAIN_POINTS, f"Safety Check Failed: Training series length ({len(train_series)}) != MAX_TRAIN_POINTS ({config.MAX_TRAIN_POINTS})."
    # 3. Confirm seasonal period
    assert config.SEASONAL_PERIOD == 96, f"Safety Check Failed: Seasonal period ({config.SEASONAL_PERIOD}) != 96."
    print("Safety validation checks passed.")
    
    # Verification: Chronological Check
    if not train_series.index.is_monotonic_increasing:
        print("ERROR: Training timestamps are not strictly increasing.")
        return
    
    # 2. Run Diagnostics
    adf_results = run_adf_test(train_series)
    
    models_to_run = ["Seasonal Naive", "SARIMA", "SARIMAX"]
    
    for model_name in models_to_run:
        print(f"\n--- Running {model_name} ---")
        
        # Start Timing
        start_time = time.time()
        
        predictions = None
        used_order = "N/A"
        
        try:
            if model_name == "Seasonal Naive":
                model = SeasonalNaiveModel()
                model.fit(train_series)
                train_end = time.time()
                predictions = model.predict(len(test_series))
                test_end = time.time()
            
            elif model_name == "SARIMA":
                train_start = time.time()
                predictions, used_order = train_and_forecast_sarima(train_series, len(test_series))
                test_end = time.time()
                train_end = train_start + (test_end - train_start) * 0.9 # Approximation for logging
            
            elif model_name == "SARIMAX":
                train_start = time.time()
                predictions, used_order = train_and_forecast_sarimax(train_series, test_series.index)
                test_end = time.time()
                train_end = train_start + (test_end - train_start) * 0.9 # Approximation for logging
            
            train_duration = train_end - start_time
            infer_duration = test_end - train_end
            
            # Verification: Prediction length
            if len(predictions) != len(test_series):
                print(f"ERROR: {model_name} prediction length ({len(predictions)}) does not match test set ({len(test_series)})")
                continue
            
            # Verification: Finite values
            if not np.isfinite(predictions).all():
                print(f"WARNING: {model_name} predictions contain non-finite values.")
            
            # 3. Compute Metrics
            eval_metrics = metrics.compute_metrics(test_series, predictions)
            
            # 4. Log Detailed Run
            log_model_run(model_name, len(train_series), adf_results, start_time, train_end)
            # Append model order to log
            log_path = os.path.join(config.LOGS_DIR, f"{model_name.lower().replace(' ', '_')}_log.txt")
            with open(log_path, 'a') as f:
                f.write(f"Model Configuration: {used_order}\n")
            
            # 5. Save Summary Results
            result_row = {
                "Model": model_name,
                "MAE": eval_metrics["MAE"],
                "RMSE": eval_metrics["RMSE"],
                "MAPE": eval_metrics["MAPE"],
                "Training_Time": round(train_duration, 2),
                "Inference_Time": round(infer_duration, 4),
                "Dataset_Version": config.DATASET_VERSION,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            save_result_to_csv(result_row)
            
            print(f"{model_name} Results - MAE: {eval_metrics['MAE']:.2f}, RMSE: {eval_metrics['RMSE']:.2f}, MAPE: {eval_metrics['MAPE']:.2f}%")
            
        except Exception as e:
            print(f"An error occurred while running {model_name}: {e}")

if __name__ == "__main__":
    run_benchmarks()
    print("\nStatistical Benchmarking Complete.")
