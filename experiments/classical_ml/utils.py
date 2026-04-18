import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
from . import config

def load_canonical_data(mode="train"):
    """
    Loads canonical dataset and logs characteristics.
    """
    path = config.TRAIN_PATH if mode == "train" else config.TEST_PATH
    
    print(f"Loading {mode} dataset from {path}...")
    df = pd.read_parquet(path)
    
    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]
    
    # Logging characteristics
    n_samples = len(df)
    n_features = len(X.columns)
    
    print(f"--- Dataset Characteristics ({mode}) ---")
    print(f"Number of samples: {n_samples:,}")
    print(f"Number of feature columns: {n_features:,}")
    print("------------------------------------------")
    
    # Basic integrity check
    if df.isnull().values.any():
        print("WARNING: Dataset contains NaNs. Consider further imputation.")
        
    return X, y

def log_results(model_name, mae, rmse, mape, train_time, infer_time):
    """
    Appends experiment results to the centralized results file.
    """
    results_path = config.RESULTS_FILE
    
    new_result = {
        "Model": model_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4),
        "Training_Time": round(train_time, 2),
        "Inference_Time": round(infer_time, 4),
        "Random_Seed": config.SEED,
        "Dataset_Version": config.DATASET_VERSION,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    df_new = pd.DataFrame([new_result])
    
    if not os.path.exists(results_path):
        df_new.to_csv(results_path, index=False)
    else:
        df_new.to_csv(results_path, mode='a', header=False, index=False)
        
    print(f"Results for {model_name} logged to {results_path}")

def is_model_completed(model_name):
    """
    Checks if a model has already been successfully evaluated and logged.
    """
    results_path = config.RESULTS_FILE
    if not os.path.exists(results_path):
        return False
    try:
        df = pd.read_csv(results_path)
        return model_name in df['Model'].values
    except Exception:
        return False
