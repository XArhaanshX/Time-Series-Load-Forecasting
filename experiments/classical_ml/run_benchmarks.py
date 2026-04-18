import time
import os
from . import config, utils
from .evaluation import metrics
from .models.baselines import PersistenceModel, SeasonalPersistenceModel
import subprocess

def run_baselines():
    print("\n=== Running Baseline Models ===")
    X_train, y_train = utils.load_canonical_data(mode="train")
    X_test, y_test = utils.load_canonical_data(mode="test")
    
    baselines = [
        ("Persistence", PersistenceModel()),
        ("Seasonal Persistence", SeasonalPersistenceModel())
    ]
    
    for name, model in baselines:
        if utils.is_model_completed(name):
            print(f"Skipping {name} (Already completed)")
            continue
            
        print(f"Evaluating {name}...")
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        infer_time = time.time() - start_time
        
        mae, rmse, mape = metrics.compute_metrics(y_test, y_pred)
        utils.log_results(name, mae, rmse, mape, 0.0, infer_time)
        print(f"{name} Results - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

def run_ml_scripts():
    print("\n=== Running Classical ML Model Scripts ===")
    scripts = [
        "models.train_linear_models",
        "models.train_tree_models",
        "models.train_boosting_models"
    ]
    
    for script in scripts:
        print(f"Executing {script}...")
        # Execute as module to preserve imports
        subprocess.run(["python", "-m", f"experiments.classical_ml.{script}"], check=True)

if __name__ == "__main__":
    # Ensure current directory is SLTF root for module imports
    print("Classical ML Benchmarking Pipeline Started.")
    
    # Check Result file existence
    if os.path.exists(config.RESULTS_FILE):
        print(f"Appending to existing results file: {config.RESULTS_FILE}")
    else:
        print(f"Creating new results file: {config.RESULTS_FILE}")
    
    # 1. Run Baselines
    run_baselines()
    
    # 2. Run ML Models
    # Note: These are computationally intensive.
    run_ml_scripts()
    
    print("\nBenchmarking Complete. Summary available in:", config.RESULTS_FILE)
