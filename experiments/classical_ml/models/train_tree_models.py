import time
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from .. import config, utils
from ..evaluation import metrics

def train_and_evaluate(model_name, model_obj):
    if utils.is_model_completed(model_name):
        print(f"Skipping {model_name} (Already completed)")
        return
        
    # Load data
    X_train, y_train = utils.load_canonical_data(mode="train")
    X_test, y_test = utils.load_canonical_data(mode="test")
    
    # Train
    print(f"Training {model_name}...")
    start_time = time.time()
    model_obj.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Save Model
    model_path = os.path.join(config.SAVED_MODELS_DIR, f"{model_name.lower().replace(' ', '_')}.joblib")
    joblib.dump(model_obj, model_path)
    print(f"Model saved to {model_path}")
    
    # Inference
    print(f"Evaluating {model_name}...")
    start_time = time.time()
    y_pred = model_obj.predict(X_test)
    infer_time = time.time() - start_time
    
    # Metrics
    mae, rmse, mape = metrics.compute_metrics(y_test, y_pred)
    
    # Log Results
    utils.log_results(model_name, mae, rmse, mape, train_time, infer_time)
    
    print(f"{model_name} Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

if __name__ == "__main__":
    # Random Forest (Subsampled for speed if necessary, but here we use defaults)
    train_and_evaluate("Random Forest", RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=config.SEED))
    
    # Gradient Boosting
    train_and_evaluate("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, random_state=config.SEED))
