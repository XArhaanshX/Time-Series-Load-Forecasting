import time
import joblib
import os
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .. import config, utils
from ..evaluation import metrics

def train_and_evaluate(model_name, model_obj, use_scaler=True):
    if utils.is_model_completed(model_name):
        print(f"Skipping {model_name} (Already completed)")
        return
        
    # Load data
    X_train, y_train = utils.load_canonical_data(mode="train")
    X_test, y_test = utils.load_canonical_data(mode="test")
    
    # Setup Pipeline
    if use_scaler:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model_obj)
        ])
    else:
        pipeline = Pipeline([
            ('model', model_obj)
        ])
        
    # Train
    print(f"Training {model_name}...")
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Save Model
    model_path = os.path.join(config.SAVED_MODELS_DIR, f"{model_name.lower().replace(' ', '_')}.joblib")
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")
    
    # Inference
    print(f"Evaluating {model_name}...")
    start_time = time.time()
    y_pred = pipeline.predict(X_test)
    infer_time = time.time() - start_time
    
    # Metrics
    mae, rmse, mape = metrics.compute_metrics(y_test, y_pred)
    
    # Log Results
    utils.log_results(model_name, mae, rmse, mape, train_time, infer_time)
    
    print(f"{model_name} Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

if __name__ == "__main__":
    # Linear Regression
    train_and_evaluate("Linear Regression", LinearRegression())
    
    # Ridge
    train_and_evaluate("Ridge", Ridge(random_state=config.SEED))
    
    # Lasso
    train_and_evaluate("Lasso", Lasso(alpha=0.1, max_iter=5000, random_state=config.SEED))
