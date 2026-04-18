import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Import modules
from experiments.deep_learning import config
from experiments.deep_learning.training import dataset_loader, training_utils
from experiments.deep_learning.evaluation import metrics
from experiments.deep_learning.models.lstm_model import LSTMModel
from experiments.deep_learning.models.gru_model import GRUModel
from experiments.deep_learning.models.cnn_lstm_model import CNNLSTMModel
from experiments.deep_learning.models.tcn_model import TCNModel

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_training_suite():
    set_seeds(config.SEED)
    device = config.DEVICE
    print(f"--- Deep Learning Benchmarking Pipeline ---")
    print(f"Device used: {device}")
    
    # 1. Load Scaler and Metadata
    y_scaler = joblib.load(os.path.join(config.DATASET_PATH, "y_scaler.pkl"))
    with open(os.path.join(config.DATASET_PATH, "dataset_summary.json"), "r") as f:
        import json
        ds_summary = json.load(f)
    
    print(f"Dataset Version: {ds_summary['dataset_version']}")
    print(f"Train Shape: {ds_summary['train_tensor_shape']}")
    print(f"Test Shape: {ds_summary['test_tensor_shape']}")
    
    # 2. Get Dataloaders
    train_loader, val_loader, test_loader = dataset_loader.get_dataloaders(
        config.DATASET_PATH, config.BATCH_SIZE, seed=config.SEED, device=device
    )
    
    input_dim = ds_summary['num_features']
    
    models_to_train = [
        ("LSTM", LSTMModel),
        ("GRU", GRUModel),
        ("CNN-LSTM", CNNLSTMModel),
        ("TCN", TCNModel)
    ]
    
    all_results = []
    
    # Ensure log/result directories
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.RESULTS_PATH), exist_ok=True)
    
    for model_name, model_class in models_to_train:
        print(f"\n--- Training {model_name} ---")
        model = model_class(input_dim=input_dim).to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"{model_name.lower()}_best.pt")
        early_stopping = training_utils.EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, path=checkpoint_path)
        
        history = []
        start_time = time.time()
        
        # Disable anomaly detection for speed
        torch.autograd.set_detect_anomaly(False)
        
        for epoch in range(1, config.EPOCHS + 1):
            train_loss = training_utils.train_epoch(model, train_loader, criterion, optimizer, device, config.GRAD_CLIP)
            val_loss = training_utils.validate_epoch(model, val_loader, criterion, device)
            
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
            
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:03d}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
                
            scheduler.step(val_loss)
            early_stopping(val_loss, model)
            
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break
        
        training_time = time.time() - start_time
        
        # Save Log
        pd.DataFrame(history).to_csv(os.path.join(config.LOG_DIR, f"{model_name.lower()}_training_log.csv"), index=False)
        
        # 3. Evaluation on Test Set
        print(f"Evaluation on Test Set...")
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        
        all_preds = []
        all_targets = []
        
        infer_start_time = time.time()
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                preds = model(X_batch)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y_batch.numpy())
        
        inference_time = (time.time() - infer_start_time) / len(test_loader.dataset)
        
        y_pred_scaled = np.concatenate(all_preds)
        y_true_scaled = np.concatenate(all_targets)
        
        # --- SAFE INVERSE SCALING ---
        y_pred_mw = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true_mw = y_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
        
        # Metrics in MW Scale
        mae, rmse, mape = metrics.compute_metrics(y_true_mw, y_pred_mw)
        
        print(f"Results (MW Scale): MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")
        
        all_results.append({
            "Model": model_name,
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "MAPE": round(mape, 4),
            "Training_Time": round(training_time, 2),
            "Inference_Time": round(inference_time, 6),
            "Random_Seed": config.SEED,
            "Dataset_Version": config.DATASET_VERSION,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    # 4. Save Final Results Table
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(config.RESULTS_PATH, index=False)
    print(f"\nFinal results saved to {config.RESULTS_PATH}")

if __name__ == "__main__":
    run_training_suite()
