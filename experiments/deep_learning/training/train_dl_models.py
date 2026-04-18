import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import pandas as pd
from datetime import datetime

# Import project-specific modules
from .. import config
from . import training_utils
from .early_stopping import EarlyStopping
from ..evaluation import metrics

# Import models
from ..models.lstm_model import LSTMModel
from ..models.gru_model import GRUModel
from ..models.cnn_lstm_model import CNNLSTMModel
from ..models.tcn_model import TCNModel

def set_seed(seed):
    """
    Enforces deterministic execution.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(model, loader, criterion, optimizer, device, clip_value=1.0):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, y_batch)
            total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_test(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    start_time = time.time()
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            output = model(X_batch)
            all_preds.append(output.cpu().numpy())
            all_targets.append(y_batch.numpy())
    
    inference_time = (time.time() - start_time) / len(loader.dataset)
    y_pred = np.concatenate(all_preds).flatten()
    y_true = np.concatenate(all_targets).flatten()
    
    # Metrics
    mae, rmse, mape = metrics.compute_metrics(y_true, y_pred)
    return mae, rmse, mape, inference_time, y_pred

def run_experiment(model_name, model_class, train_loader, val_loader, test_loader, device):
    print(f"\n--- Training {model_name} ---")
    input_dim = next(iter(train_loader))[0].shape[2]
    model = model_class(input_dim=input_dim).to(device)
    
    criterion = nn.L1Loss() # MAE
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    save_path = os.path.join(config.MODEL_SAVE_DIR, f"{model_name.lower().replace('-', '_')}.pt")
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, path=save_path, verbose=True)
    
    log_file = os.path.join(config.LOG_DIR, f"{model_name.lower().replace('-', '_')}_training_log.csv")
    history = []
    
    start_time = time.time()
    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        history.append({
            "Epoch": epoch,
            "Train_Loss": train_loss,
            "Validation_Loss": val_loss,
            "Learning_Rate": current_lr
        })
        
        print(f"Epoch {epoch}/{config.EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr}")
        
        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
            
    training_time = time.time() - start_time
    
    # Save training log
    pd.DataFrame(history).to_csv(log_file, index=False)
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(save_path))
    mae, rmse, mape, infer_time, y_pred = evaluate_test(model, test_loader, device)
    
    # Results logging
    results = {
        "Model": model_name,
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "MAPE": round(float(mape), 4),
        "Training_Time": round(training_time, 2),
        "Inference_Time": round(float(infer_time), 6),
        "Random_Seed": config.SEED,
        "Dataset_Version": config.DATASET_VERSION,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Verification: NaN/Inf check in predictions
    if np.isnan(y_pred).any() or np.isinf(y_pred).any():
        print(f"WARNING: Invalid values detected in {model_name} predictions.")
    
    print(f"{model_name} Final Results - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
    
    return results

def main():
    set_seed(config.SEED)
    
    # Ensure directories exist
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.RESULTS_FILE), exist_ok=True)
    
    # Load and Verify Data
    X_train, X_test, y_train, y_test = training_utils.load_and_verify_data(config.DATASET_DIR)
    
    # Get Dataloaders
    train_loader, val_loader, test_loader = training_utils.get_dataloaders(
        X_train, y_train, X_test, y_test, 
        batch_size=config.BATCH_SIZE, 
        val_split=config.VAL_SPLIT,
        seed=config.SEED
    )
    
    device = config.DEVICE
    print(f"Executing on device: {device}")
    
    models_to_train = [
        ("LSTM", LSTMModel),
        ("GRU", GRUModel),
        ("CNN-LSTM", CNNLSTMModel),
        ("TCN", TCNModel)
    ]
    
    all_results = []
    for name, m_class in models_to_train:
        res = run_experiment(name, m_class, train_loader, val_loader, test_loader, device)
        all_results.append(res)
    
    # Save final results CSV
    results_df = pd.DataFrame(all_results)
    if os.path.exists(config.RESULTS_FILE):
        # Append to existing if desired, but user specified creating it for the pipeline
        results_df.to_csv(config.RESULTS_FILE, index=False)
    else:
        results_df.to_csv(config.RESULTS_FILE, index=False)
        
    print(f"\nBenchmarking Complete. Results saved to {config.RESULTS_FILE}")
    
    # Final Verification
    if len(all_results) != 4:
        print(f"CRITICAL: Expected 4 results, got {len(all_results)}")
    else:
        print("SUCCESS: Pipeline completed with exactly 4 models.")

if __name__ == "__main__":
    main()
