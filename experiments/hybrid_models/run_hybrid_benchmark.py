import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from datetime import datetime

from . import config
from .gru_residual_model import GRUResidualModel
from .dataset_loader import get_dataloaders
from .residual_dataset_builder import load_lgbm_model
from . import training_utils

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_hybrid_pipeline(dry_run=False):
    set_seeds(config.SEED)
    device = config.DEVICE
    print(f"--- Hybrid Residual Forecasting Stage (Stage 5) ---")
    print(f"Device: {device}")
    
    # 1. Load DataLoaders (includes residual target computation)
    train_loader, val_loader, test_loader = get_dataloaders()
    
    # 2. Initialize GRU Residual Model
    model = GRUResidualModel(
        input_dim=config.FEATURE_COUNT,
        hidden_dim_1=64,
        hidden_dim_2=32
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    checkpoint_path = os.path.join(config.MODELS_DIR, "gru_residual_best.pt")
    early_stopping = training_utils.EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, path=checkpoint_path)
    
    # 3. Training Loop
    epochs = 1 if dry_run else config.EPOCHS
    history = []
    
    print(f"Starting training for {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        train_loss = training_utils.train_epoch(model, train_loader, criterion, optimizer, device, config.GRAD_CLIP)
        val_loss = training_utils.validate_epoch(model, val_loader, criterion, device)
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]['lr']
        })
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break
            
    training_time = time.time() - start_time
    
    # 4. Hybrid Inference
    print("\n--- Performing Hybrid Inference ---")
    # Load best checkpoint
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    # Baseline: LightGBM Predictions
    lgbm_model = load_lgbm_model()
    
    # Load canonical test set for baseline predictions
    test_df = pd.read_parquet(config.LGBM_TEST_DATA)
    if test_df.index.name != 'timestamp' and 'timestamp' in test_df.columns:
        test_df = test_df.set_index('timestamp')
        
    ts_test = np.load(os.path.join(config.V3_DATASET_PATH, "timestamps_test.npy"), allow_pickle=True)
    ts_test_idx = pd.to_datetime(ts_test)
    
    # Selection and ordering must match model training
    feature_names = lgbm_model.feature_name_
    
    # Note: Test set might not contain all timestamps found in dataset_v3 if they were in the train set originally
    # but we can join/reindex if needed.
    # However, build_residual_dataset concatenated train and test sets into a full_df.
    # To be extremely safe, we should load both again or use a combined one.
    train_df = pd.read_parquet(config.LGBM_TRAIN_DATA)
    if train_df.index.name != 'timestamp' and 'timestamp' in train_df.columns:
        train_df = train_df.set_index('timestamp')
    full_df = pd.concat([train_df, test_df], axis=0)
    
    X_test_lgbm = full_df.loc[ts_test_idx, feature_names]
    y_base_test = lgbm_model.predict(X_test_lgbm)
    actual_test = full_df.loc[ts_test_idx, "load_MW"].values
    
    # Correction: GRU Residual Predictions
    gru_res_preds = []
    infer_start_time = time.time()
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            gru_res_preds.append(preds.cpu().numpy())
    
    inference_time = time.time() - infer_start_time
    gru_res_preds = np.concatenate(gru_res_preds).flatten()
    
    # Final Forecast
    y_final = y_base_test + gru_res_preds
    
    # 5. Evaluation
    mae, rmse, mape = training_utils.compute_metrics(actual_test, y_final)
    
    # Baseline comparison (LGBM only)
    b_mae, b_rmse, b_mape = training_utils.compute_metrics(actual_test, y_base_test)
    
    print("\n--- Evaluation Results ---")
    print(f"LGBM Baseline: MAE={b_mae:.2f}, RMSE={b_rmse:.2f}, MAPE={b_mape:.2f}%")
    print(f"Hybrid Model:  MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")
    print(f"Improvement:   {b_mape - mape:.4f}%")
    
    # 6. Logging
    result = {
        "Model": "LightGBM_GRU_Hybrid",
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4),
        "Training_Time": round(training_time, 2),
        "Inference_Time": round(inference_time, 6),
        "Dataset_Version": config.DATASET_VERSION,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    df_results = pd.DataFrame([result])
    if os.path.exists(config.RESULTS_FILE):
        df_results.to_csv(config.RESULTS_FILE, mode='a', header=False, index=False)
    else:
        df_results.to_csv(config.RESULTS_FILE, index=False)
        
    # Save training history
    pd.DataFrame(history).to_csv(os.path.join(config.LOGS_DIR, "hybrid_training_log.csv"), index=False)
    
    print(f"\nPipeline complete. Results saved to {config.RESULTS_FILE}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run for only 1 epoch")
    args = parser.parse_args()
    
    run_hybrid_pipeline(dry_run=args.dry_run)
