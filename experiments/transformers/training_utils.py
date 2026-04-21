import torch
import numpy as np
import random
import time
import os
import joblib
from tqdm import tqdm
from . import config, metrics

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = config.CUDNN_DETERMINISTIC
    torch.backends.cudnn.benchmark = config.CUDNN_BENCHMARK

class EarlyStopping:
    def __init__(self, patience=10, path="best_checkpoint.pt"):
        self.patience = patience
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, model_name):
        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model_name = model_name
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        self.criterion = torch.nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=config.SCHEDULER_FACTOR, 
            patience=config.SCHEDULER_PATIENCE
        )
        
        self.checkpoint_path = os.path.join(config.MODELS_DIR, f"{model_name}_best.pt")
        self.early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, path=self.checkpoint_path)
        
        # Load scaler for inverse transformation
        scaler_path = os.path.join(config.DATASET_V3_DIR, "y_scaler.pkl")
        self.y_scaler = joblib.load(scaler_path)
        
        self.log_file = os.path.join(config.LOGS_DIR, f"{model_name}_training.log")
        with open(self.log_file, "w") as f:
            f.write("epoch,train_loss,val_loss,lr\n")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for X, y in self.train_loader:
            X, y = X.to(config.DEVICE), y.to(config.DEVICE)
            
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.criterion(output, y)
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRADIENT_CLIP)
            
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(config.DEVICE), y.to(config.DEVICE)
                output = self.model(X)
                loss = self.criterion(output, y)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def train(self):
        print(f"Starting training for {self.model_name}...")
        start_time = time.time()
        
        for epoch in range(1, config.EPOCHS + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}/{config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {lr:.6f}")
            
            # Log progress
            with open(self.log_file, "a") as f:
                f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{lr:.6f}\n")
            
            self.scheduler.step(val_loss)
            self.early_stopping(val_loss, self.model)
            
            if self.early_stopping.early_stop:
                print("Early stopping triggered.")
                break
        
        self.training_time = time.time() - start_time
        print(f"Training complete. Best Val Loss: {self.early_stopping.best_loss:.6f}")
        
    def evaluate(self):
        print(f"Evaluating {self.model_name} on test set...")
        # Load best weights
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        self.model.eval()
        
        all_preds = []
        all_targets = []
        
        inf_start = time.time()
        with torch.no_grad():
            for X, y in self.test_loader:
                X = X.to(config.DEVICE)
                output = self.model(X)
                all_preds.append(output.cpu().numpy())
                all_targets.append(y.numpy())
        
        self.inference_time = time.time() - inf_start
        
        y_pred = np.concatenate(all_preds).reshape(-1, 1)
        y_true = np.concatenate(all_targets).reshape(-1, 1)
        
        # Inverse Scaling (Critical for MW metrics)
        y_pred_mw = self.y_scaler.inverse_transform(y_pred).flatten()
        y_true_mw = self.y_scaler.inverse_transform(y_true).flatten()
        
        mae, rmse, mape = metrics.compute_metrics(y_true_mw, y_pred_mw)
        
        print(f"Results for {self.model_name} (MW):")
        print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}%")
        
        return mae, rmse, mape

def numerical_dry_run(model, loader):
    print("--- Running Numerical Dry Run ---")
    model.train()
    model.to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = torch.nn.MSELoss()
    
    # Get one batch
    X, y = next(iter(loader))
    X, y = X.to(config.DEVICE), y.to(config.DEVICE)
    
    # Forward Pass
    try:
        output = model(X)
        print("Forward pass successful.")
    except Exception as e:
        print(f"Forward pass FAILED: {e}")
        return False
        
    if not torch.isfinite(output).all():
        print("Numerical instability detected in output (NaN/Inf)!")
        return False
        
    # Backward Pass
    try:
        loss = criterion(output, y)
        loss.backward()
        print("Backward pass successful.")
    except Exception as e:
        print(f"Backward pass FAILED: {e}")
        return False
        
    # Check Gradients
    finite_grads = True
    for name, param in model.named_parameters():
        if param.grad is not None:
            if not torch.isfinite(param.grad).all():
                print(f"Gradient instability detected in {name}!")
                finite_grads = False
    
    if not finite_grads:
        return False
        
    print("Numerical Dry Run COMPLETED SUCCESSFULLLY.")
    return True
