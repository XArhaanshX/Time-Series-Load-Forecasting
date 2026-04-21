import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import joblib
from . import config

class TransformerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_and_validate_data():
    print(f"--- Loading dataset_v3 from {config.DATASET_V3_DIR} ---")
    
    # 1. Load Tensors
    X_train = np.load(os.path.join(config.DATASET_V3_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(config.DATASET_V3_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(config.DATASET_V3_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(config.DATASET_V3_DIR, "y_test.npy"))
    
    # 1.1 Truncate training data for CPU efficiency
    print(f"Truncating training data to {config.MAX_TRAIN_SAMPLES} samples for CPU efficiency.")
    X_train = X_train[:config.MAX_TRAIN_SAMPLES]
    y_train = y_train[:config.MAX_TRAIN_SAMPLES]
    
    # Load Timestamps for validation
    ts_train = np.load(os.path.join(config.DATASET_V3_DIR, "timestamps_train.npy"), allow_pickle=True)
    ts_test = np.load(os.path.join(config.DATASET_V3_DIR, "timestamps_test.npy"), allow_pickle=True)

    # 2. Mandatory Validations
    print("Performing mandatory data integrity checks...")
    
    # Feature Consistency
    assert X_train.shape[2] == config.EXPECTED_FEATURES, \
        f"Feature mismatch: Expected {config.EXPECTED_FEATURES}, got {X_train.shape[2]}"
    
    # Shape Check
    assert X_train.shape[1] == config.SEQUENCE_LENGTH, \
        f"Sequence length mismatch: Expected {config.SEQUENCE_LENGTH}, got {X_train.shape[1]}"
    
    # Numerical Integrity
    assert np.isfinite(X_train).all(), "NaN or Inf values detected in X_train"
    assert np.isfinite(y_train).all(), "NaN or Inf values detected in y_train"
    
    # Scaling Validation
    train_mean = X_train.mean()
    train_std = X_train.std()
    print(f"X_train stats -> mean: {train_mean:.4f}, std: {train_std:.4f}")
    assert abs(train_mean) < 0.1, f"X_train mean too high: {train_mean:.4f}"
    assert 0.8 < train_std < 1.2, f"X_train std out of range: {train_std:.4f}"
    
    y_mean = y_train.mean()
    y_std = y_train.std()
    print(f"y_train stats -> mean: {y_mean:.4f}, std: {y_std:.4f}")
    assert abs(y_mean) < 0.3, f"y_train mean too high: {y_mean:.4f}" 
    assert 0.7 < y_std < 1.3, f"y_train std out of range: {y_std:.4f}"
    
    # Temporal Integrity
    # Convert to datetime if they are strings/numpy objects
    ts_train_max = np.max(ts_train)
    ts_test_min = np.min(ts_test)
    print(f"Max train timestamp: {ts_train_max}")
    print(f"Min test timestamp: {ts_test_min}")
    assert ts_train_max < ts_test_min, "Temporal overlap detected: train and test partitions are not strictly chronological!"

    print("SUCCESS: Data integrity verified.")
    return X_train, X_test, y_train, y_test

def get_dataloaders():
    X_train_full, X_test, y_train_full, y_test = load_and_validate_data()
    
    full_train_ds = TransformerDataset(X_train_full, y_train_full)
    test_ds = TransformerDataset(X_test, y_test)
    
    # Deterministic 80/20 split for validation
    train_size = int(0.8 * len(full_train_ds))
    val_size = len(full_train_ds) - train_size
    
    generator = torch.Generator().manual_seed(config.SEED)
    train_ds, val_ds = random_split(full_train_ds, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        pin_memory=True if "cuda" in config.DEVICE else False
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        pin_memory=True if "cuda" in config.DEVICE else False
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        pin_memory=True if "cuda" in config.DEVICE else False
    )
    
    return train_loader, val_loader, test_loader
