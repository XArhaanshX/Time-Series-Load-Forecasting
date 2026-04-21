import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from . import config
from .residual_dataset_builder import build_residual_targets

class ResidualDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).view(-1, 1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders():
    # 1. Load sequence features
    print(f"Loading sequence features from {config.V3_DATASET_PATH}...")
    X_train_full = np.load(os.path.join(config.V3_DATASET_PATH, "X_train.npy"))
    X_test = np.load(os.path.join(config.V3_DATASET_PATH, "X_test.npy"))
    
    # 2. Build residual targets
    res_train_full, res_test = build_residual_targets()
    
    # 3. Integrity checks
    assert len(X_train_full) == len(res_train_full), f"Mismatch: X_train {len(X_train_full)}, res_train {len(res_train_full)}"
    assert len(X_test) == len(res_test), f"Mismatch: X_test {len(X_test)}, res_test {len(res_test)}"
    
    # 4. Chronological Validation Split (80/20)
    split_idx = int(len(X_train_full) * (1 - config.VALIDATION_SPLIT))
    
    X_train = X_train_full[:split_idx]
    res_train = res_train_full[:split_idx]
    
    X_val = X_train_full[split_idx:]
    res_val = res_train_full[split_idx:]
    
    print(f"Dataset split: Train={len(X_train):,}, Val={len(X_val):,}, Test={len(X_test):,}")
    
    # 5. Create Dataset objects
    train_ds = ResidualDataset(X_train, res_train)
    val_ds = ResidualDataset(X_val, res_val)
    test_ds = ResidualDataset(X_test, res_test)
    
    # 6. Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=False) # Keep order for safety, though features are localized
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    X_batch, y_batch = next(iter(train_loader))
    print(f"Batch X shape: {X_batch.shape}")
    print(f"Batch y shape: {y_batch.shape}")
    print("Dataset loader verified.")
