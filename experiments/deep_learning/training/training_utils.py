import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import logging

# Set up logging for training utils
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_and_verify_data(dataset_dir):
    """
    Loads .npy tensors and performs strict shape/integrity verification.
    """
    try:
        X_train = np.load(os.path.join(dataset_dir, "X_train.npy"))
        X_test = np.load(os.path.join(dataset_dir, "X_test.npy"))
        y_train = np.load(os.path.join(dataset_dir, "y_train.npy"))
        y_test = np.load(os.path.join(dataset_dir, "y_test.npy"))
        
        # Log dimensions
        print("\n--- Loaded Deep Learning Dataset ---")
        print(f"Train Samples: {X_train.shape[0]}")
        print(f"Test Samples: {X_test.shape[0]}")
        print(f"Sequence Length: {X_train.shape[1]}")
        print(f"Feature Count: {X_train.shape[2]}")
        
        # Hard Failure Checks
        if X_train.shape[1] != 96:
            raise RuntimeError(f"HARD FAILURE: Expected sequence length 96, got {X_train.shape[1]}")
        
        if X_train.shape[0] != y_train.shape[0] or X_test.shape[0] != y_test.shape[0]:
            raise RuntimeError("HARD FAILURE: Mismatch between features (X) and target (y) sample counts.")
            
        if np.isnan(X_train).any() or np.isnan(X_test).any() or np.isnan(y_train).any():
            raise RuntimeError("HARD FAILURE: NaN values detected in input tensors.")
            
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"Error during data validation: {e}")
        raise

def get_dataloaders(X_train, y_train, X_test, y_test, batch_size, val_split=0.2, seed=42):
    """
    Splits training data into train/val and returns DataLoaders.
    """
    num_train = len(X_train)
    indices = list(range(num_train))
    split = int(np.floor(val_split * num_train))
    
    # Shuffle for splitting but keep seed for reproducibility
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    train_idx, val_idx = indices[split:], indices[:split]
    
    # Create Datasets
    train_ds = SequenceDataset(X_train[train_idx], y_train[train_idx])
    val_ds = SequenceDataset(X_train[val_idx], y_train[val_idx])
    test_ds = SequenceDataset(X_test, y_test)
    
    # Create Loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
