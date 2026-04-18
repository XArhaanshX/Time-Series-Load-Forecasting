import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        # Convert to torch tensors
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders(dataset_path, batch_size, val_split=0.2, seed=42, device="cpu"):
    """
    Loads v3 dataset and returns train, val, and test loaders.
    """
    # 1. Load Tensors
    X_train_all = np.load(os.path.join(dataset_path, "X_train.npy"))
    X_test = np.load(os.path.join(dataset_path, "X_test.npy"))
    y_train_all = np.load(os.path.join(dataset_path, "y_train.npy"))
    y_test = np.load(os.path.join(dataset_path, "y_test.npy"))
    
    # 2. Deterministic Train/Val Split
    num_total_train = len(X_train_all)
    indices = np.arange(num_total_train)
    
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    val_size = int(num_total_train * val_split)
    val_idx, train_idx = indices[:val_size], indices[val_size:]
    
    # 3. Create Datasets
    train_ds = SequenceDataset(X_train_all[train_idx], y_train_all[train_idx])
    val_ds = SequenceDataset(X_train_all[val_idx], y_train_all[val_idx])
    test_ds = SequenceDataset(X_test, y_test)
    
    # 4. DataLoader Config (GPU Optimization)
    is_cuda = (device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"))
    loader_kwargs = {
        "batch_size": batch_size,
        "pin_memory": True if is_cuda else False,
        "num_workers": 2 if is_cuda else 0
    }
    
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    
    return train_loader, val_loader, test_loader
