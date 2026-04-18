import numpy as np
import pandas as pd

class PersistenceModel:
    """
    Predicts y(t) = y(t-1).
    Dynamically locates the t-1 lag column.
    """
    def __init__(self):
        self.lag_col = None

    def fit(self, X, y=None):
        # Locate column for t-1. Typically starts with 'load_MW_t-1' or matches pattern
        # Since X columns are like 'feat_t-k', we look for 'load_MW_t-1'
        candidates = [c for c in X.columns if c.endswith('_t-1') and 'load' in c.lower()]
        if not candidates:
            # Fallback to any t-1 feature if load specifically not found
            candidates = [c for c in X.columns if c.endswith('_t-1')]
            
        if not candidates:
            raise ValueError("Could not dynamically locate a lag_t-1 column in X.")
            
        self.lag_col = candidates[0]
        print(f"PersistenceModel: Using {self.lag_col} for predictions.")
        return self

    def predict(self, X):
        if self.lag_col is None:
            raise ValueError("Model must be fitted to discover lag columns first.")
        return X[self.lag_col].values

class SeasonalPersistenceModel:
    """
    Predicts y(t) = y(t-96).
    Dynamically locates the t-96 lag column.
    """
    def __init__(self):
        self.lag_col = None

    def fit(self, X, y=None):
        # Locate column for t-96.
        candidates = [c for c in X.columns if c.endswith('_t-96') and 'load' in c.lower()]
        if not candidates:
            candidates = [c for c in X.columns if c.endswith('_t-96')]
            
        if not candidates:
            raise ValueError("Could not dynamically locate a lag_t-96 column in X.")
            
        self.lag_col = candidates[0]
        print(f"SeasonalPersistenceModel: Using {self.lag_col} for predictions.")
        return self

    def predict(self, X):
        if self.lag_col is None:
            raise ValueError("Model must be fitted to discover lag columns first.")
        return X[self.lag_col].values
