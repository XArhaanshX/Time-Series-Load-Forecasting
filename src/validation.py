import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

def run_validation_splits(df, config):
    logger.info("Starting walk-forward validation splitting.")
    split_dir = config['paths']['split_data']
    os.makedirs(split_dir, exist_ok=True)
    
    # 15-min intervals in a year: 365 * 24 * 4 = 35040
    test_size = 35040 
    total_len = len(df)
    
    # Number of splits: We'll do 3 expanding window splits if data allows
    # Split 1: Train (Start to T-2Y), Test (T-2Y to T-1Y)
    # Split 2: Train (Start to T-1Y), Test (T-1Y to Now)
    
    # Simple logic for this pipeline: 
    # Use the last specified units from config if possible
    
    splits = []
    
    # Example Split: Last year as test
    if total_len > test_size * 2:
        # Split 1
        train_df = df.iloc[:-test_size]
        test_df = df.iloc[-test_size:]
        
        train_path = os.path.join(split_dir, 'train_fold_0.parquet')
        test_path = os.path.join(split_dir, 'test_fold_0.parquet')
        
        train_df.to_parquet(train_path)
        test_df.to_parquet(test_path)
        
        logger.info(f"Split 0 saved: Train {len(train_df)}, Test {len(test_df)}")
        splits.append((train_path, test_path))
    else:
        # Fallback to 80/20 if dataset is unexpectedly small
        split_idx = int(total_len * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        train_path = os.path.join(split_dir, 'train_final.parquet')
        test_path = os.path.join(split_dir, 'test_final.parquet')
        
        train_df.to_parquet(train_path)
        test_df.to_parquet(test_path)
        splits.append((train_path, test_path))

    return splits
