import os
import time
import pandas as pd
from datetime import datetime

from . import config, dataset_loader, training_utils
from .models.transformer_model import BaselineTransformer
from .models.patchtst_model import PatchTST

def log_results_to_csv(model_name, mae, rmse, mape, train_time, inf_time):
    results_file = os.path.join(config.RESULTS_DIR, "transformer_model_results.csv")
    
    new_result = {
        "Model": model_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4),
        "Training_Time": round(train_time, 2),
        "Inference_Time": round(inf_time, 4),
        "Dataset_Version": config.DATASET_VERSION,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    df = pd.DataFrame([new_result])
    
    if os.path.exists(results_file):
        df_existing = pd.read_csv(results_file)
        df = pd.concat([df_existing, df], ignore_index=True)
    
    df.to_csv(results_file, index=False)
    print(f"Results appended to {results_file}")

def main():
    print("\n" + "="*50)
    print("STAGE 4: TRANSFORMER BENCHMARKING PIPELINE")
    print("="*50)
    
    # 1. Initialization
    training_utils.set_seed(config.SEED)
    
    # 2. Data Loading
    train_loader, val_loader, test_loader = dataset_loader.get_dataloaders()
    
    # 3. Model Suite
    models_to_train = [
        ("BaselineTransformer", BaselineTransformer(
            n_features=config.EXPECTED_FEATURES,
            d_model=config.D_MODEL,
            n_heads=config.NUM_HEADS,
            n_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
            seq_len=config.SEQUENCE_LENGTH
        )),
        ("PatchTST", PatchTST(
            n_features=config.EXPECTED_FEATURES,
            seq_len=config.SEQUENCE_LENGTH,
            patch_len=config.PATCH_LENGTH,
            stride=config.PATCH_STRIDE,
            d_model=config.D_MODEL,
            n_heads=config.NUM_HEADS,
            n_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT
        ))
    ]
    
    for name, model in models_to_train:
        print(f"\nEvaluating Model Architecture: {name}")
        
        # Numerical Dry Run
        if not training_utils.numerical_dry_run(model, train_loader):
            print(f"CRITICAL FAILURE: Numerical dry run failed for {name}. Aborting model training.")
            continue
            
        # Training Orchestration
        trainer = training_utils.Trainer(model, train_loader, val_loader, test_loader, name)
        trainer.train()
        
        # Test Evaluation
        mae, rmse, mape = trainer.evaluate()
        
        # Persistence
        log_results_to_csv(name, mae, rmse, mape, trainer.training_time, trainer.inference_time)
        
    print("\n" + "="*50)
    print("TRANSFORMER BENCHMARKING COMPLETE")
    print("="*50)

if __name__ == "__main__":
    main()
