import logging
import os
import yaml
import json
import pandas as pd
from src.ingestion import run_ingestion
from src.preprocessing import run_preprocessing
from src.stationarity_analysis import run_stationarity_analysis
from src.lag_analysis import run_lag_analysis
from src.feature_engineering import run_feature_engineering
from src.feature_selection import run_feature_selection
from src.dataset_construction import run_dataset_construction
from src.validation import run_validation_splits

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline_pre_model.log')
    ]
)
logger = logging.getLogger("PipelineOrchestrator")

def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_full_pipeline():
    try:
        logger.info("=== STARTING HARYANA LOAD FORECASTING DATA PIPELINE ===")
        config = load_config()
        
        # 1. Ingestion
        logger.info("--- Stage 1: Ingestion ---")
        raw_df = run_ingestion(config)
        
        # 2. Preprocessing
        logger.info("--- Stage 2: Preprocessing ---")
        clean_df, pre_report = run_preprocessing(raw_df, config)
        
        # 3. Stationarity Analysis
        logger.info("--- Stage 3: Stationarity Analysis ---")
        adf_report = run_stationarity_analysis(clean_df, config)
        
        # 4. Lag Analysis
        logger.info("--- Stage 4: Lag Analysis ---")
        optimal_lag = run_lag_analysis(clean_df, config)
        
        # 5. Feature Engineering
        logger.info("--- Stage 5: Feature Engineering ---")
        feat_df = run_feature_engineering(clean_df, config)
        
        # 6. Feature Selection
        logger.info("--- Stage 6: Feature Selection ---")
        selected_features = run_feature_selection(feat_df, config)
        
        # 7. Dataset Construction
        logger.info("--- Stage 7: Dataset Construction ---")
        final_matrix = run_dataset_construction(feat_df, selected_features, optimal_lag, config)
        
        # 8. Validation Splits
        logger.info("--- Stage 8: Validation Splits ---")
        splits = run_validation_splits(final_matrix, config)
        
        logger.info("=== PIPELINE EXECUTION COMPLETED SUCCESSFULLY ===")
        
        # Final Summary
        summary = {
            "total_observations": len(clean_df),
            "optimal_lag": int(optimal_lag),
            "features_selected": len(selected_features),
            "matrix_shape": final_matrix.shape,
            "splits_generated": len(splits)
        }
        with open(os.path.join(config['paths']['results'], 'pipeline_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
        
        print("\nPipeline Summary:")
        print(json.dumps(summary, indent=4))

    except Exception as e:
        logger.error(f"Pipeline failed at some stage: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    run_full_pipeline()
