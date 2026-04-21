import os
import torch

# Random Seed for Reproducibility
SEED = 42

# Dataset Configuration
DATASET_VERSION = "dataset_v3_sequence_dataset"
V3_DATASET_PATH = "experiments/deep_learning/dataset_v3"
SEQUENCE_LENGTH = 192
FEATURE_COUNT = 45

# Hybrid Model Configuration
LGBM_MODEL_PATH = "experiments/classical_ml/models/saved_models/lightgbm.joblib"
LGBM_TRAIN_DATA = "experiments/canonical_dataset/train.parquet"
LGBM_TEST_DATA = "experiments/canonical_dataset/test.parquet"
ENGINEERED_FEATURES_PATH = "data/features/engineered_features.parquet"

# Training Hyperparameters
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001
GRAD_CLIP = 1.0
EARLY_STOPPING_PATIENCE = 5
VALIDATION_SPLIT = 0.2

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "hybrid_model_results.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Ensure directories exist
for d in [RESULTS_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
