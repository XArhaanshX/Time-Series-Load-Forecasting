# Configuration for Classical ML Benchmarking
import os

SEED = 42
TARGET_COLUMN = "load_MW"
DATASET_VERSION = "v1_canonical_sliding_window"

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_ROOT = os.path.dirname(BASE_DIR)
CANONICAL_DATA_DIR = os.path.join(EXPERIMENTS_ROOT, "canonical_dataset")

TRAIN_PATH = os.path.join(CANONICAL_DATA_DIR, "train.parquet")
TEST_PATH = os.path.join(CANONICAL_DATA_DIR, "test.parquet")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "model_results.csv")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "models", "saved_models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Ensure directories exist
for d in [RESULTS_DIR, SAVED_MODELS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)
