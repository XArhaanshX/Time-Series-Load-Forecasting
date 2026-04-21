import os

# Paths
DATASET_PATH = "experiments/canonical_dataset/"
TRAIN_FILE = "train.parquet"
TEST_FILE = "test.parquet"

TRAIN_PATH = os.path.join(DATASET_PATH, TRAIN_FILE)
TEST_PATH = os.path.join(DATASET_PATH, TEST_FILE)

TARGET_COLUMN = "load_MW"

# Time Series Properties
FREQUENCY = "15min"
# 96 observations correspond to one full daily cycle (24h * 4 intervals/h)
SEASONAL_PERIOD = 96

# Computational Constraints
# SARIMA complexity grows rapidly with dataset size. 
# Training is limited to the most recent 10k points to remain feasible.
MAX_TRAIN_POINTS = 10000

# Results and Logging
RESULTS_FILE = "experiments/statistical_models/results/statistical_model_results.csv"
LOGS_DIR = "experiments/statistical_models/logs/"
DATASET_VERSION = "canonical_v1"

SEED = 42
