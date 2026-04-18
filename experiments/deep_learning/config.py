import torch

# Random Seed for Reproducibility
SEED = 42

# Dataset Configuration
DATASET_VERSION = "v2_sequence_dataset"
SEQUENCE_LENGTH = 96
VAL_SPLIT = 0.2

# Training Hyperparameters
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001

# Callbacks
EARLY_STOPPING_PATIENCE = 10

# Paths
RESULTS_FILE = "experiments/deep_learning/results/dl_model_results.csv"
DATASET_DIR = "experiments/deep_learning/dataset"
MODEL_SAVE_DIR = "experiments/deep_learning/models/saved_models"
LOG_DIR = "experiments/deep_learning/logs"

# Global Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
