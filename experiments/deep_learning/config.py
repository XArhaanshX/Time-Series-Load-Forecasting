import torch

# Random Seed for Reproducibility
SEED = 42

# Dataset Configuration
DATASET_VERSION = "v3_sequence_dataset"
DATASET_PATH = "experiments/deep_learning/dataset_v3"
SEQUENCE_LENGTH = 192

# Training Hyperparameters
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
GRAD_CLIP = 1.0

# Callbacks
EARLY_STOPPING_PATIENCE = 10

# Paths
RESULTS_PATH = "experiments/deep_learning/results/dl_model_results.csv"
CHECKPOINT_DIR = "experiments/deep_learning/models"
LOG_DIR = "experiments/deep_learning/logs"

# Global Device handling
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
