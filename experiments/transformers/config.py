import torch
import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_V3_DIR = os.path.join("experiments", "deep_learning", "dataset_v3")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# --- Dataset Configuration ---
SEQUENCE_LENGTH = 192
EXPECTED_FEATURES = 45
TARGET_COL = "load_MW"
DATASET_VERSION = "dataset_v3_sequence_dataset"
MAX_TRAIN_SAMPLES = 30000

# --- Hyperparameters ---
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.0005
EARLY_STOPPING_PATIENCE = 5
SEED = 42

# --- Model Specifics ---
D_MODEL = 64
NUM_HEADS = 4
NUM_LAYERS = 4
DROPOUT = 0.1
PATCH_LENGTH = 16
PATCH_STRIDE = 8

# --- Safety Check ---
assert D_MODEL % NUM_HEADS == 0, "D_MODEL must be divisible by NUM_HEADS"

# --- Training Stability ---
GRADIENT_CLIP = 1.0
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5

# --- Device ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Reproducibility ---
CUDNN_DETERMINISTIC = True
CUDNN_BENCHMARK = False

# Ensure directories exist
for d in [RESULTS_DIR, LOGS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)
