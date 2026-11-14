
import os
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from dotenv import load_dotenv

# ==================================================
# 1. PROJECT PATHS
# ==================================================

SRC_DIR = Path(__file__).resolve().parent
REPO_DIR = SRC_DIR.parent
PROJECT_DIR = SRC_DIR.parents[2]

DATA = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
LOG_DIR = RESULTS_DIR / "logs"

for folder in [RESULTS_DIR, CHECKPOINT_DIR, LOG_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

if not DATA.exists():
    raise FileNotFoundError(
        f"Dataset introuvable : {DATA}"
    )

# ==================================================
# 2. DATASET CLASSES
# ==================================================

CLASSES = sorted([
    p.name for p in DATA.iterdir()
    if p.is_dir()
])

FEATURES = len(CLASSES)

if FEATURES != 7:
    raise ValueError(
        f"7 classes attendues, {FEATURES} trouvées : {CLASSES}"
    )

# ==================================================
# 3. REPRODUCIBILITY
# ==================================================

RANDOM_STATE = 42

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

# ==================================================
# 4. DEVICE
# ==================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==================================================
# 5. DATASET PARAMETERS
# ==================================================

IMAGE_SIZE = 224

TEST_SIZE = 0.10
VALI_SIZE = 0.10

# ==================================================
# 6. TRAINING PARAMETERS
# ==================================================

EPOCHS = 30

# GPU memory adaptation for RTX 3050 4 GB
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 16

lr = 0.0001
WEIGHT_DECAY = 0.0001

CRITERION = nn.CrossEntropyLoss()

TRAINING = True
AUGMENT = True

DATATYPE = "potato_leaf_disease"
NEW_DATASET = True

NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()
USE_AMP = torch.cuda.is_available()

# ==================================================
# 7. OPTIMIZER AND SCHEDULER
# ==================================================

def create_optimizer(model):
    trainable_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]

    return optim.Adam(
        trainable_parameters,
        lr=lr,
        weight_decay=WEIGHT_DECAY
    )


def create_scheduler(optimizer):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=5,
        factor=0.5
    )

# ==================================================
# 8. EXPERIMENT OUTPUT
# ==================================================

EXPERIMENT_NAME = (
    "Hybrid_EffNetViT_30epochs_seed42"
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR / f"{EXPERIMENT_NAME}_best.pth"
)

LAST_CHECKPOINT = (
    CHECKPOINT_DIR / f"{EXPERIMENT_NAME}_last.pth"
)

CSV_LOG = (
    LOG_DIR / f"{EXPERIMENT_NAME}.csv"
)

# ==================================================
# 9. WANDB
# ==================================================

load_dotenv(PROJECT_DIR / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

WANDB_MODE = (
    "online" if os.getenv("WANDB_API_KEY")
    else "disabled"
)

def initialize_wandb():
    return wandb.init(
        project=os.getenv(
            "WANDB_PROJECT",
            "potato-disease-diagnosis"
        ),
        name=EXPERIMENT_NAME,
        mode=WANDB_MODE,
        config={
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
            "effective_batch_size": BATCH_SIZE * GRAD_ACCUM_STEPS,
            "learning_rate": lr,
            "weight_decay": WEIGHT_DECAY,
            "seed": RANDOM_STATE,
            "architecture": "EfficientNetV2B3_ViT",
            "dataset": DATATYPE,
            "image_size": IMAGE_SIZE
        }
    )

# ==================================================
# 10. CONFIGURATION SUMMARY
# ==================================================

print("\n========== CONFIGURATION ==========")
print("Dataset:", DATA)
print("Classes:", CLASSES)
print("Nombre de classes:", FEATURES)
print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3),
            2
        ),
        "GB"
    )

print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)
print("Accumulation:", GRAD_ACCUM_STEPS)
print("Learning rate:", lr)
print("Weight decay:", WEIGHT_DECAY)
print("WandB mode:", WANDB_MODE)
print("===================================\n")
