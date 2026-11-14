
import os
import logging
from pathlib import Path
from time import strftime, gmtime

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from dotenv import load_dotenv

from models import EfficientNetV2B3ViT

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

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
    raise FileNotFoundError(f"Dataset introuvable : {DATA}")

CLASSES = sorted([
    p.name for p in DATA.iterdir() if p.is_dir()
])

FEATURES = len(CLASSES)

if FEATURES != 7:
    raise ValueError(f"7 classes attendues, {FEATURES} trouvées")

# --------------------------------------------------
# REPRODUCIBILITY
# --------------------------------------------------

RANDOM_STATE = 42

import random
import numpy as np

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

# --------------------------------------------------
# DATASET
# --------------------------------------------------

TEST_SIZE = 0.1
VALI_SIZE = 0.1

# --------------------------------------------------
# TRAINING
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

EPOCHS = 30
BATCH_SIZE = 64

lr = 0.0001
WEIGHT_DECAY = 0.0001

CRITERION = nn.CrossEntropyLoss()

TRAINING = True
AUGMENT = True
DATATYPE = "potatodata"
NEW_DATASET = True

# --------------------------------------------------
# MODEL
# --------------------------------------------------

MODELS = {
    "EffNetViT": EfficientNetV2B3ViT().to(DEVICE)
}

OPTIMIZERS = {
    "EffNetViT": optim.Adam(
        MODELS["EffNetViT"].parameters(),
        lr=lr,
        weight_decay=WEIGHT_DECAY
    )
}

SCHEDULER = {
    "EffNetViT": optim.lr_scheduler.ReduceLROnPlateau(
        OPTIMIZERS["EffNetViT"],
        patience=5,
        factor=0.5
    )
}

# --------------------------------------------------
# EXPERIMENT OUTPUT
# --------------------------------------------------

EXPERIMENT_NAME = "Hybrid_EffNetViT_30epochs_seed42"

BEST_CHECKPOINT = (
    CHECKPOINT_DIR / f"{EXPERIMENT_NAME}_best.pth"
)

LAST_CHECKPOINT = (
    CHECKPOINT_DIR / f"{EXPERIMENT_NAME}_last.pth"
)

CSV_LOG = LOG_DIR / f"{EXPERIMENT_NAME}.csv"

# --------------------------------------------------
# WANDB
# --------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

wandb_mode = (
    "online" if os.getenv("WANDB_API_KEY") else "disabled"
)

wandb.init(
    project=os.getenv("WANDB_PROJECT", "potato-disease-diagnosis"),
    name=EXPERIMENT_NAME,
    mode=wandb_mode,
    config={
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": lr,
        "weight_decay": WEIGHT_DECAY,
        "seed": RANDOM_STATE,
        "architecture": "EfficientNetV2B3_ViT",
        "dataset": DATATYPE
    }
)

print("Dataset:", DATA)
print("Classes:", CLASSES)
print("Nombre de classes:", FEATURES)
print("Device:", DEVICE)
print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)
