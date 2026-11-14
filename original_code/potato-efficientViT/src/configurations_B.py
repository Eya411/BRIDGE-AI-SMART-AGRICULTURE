from pathlib import Path
import torch

# =====================================================
# CHEMINS
# =====================================================

SRC_DIR = Path(__file__).resolve().parent

PROJECT_DIR = SRC_DIR.parents[2]

DATA = PROJECT_DIR / "data"

RESULTS_DIR = PROJECT_DIR / "results"

BASELINE_CHECKPOINT = (
    RESULTS_DIR
    / "checkpoints"
    / "Hybrid_EffNetViT_30epochs_seed42_best.pth"
)

SPLIT_FILE = RESULTS_DIR / "data_split.json"

B_RESULTS_DIR = RESULTS_DIR / "fine_tuning_B"

B_CHECKPOINT_DIR = B_RESULTS_DIR / "checkpoints"

B_LOG_DIR = B_RESULTS_DIR / "logs"

B_METRICS_DIR = B_RESULTS_DIR / "metrics"

for directory in [
    B_RESULTS_DIR,
    B_CHECKPOINT_DIR,
    B_LOG_DIR,
    B_METRICS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)


# =====================================================
# DATASET
# =====================================================

CLASSES = sorted([
    p.name for p in DATA.iterdir() if p.is_dir()
])

NUM_CLASSES = len(CLASSES)

IMAGE_SIZE = 224

TEST_SIZE = 0.10

VALI_SIZE = 0.10

RANDOM_STATE = 42


# =====================================================
# ENTRAINEMENT
# =====================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

EPOCHS = 30

BATCH_SIZE = 4

GRAD_ACCUM_STEPS = 16

NUM_WORKERS = 0

PIN_MEMORY = DEVICE.type == "cuda"

USE_AMP = DEVICE.type == "cuda"

SEED = 42

# Learning rates séparés
LR_EFFICIENTNET = 1e-5

LR_HEAD = 1e-4

WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 8

SCHEDULER_PATIENCE = 3

SCHEDULER_FACTOR = 0.5


# =====================================================
# FICHIERS DE SORTIE
# =====================================================

BEST_CHECKPOINT_B = (
    B_CHECKPOINT_DIR / "best_model_B.pth"
)

LAST_CHECKPOINT_B = (
    B_CHECKPOINT_DIR / "last_model_B.pth"
)

CSV_LOG_B = B_LOG_DIR / "training_log_B.csv"

METRICS_JSON_B = B_METRICS_DIR / "metrics_B.json"

CONFUSION_MATRIX_B = (
    B_METRICS_DIR / "confusion_matrix_B.png"
)

CLASSIFICATION_REPORT_B = (
    B_METRICS_DIR / "classification_report_B.txt"
)


# =====================================================
# VERIFICATIONS
# =====================================================

if not DATA.exists():
    raise FileNotFoundError(f"Dataset introuvable : {DATA}")

if NUM_CLASSES != 7:
    raise ValueError(
        f"7 classes attendues, {NUM_CLASSES} trouvées."
    )

if not BASELINE_CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Checkpoint A introuvable : {BASELINE_CHECKPOINT}"
    )

if not SPLIT_FILE.exists():
    raise FileNotFoundError(
        f"Fichier de split introuvable : {SPLIT_FILE}"
    )


def print_configuration():

    print("\n========== CONFIGURATION B ==========")

    print("Dataset :", DATA)
    print("Classes :", CLASSES)
    print("Nombre de classes :", NUM_CLASSES)

    print("Device :", DEVICE)
    print("Epochs :", EPOCHS)
    print("Batch size :", BATCH_SIZE)
    print("Gradient accumulation :", GRAD_ACCUM_STEPS)

    print("LR EfficientNet :", LR_EFFICIENTNET)
    print("LR Head :", LR_HEAD)

    print("Checkpoint A :", BASELINE_CHECKPOINT)
    print("Checkpoint B :", BEST_CHECKPOINT_B)

    print("=====================================\n")