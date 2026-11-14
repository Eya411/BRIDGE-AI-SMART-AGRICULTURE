
import csv
from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

# Project root: BRIDGE_AI_POTATO
ROOT = Path(__file__).resolve().parents[3]

# Training log CSV (located in the main project directory)
CSV_FILE = (
    ROOT
    / "results"
    / "fine_tuning_B"
    / "logs"
    / "training_log_B.csv"
)

# Output directory (inside potato-efficientViT)
OUTPUT_DIR = (
    ROOT
    / "original_code"
    / "potato-efficientViT"
    / "results"
    / "fine_tuning_B"
    / "graphs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. MAIN FUNCTION
# ============================================================

def main():

    print("\n" + "=" * 55)
    print("       MODEL B - TRAINING GRAPHS")
    print("=" * 55)

    print(f"\nProject root: {ROOT}")
    print(f"CSV file: {CSV_FILE}")

    # --------------------------------------------------------
    # Check CSV existence
    # --------------------------------------------------------

    if not CSV_FILE.exists():

        raise FileNotFoundError(
            f"\nTraining CSV not found:\n{CSV_FILE}"
        )

    # --------------------------------------------------------
    # Load CSV data
    # --------------------------------------------------------

    with open(
        CSV_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(
            "The training CSV file is empty."
        )

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = [
        "epoch",
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy",
        "lr_effnet",
        "lr_head"
    ]

    for column in required_columns:

        if column not in rows[0]:

            raise ValueError(
                f"Missing CSV column: {column}"
            )

    # --------------------------------------------------------
    # Extract training metrics
    # --------------------------------------------------------

    epochs = [
        int(r["epoch"]) for r in rows
    ]

    train_loss = [
        float(r["train_loss"]) for r in rows
    ]

    val_loss = [
        float(r["val_loss"]) for r in rows
    ]

    train_acc = [
        float(r["train_accuracy"]) for r in rows
    ]

    val_acc = [
        float(r["val_accuracy"]) for r in rows
    ]

    lr_effnet = [
        float(r["lr_effnet"]) for r in rows
    ]

    lr_head = [
        float(r["lr_head"]) for r in rows
    ]

    print(f"\nNumber of epochs: {len(epochs)}")

    # ========================================================
    # 3. LOSS GRAPH
    # ========================================================

    plt.figure(figsize=(9, 6))

    plt.plot(
        epochs,
        train_loss,
        marker="o",
        label="Training loss"
    )

    plt.plot(
        epochs,
        val_loss,
        marker="o",
        label="Validation loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        "Training and Validation Loss - Model B"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "loss_B.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # 4. ACCURACY GRAPH
    # ========================================================

    plt.figure(figsize=(9, 6))

    plt.plot(
        epochs,
        train_acc,
        marker="o",
        label="Training accuracy"
    )

    plt.plot(
        epochs,
        val_acc,
        marker="o",
        label="Validation accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title(
        "Training and Validation Accuracy - Model B"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "accuracy_B.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # 5. LEARNING RATE GRAPH
    # ========================================================

    plt.figure(figsize=(9, 6))

    plt.plot(
        epochs,
        lr_effnet,
        marker="o",
        label="EfficientNet learning rate"
    )

    plt.plot(
        epochs,
        lr_head,
        marker="o",
        label="Classification head learning rate"
    )

    plt.yscale("log")

    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")

    plt.title(
        "Learning Rate Evolution - Model B"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "learning_rate_B.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # 6. FINAL RESULTS
    # ========================================================

    print("\n" + "=" * 55)
    print("       GRAPHS GENERATED SUCCESSFULLY")
    print("=" * 55)

    print("\nGenerated files:")

    print("- loss_B.png")
    print("- accuracy_B.png")
    print("- learning_rate_B.png")

    print(f"\nOutput directory:\n{OUTPUT_DIR}")

    print("\n" + "=" * 55)


# ============================================================
# 7. EXECUTION
# ============================================================

if __name__ == "__main__":
    main()
