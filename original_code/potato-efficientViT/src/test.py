
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)
from torch.nn import functional as F

from configurations import DEVICE, CLASSES, RESULTS_DIR
from dataset import Dataset
from models import EfficientNetV2B3ViT


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_PATH = (
    RESULTS_DIR
    / "checkpoints"
    / "Hybrid_EffNetViT_30epochs_seed42_best.pth"
)

OUTPUT_DIR = RESULTS_DIR / "baseline"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURES_DIR = OUTPUT_DIR / "figures"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ============================================================
# CHARGEMENT DU MODELE
# ============================================================

def load_model(checkpoint_path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint introuvable : {checkpoint_path}"
        )

    model = EfficientNetV2B3ViT(num_labels=len(CLASSES)).to(DEVICE)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Le checkpoint ne contient pas model_state_dict."
        )

    if checkpoint.get("num_classes", len(CLASSES)) != len(CLASSES):
        raise ValueError("Le nombre de classes du checkpoint est incorrect.")

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    logging.info("Checkpoint chargé correctement.")
    logging.info("Époque du checkpoint : %s", checkpoint.get("epoch"))
    logging.info("Validation loss : %s", checkpoint.get("val_loss"))
    logging.info("Validation accuracy : %s", checkpoint.get("val_accuracy"))

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, test_loader):

    all_labels = []
    all_predictions = []

    total_loss = 0.0
    total_samples = 0

    model.eval()

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            # Somme des pertes pour obtenir ensuite
            # une moyenne pondérée par le nombre d'images.
            batch_loss = F.cross_entropy(
                outputs,
                labels,
                reduction="sum",
            )

            total_loss += batch_loss.item()
            total_samples += labels.size(0)

            predictions = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )

    if total_samples == 0:
        raise ValueError("Le test set est vide.")

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    test_loss = total_loss / total_samples

    accuracy = accuracy_score(all_labels, all_predictions)

    precision = precision_score(
        all_labels,
        all_predictions,
        labels=range(len(CLASSES)),
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        labels=range(len(CLASSES)),
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        labels=range(len(CLASSES)),
        average="macro",
        zero_division=0,
    )

    mcc = matthews_corrcoef(all_labels, all_predictions)

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=range(len(CLASSES)),
    )

    report = classification_report(
        all_labels,
        all_predictions,
        labels=range(len(CLASSES)),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    results = {
        "test_loss": float(test_loss),
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "mcc": float(mcc),
        "number_of_test_images": int(total_samples),
        "classes": CLASSES,
        "class_metrics": report,
        "confusion_matrix": cm.tolist(),
    }

    return results, cm


# ============================================================
# SAUVEGARDE DES RESULTATS
# ============================================================

def save_results(results, cm):

    metrics_path = METRICS_DIR / "baseline_test_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False,
        )

    figure_path = FIGURES_DIR / "baseline_confusion_matrix.png"

    fig, ax = plt.subplots(figsize=(10, 8))

    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set(
        xticks=np.arange(len(CLASSES)),
        yticks=np.arange(len(CLASSES)),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        xlabel="Classe prédite",
        ylabel="Classe réelle",
        title="Baseline A - Matrice de confusion",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    threshold = cm.max() / 2 if cm.size else 0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logging.info("Métriques sauvegardées : %s", metrics_path)
    logging.info("Matrice sauvegardée : %s", figure_path)


# ============================================================
# MAIN
# ============================================================

def main():

    logging.info("Device : %s", DEVICE)
    logging.info("Classes : %s", CLASSES)

    dataset = Dataset()

    _, _, test_loader = dataset.prepare_dataset()

    logging.info(
        "Nombre d'images du test set : %d",
        len(test_loader.dataset),
    )

    model = load_model(CHECKPOINT_PATH)

    results, cm = evaluate(model, test_loader)

    logging.info("================ RESULTATS TEST A ================")
    logging.info("Test loss       : %.4f", results["test_loss"])
    logging.info("Accuracy        : %.4f", results["accuracy"])
    logging.info("Precision macro : %.4f", results["precision_macro"])
    logging.info("Recall macro    : %.4f", results["recall_macro"])
    logging.info("F1-score macro  : %.4f", results["f1_macro"])
    logging.info("MCC             : %.4f", results["mcc"])

    logging.info("================ METRIQUES PAR CLASSE ================")

    for class_name in CLASSES:
        metrics = results["class_metrics"][class_name]

        logging.info(
            "%s | Precision: %.4f | Recall: %.4f | F1: %.4f",
            class_name,
            metrics["precision"],
            metrics["recall"],
            metrics["f1-score"],
        )

    save_results(results, cm)

    logging.info("Évaluation de A terminée.")


if __name__ == "__main__":
    main()
