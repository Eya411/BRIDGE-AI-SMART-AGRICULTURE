
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from configurations_B import (
    BEST_CHECKPOINT_B,
    CLASSES
)

from evaluation_utils_B import (
    create_test_loader,
    load_evaluation_model,
    predict_model
)


# =====================================================
# 1. DOSSIER DE RESULTATS
# =====================================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "results" / "evaluation_B"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# 2. EVALUATION
# =====================================================

def main():

    print("\n========== EVALUATION FINALE B ==========\n")

    loader = create_test_loader()

    model = load_evaluation_model(
        BEST_CHECKPOINT_B
    )

    y_true, y_pred, probabilities = predict_model(
        model,
        loader
    )

    # =================================================
    # 3. METRIQUES GLOBALES
    # =================================================

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision_macro = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall_macro = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_weighted = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("\n========== RESULTATS GLOBAUX B ==========")

    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Precision macro: {precision_macro:.4f}")
    print(f"Recall macro   : {recall_macro:.4f}")
    print(f"F1 macro       : {f1_macro:.4f}")
    print(f"F1 weighted    : {f1_weighted:.4f}")

    print("Nombre de tests :", len(y_true))

    # =================================================
    # 4. RAPPORT PAR CLASSE
    # =================================================

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        zero_division=0,
        output_dict=True
    )

    print("\n========== RAPPORT PAR CLASSE ==========\n")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(CLASSES))),
            target_names=CLASSES,
            zero_division=0
        )
    )

    # Sauvegarde JSON.
    import json

    with open(
        OUTPUT_DIR / "classification_report_B.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=4,
            ensure_ascii=False
        )

    # =================================================
    # 5. SAUVEGARDE DES METRIQUES
    # =================================================

    metrics = {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    }

    with open(
        OUTPUT_DIR / "metrics_B.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(["Metric", "Value"])

        for key, value in metrics.items():

            writer.writerow([key, value])

    # =================================================
    # 6. MATRICE DE CONFUSION
    # =================================================

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(CLASSES)))
    )

    np.savetxt(
        OUTPUT_DIR / "confusion_matrix_B.csv",
        cm,
        delimiter=",",
        fmt="%d"
    )

    plt.figure(figsize=(10, 8))

    plt.imshow(cm, interpolation="nearest")

    plt.title("Matrice de confusion - Modele B")

    plt.colorbar()

    positions = np.arange(len(CLASSES))

    plt.xticks(
        positions,
        CLASSES,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        positions,
        CLASSES
    )

    plt.xlabel("Classe prédite")

    plt.ylabel("Classe réelle")

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "confusion_matrix_B.png",
        dpi=300
    )

    plt.close()

    # =================================================
    # 7. SAUVEGARDE DES PREDICTIONS
    # =================================================

    with open(
        OUTPUT_DIR / "predictions_B.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "true_label",
            "predicted_label",
            "correct",
            "probability"
        ])

        for i in range(len(y_true)):

            writer.writerow([
                CLASSES[y_true[i]],
                CLASSES[y_pred[i]],
                y_true[i] == y_pred[i],
                max(probabilities[i])
            ])

    # =================================================
    # 8. FIN
    # =================================================

    print("\n========== EVALUATION B TERMINEE ==========")

    print("Résultats sauvegardés dans :")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
