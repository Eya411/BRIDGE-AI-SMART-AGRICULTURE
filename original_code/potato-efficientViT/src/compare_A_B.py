
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from configurations_B import (
    BASELINE_CHECKPOINT,
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

OUTPUT_DIR = ROOT / "results" / "comparison_A_B"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# 2. CALCUL DES METRIQUES
# =====================================================

def calculate_metrics(y_true, y_pred):

    return {
        "accuracy": accuracy_score(y_true, y_pred),

        "precision_macro": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "recall_macro": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "f1_weighted": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    }


# =====================================================
# 3. EVALUATION D'UN MODELE
# =====================================================

def evaluate_checkpoint(name, checkpoint_path, loader):

    print(f"\nEvaluation du modele {name}...")

    model = load_evaluation_model(
        checkpoint_path
    )

    y_true, y_pred, probabilities = predict_model(
        model,
        loader
    )

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        zero_division=0,
        output_dict=True
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(CLASSES)))
    )

    print(f"\nResultats du modele {name}")

    for metric, value in metrics.items():

        print(f"{metric}: {value:.4f}")

    del model

    return {
        "metrics": metrics,
        "report": report,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "probabilities": probabilities
    }


# =====================================================
# 4. COMPARAISON
# =====================================================

def main():

    print("\n========== COMPARAISON A / B ==========\n")

    # Même DataLoader pour les deux modèles.
    loader = create_test_loader()

    result_A = evaluate_checkpoint(
        "A",
        BASELINE_CHECKPOINT,
        loader
    )

    result_B = evaluate_checkpoint(
        "B",
        BEST_CHECKPOINT_B,
        loader
    )

    # Vérifier que les labels sont identiques.
    assert result_A["y_true"] == result_B["y_true"]

    # =================================================
    # 5. COMPARAISON DES METRIQUES
    # =================================================

    metrics_A = result_A["metrics"]
    metrics_B = result_B["metrics"]

    metric_names = list(metrics_A.keys())

    print("\n========== COMPARAISON GLOBALE ==========\n")

    for metric in metric_names:

        value_A = metrics_A[metric]
        value_B = metrics_B[metric]

        difference = value_B - value_A

        print(
            f"{metric:20s} "
            f"A={value_A:.4f} "
            f"B={value_B:.4f} "
            f"Delta={difference:+.4f}"
        )

    # =================================================
    # 6. CSV GLOBAL
    # =================================================

    with open(
        OUTPUT_DIR / "comparison_metrics.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Metric",
            "Model_A",
            "Model_B",
            "Difference_B_minus_A"
        ])

        for metric in metric_names:

            a = metrics_A[metric]
            b = metrics_B[metric]

            writer.writerow([
                metric,
                a,
                b,
                b - a
            ])

    # =================================================
    # 7. RAPPORT PAR CLASSE
    # =================================================

    with open(
        OUTPUT_DIR / "comparison_per_class.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "Model_A": result_A["report"],
                "Model_B": result_B["report"]
            },
            f,
            indent=4,
            ensure_ascii=False
        )
 # =================================================
    # 7 BIS. COMPARAISON DES METRIQUES PAR CLASSE
    # =================================================

    per_class_rows = []

    print("\n========== RESULTATS PAR CLASSE ==========\n")

    for class_name in CLASSES:

        report_A = result_A["report"][class_name]
        report_B = result_B["report"][class_name]

        recall_A = report_A["recall"]
        recall_B = report_B["recall"]

        f1_A = report_A["f1-score"]
        f1_B = report_B["f1-score"]

        precision_A = report_A["precision"]
        precision_B = report_B["precision"]

        support = report_A["support"]

        per_class_rows.append([
            class_name,
            support,
            precision_A,
            precision_B,
            recall_A,
            recall_B,
            recall_B - recall_A,
            f1_A,
            f1_B,
            f1_B - f1_A
        ])

        print(
            f"{class_name:15s} "
            f"Recall A={recall_A:.4f} "
            f"Recall B={recall_B:.4f} "
            f"Delta={recall_B - recall_A:+.4f} | "
            f"F1 A={f1_A:.4f} "
            f"F1 B={f1_B:.4f} "
            f"Delta={f1_B - f1_A:+.4f}"
        )

    with open(
        OUTPUT_DIR / "comparison_per_class.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Class",
            "Support",
            "Precision_A",
            "Precision_B",
            "Recall_A",
            "Recall_B",
            "Delta_Recall",
            "F1_A",
            "F1_B",
            "Delta_F1"
        ])

        writer.writerows(per_class_rows)
    # =================================================
    # 8. COMPARAISON GRAPHIQUE
    # =================================================

    x = np.arange(len(metric_names))

    width = 0.35

    values_A = [
        metrics_A[m] for m in metric_names
    ]

    values_B = [
        metrics_B[m] for m in metric_names
    ]

    plt.figure(figsize=(11, 6))

    plt.bar(
        x - width / 2,
        values_A,
        width,
        label="Model A"
    )

    plt.bar(
        x + width / 2,
        values_B,
        width,
        label="Model B"
    )

    plt.xticks(
        x,
        metric_names,
        rotation=25,
        ha="right"
    )

    plt.ylabel("Score")

    plt.ylim(0, 1)

    plt.title("Comparaison des performances A et B")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "comparison_metrics.png",
        dpi=300
    )

    plt.close()

    # =================================================
    # 9. MATRICES DE CONFUSION
    # =================================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(17, 7)
    )

    for ax, result, name in zip(
        axes,
        [result_A, result_B],
        ["Model A", "Model B"]
    ):

        cm = result["confusion_matrix"]

        ax.imshow(cm)

        ax.set_title(f"Confusion Matrix - {name}")

        ax.set_xticks(range(len(CLASSES)))
        ax.set_xticklabels(
            CLASSES,
            rotation=45,
            ha="right"
        )

        ax.set_yticks(range(len(CLASSES)))
        ax.set_yticklabels(CLASSES)

        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")

        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):

                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center"
                )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "comparison_confusion_matrices.png",
        dpi=300
    )

    plt.close()

    # =================================================
    # 10. PREDICTIONS DES DEUX MODELES
    # =================================================

    with open(
        OUTPUT_DIR / "predictions_A_B.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "true_label",
            "prediction_A",
            "prediction_B",
            "A_correct",
            "B_correct"
        ])

        for i in range(len(result_A["y_true"])):

            true_label = result_A["y_true"][i]

            pred_A = result_A["y_pred"][i]
            pred_B = result_B["y_pred"][i]

            writer.writerow([
                CLASSES[true_label],
                CLASSES[pred_A],
                CLASSES[pred_B],
                true_label == pred_A,
                true_label == pred_B
            ])

    # =================================================
    # 11. CONCLUSION AUTOMATIQUE
    # =================================================

    delta_accuracy = (
        metrics_B["accuracy"]
        - metrics_A["accuracy"]
    )

    print("\n========== CONCLUSION ==========")

    print(
        f"Difference Accuracy B-A : "
        f"{delta_accuracy * 100:+.2f} points"
    )

    if delta_accuracy > 0:
        print("Le modele B améliore l'accuracy sur le test.")

    elif delta_accuracy < 0:
        print("Le modele B obtient une accuracy inférieure.")

    else:
        print("Les deux modèles ont la même accuracy.")

    print("\nRésultats sauvegardés dans :")
    print(OUTPUT_DIR)

    print("\n========== COMPARAISON TERMINEE ==========")


if __name__ == "__main__":
    main()
