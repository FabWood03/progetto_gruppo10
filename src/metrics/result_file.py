from pathlib import Path
from typing import List, Sequence

import numpy as np

from src.metrics.evaluator import (
    evaluate_metrics,
    confusion_counts,
    roc_curve_manual,
    calculate_auc
)

from src.metrics.plotter import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_metric_distribution
)

import csv


# ======================================================
# Salvataggio metriche numeriche su CSV
# ======================================================

def save_metrics_csv(
    metrics_list: List[dict],
    save_path: str
) -> None:
    """
    Salva tutte le metriche (una riga per fold/esperimento) in CSV.
    """
    if not metrics_list:
        raise ValueError("Lista metriche vuota")

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, mode="w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=metrics_list[0].keys()
        )
        writer.writeheader()
        writer.writerows(metrics_list)


def generate_results(
    y_true_list: List[Sequence[int]],
    y_pred_list: List[Sequence[int]],
    y_score_list: List[Sequence[float]],
    labels: List[str],
    output_dir: str,
    experiment_name: str
) -> None:
    """
    Genera TUTTI i risultati richiesti dalla consegna.

    - CSV con tutte le metriche
    - Confusion Matrix
    - ROC Curve
    - Distribuzione metriche
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    # ===============================
    # METRICHE SU OGNI FOLD
    # ===============================
    for y_true, y_pred, y_score in zip(y_true_list, y_pred_list, y_score_list):

        metrics = evaluate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score,
            metrics=[
                "accuracy",
                "error",
                "sensitivity",
                "specificity",
                "precision",
                "f1",
                "gmean",
                "auc"
            ]
        )

        all_metrics.append(metrics)

    # ===============================
    # SALVA CSV
    # ===============================
    save_metrics_csv(
        metrics_list=all_metrics,
        save_path=str(output_dir / f"{experiment_name}_metrics.csv")
    )

    # ===============================
    # CONFUSION MATRIX (media)
    # ===============================
    all_y_true = np.concatenate(y_true_list)
    all_y_pred = np.concatenate(y_pred_list)

    c = confusion_counts(all_y_true, all_y_pred)

    cm = np.array([
        [c.tn, c.fp],
        [c.fn, c.tp]
    ])

    plot_confusion_matrix(
        cm=cm,
        labels=labels,
        title=f"Confusion Matrix - {experiment_name}",
        save_path=output_dir / f"{experiment_name}_confusion_matrix.png",
        normalize=False
    )

    # ===============================
    # ROC CURVE
    # ===============================
    all_y_score = np.concatenate(y_score_list)

    fpr, tpr, _ = roc_curve_manual(all_y_true, all_y_score)
    auc_value = calculate_auc(fpr, tpr)

    plot_roc_curve(
        fpr=fpr,
        tpr=tpr,
        auc_value=auc_value,
        title=f"ROC Curve - {experiment_name}",
        save_path=output_dir / f"{experiment_name}_roc_curve.png"
    )

    # ===============================
    # DISTRIBUZIONE METRICHE
    # ===============================
    accuracies = [m["accuracy"] for m in all_metrics]

    plot_metric_distribution(
        values=accuracies,
        metric_name="Accuracy",
        title=f"Accuracy Distribution - {experiment_name}",
        save_path=output_dir / f"{experiment_name}_accuracy_distribution.png"
    )



