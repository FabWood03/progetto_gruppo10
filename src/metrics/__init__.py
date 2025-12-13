"""
Modulo metrics.

Fornisce funzioni per la valutazione delle prestazioni di un classificatore
e per la visualizzazione grafica dei risultati (matrice di confusione, ROC, AUC).
"""

from .evaluator import (
    ConfusionCounts,
    confusion_counts,
    accuracy_rate,
    error_rate,
    sensitivity,
    specificity,
    precision,
    f1_score,
    geometric_mean,
    roc_curve_manual,
    calculate_auc,
    evaluate_metrics,
)

from .plotter import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_metric_distribution,
)
