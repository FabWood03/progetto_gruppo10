import sys
import numpy as np

# Permette l'import dei moduli locali
sys.path.append(".")

from src.preprocessing.loader import DataLoader
from src.knn.classifier import KNNClassifier
from src.validation.holdout import HoldoutValidation

# Evaluator
from src.metrics.evaluator import (
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

# Plotter
from src.metrics.plotter import (
    plot_confusion_matrix,
    plot_roc_curve
)


def test_1_knn_holdout_full():
    """
    TEST 1
    Verifica completa:
    - Holdout
    - KNN
    - predict / predict_proba
    - metriche singole
    - ROC + AUC
    - evaluate_metrics
    - plot
    """

    print("\n=== TEST 1: KNN + HOLDOUT + EVALUATOR + PLOTTER ===")

    # ------------------
    # CARICAMENTO DATI
    # ------------------
    loader = DataLoader(path="../data/version_1.csv")
    x, y, _ = loader.load()

    print(f"Dataset: {x.shape[0]} campioni, {x.shape[1]} feature")
    print(f"Distribuzione classi: {np.unique(y, return_counts=True)}")

    # ------------------
    # PARAMETRI
    # ------------------
    k = 5
    seed = 42
    test_size = 0.2

    knn = KNNClassifier(k=k, distance="euclidean", random_state=seed)
    validator = HoldoutValidation(test_size=test_size, random_state=seed)

    # ------------------
    # HOLDOUT SPLIT
    # ------------------
    x_train, x_test, y_train, y_test = validator.split(x, y)

    print(f"Train samples: {x_train.shape[0]}")
    print(f"Test samples : {x_test.shape[0]}")

    # ------------------
    # TRAIN
    # ------------------
    knn.fit(x_train, y_train)

    # ------------------
    # PREDICT
    # ------------------
    y_pred = knn.predict(x_test)

    # Classe positiva = 4 (come definito nel tuo evaluator)
    y_score = knn.predict_proba(x_test)[:, 1]

    # ------------------
    # CONFUSION COUNTS
    # ------------------
    c = confusion_counts(y_test, y_pred, pos_label=4)

    print("\nMetriche singole:")
    print(f"Accuracy     : {accuracy_rate(c):.4f}")
    print(f"Error rate   : {error_rate(c):.4f}")
    print(f"Sensitivity  : {sensitivity(c):.4f}")
    print(f"Specificity  : {specificity(c):.4f}")
    print(f"Precision    : {precision(c):.4f}")
    print(f"F1-score     : {f1_score(c):.4f}")
    print(f"G-Mean       : {geometric_mean(c):.4f}")

    # ------------------
    # ROC + AUC (manuale)
    # ------------------
    fpr, tpr, thresholds = roc_curve_manual(
        y_true=y_test,
        y_score=y_score,
        pos_label=4,
        neg_label=2
    )

    auc_value = calculate_auc(fpr, tpr)
    print(f"AUC          : {auc_value:.4f}")

    # ------------------
    # EVALUATE_METRICS
    # ------------------
    all_metrics = evaluate_metrics(
        y_true=y_test,
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
            "auc",
        ],
        pos_label=4,
        neg_label=2
    )

    print("\nMetriche tramite evaluate_metrics:")
    for name, value in all_metrics.items():
        print(f"{name:<12}: {value:.4f}")

    # ------------------
    # CONFUSION MATRIX
    # ------------------
    cm = np.array([
        [c.tp, c.fn],
        [c.fp, c.tn]
    ])

    plot_confusion_matrix(
        cm=cm,
        labels=["Malignant (4)", "Benign (2)"],
        title="Confusion Matrix (Holdout)",
        save_path="outputs/test1_confusion_matrix.png",
        normalize=False
    )

    plot_confusion_matrix(
        cm=cm,
        labels=["Malignant (4)", "Benign (2)"],
        title="Confusion Matrix (Normalized)",
        save_path="outputs/test1_confusion_matrix_normalized.png",
        normalize=True
    )

    # ------------------
    # ROC CURVE
    # ------------------
    plot_roc_curve(
        fpr=fpr,
        tpr=tpr,
        auc_value=auc_value,
        title="ROC Curve (Holdout)",
        save_path="outputs/test1_roc_curve.png"
    )

    print("\nTEST 1 COMPLETATO CON SUCCESSO ✅")


if __name__ == "__main__":
    test_1_knn_holdout_full()
