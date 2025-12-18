import math
import time
import numpy as np
from itertools import combinations

from src.metrics import (
    confusion_counts, accuracy_rate, error_rate, sensitivity,
    specificity, precision, f1_score, geometric_mean
)
from src.validation.base import ValidationStrategy
import src.utils as utils


class LeavePOutValidation(ValidationStrategy):
    """
    Leave-P-Out validation teoricamente corretta per KNN:
    - nessun pre-calcolo globale delle distanze
    - nessun data leakage
    - distanze calcolate solo tra test e training
    - batch prediction
    """

    def __init__(self, p: int = 1):
        if p < 1:
            raise ValueError("p deve essere >= 1")
        self.p = p

    def validate(self, model, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples = X.shape[0]
        if self.p >= n_samples:
            raise ValueError("p deve essere < n_samples")

        all_indices = np.arange(n_samples)
        n_combs = math.comb(n_samples, self.p)

        print(f"Leave-P-Out | n={n_samples}, p={self.p}, iterazioni={n_combs}")

        aggregated_cm = np.zeros((2, 2), dtype=np.int64)
        metrics_storage = np.zeros((n_combs, 7))

        start_time = time.time()
        benchmark = 200

        for i, test_idx in enumerate(combinations(all_indices, self.p)):
            test_idx = np.asarray(test_idx)

            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_idx] = False

            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_idx], y[test_idx]

            # Fit
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Metriche
            c = confusion_counts(y_test, y_pred, pos_label=1)

            aggregated_cm[0, 0] += c.tp
            aggregated_cm[0, 1] += c.fn
            aggregated_cm[1, 0] += c.fp
            aggregated_cm[1, 1] += c.tn

            metrics_storage[i] = [
                accuracy_rate(c),
                error_rate(c),
                sensitivity(c),
                specificity(c),
                precision(c),
                f1_score(c),
                geometric_mean(c)
            ]

            # Feedback tempo
            if i == benchmark:
                elapsed = time.time() - start_time
                avg = elapsed / benchmark
                eta = avg * n_combs
                print(f"\nVelocità: {avg * 1000:.3f} ms/iter")
                print(f"Stima completamento: {utils.format_duration(eta)}")
                print("-" * 30)

            if i > 0 and i % 50000 == 0:
                print(f"Progresso: {i / n_combs:.1%} ({i}/{n_combs})", end="\r")

        # Statistiche finali
        metrics_storage = np.asarray(metrics_storage)
        means = metrics_storage.mean(axis=0)
        stds = metrics_storage.std(axis=0)

        metric_names = [
            "accuracy", "error", "sensitivity",
            "specificity", "precision", "f1", "gmean"
        ]

        summary = {
                      f"{name}_mean": float(means[i])
                      for i, name in enumerate(metric_names)
                  } | {
                      f"{name}_std": float(stds[i])
                      for i, name in enumerate(metric_names)
                  }

        return {
            "summary": summary,
            "aggregated_cm": aggregated_cm,
            "n_iterations": n_combs
        }
