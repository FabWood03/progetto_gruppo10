import math
import time
import utils
from itertools import combinations

import numpy as np

from metrics import (
    confusion_counts,
    accuracy_rate,
    error_rate,
    sensitivity,
    specificity,
    precision,
    f1_score,
    geometric_mean
)
from validation.base import ValidationStrategy


class LeavePOutValidation(ValidationStrategy):
    """
    Validazione Leave-P-Out (LpO).
    Testa tutte le possibili combinazioni di 'p' campioni come test set.
    """

    def __init__(self, p: int = 1):
        if p < 1:
            raise ValueError("p deve essere almeno 1.")
        self.p = p

    def validate(self, model, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples = X.shape[0]

        if self.p >= n_samples:
            raise ValueError(f"p ({self.p}) deve essere < {n_samples}.")

        # Calcolo Combinazioni
        n_combs = math.comb(n_samples, self.p)
        print(f"Campioni: {n_samples} | Iterazioni Totali: {n_combs}")

        if n_combs > 2000000:
            print(f"⚠️ ATTENZIONE: {n_combs} iterazioni sono moltissime.")

        # PRE-ALLOCAZIONE MEMORIA
        metrics_storage = np.zeros((n_combs, 7), dtype=np.float32)

        # Matrice di confusione aggregata
        aggregated_cm = np.zeros((2, 2), dtype=np.int64)

        all_indices = np.arange(n_samples)

        # Maschera booleana per slicing veloce
        mask = np.ones(n_samples, dtype=bool)

        start_time = time.time()

        # Passo per stimare il tempo
        benchmark_steps = 50

        # Iteratore combinazioni
        comb_iter = combinations(all_indices, self.p)

        # --- LOOP PRINCIPALE ---
        for i, test_idx_tuple in enumerate(comb_iter):

            # Gestione Indici
            mask[:] = True
            mask[list(test_idx_tuple)] = False

            # Training & Inference
            model.fit(X[mask], y[mask])
            y_pred = model.predict(X[~mask])
            y_test_fold = y[~mask]

            # Calcolo Metriche
            c = confusion_counts(y_test_fold, y_pred, pos_label=1)

            # Aggiorno la CM aggregata
            aggregated_cm[0, 0] += c.tp
            aggregated_cm[0, 1] += c.fn
            aggregated_cm[1, 0] += c.fp
            aggregated_cm[1, 1] += c.tn

            # Calcolo delle metriche singole
            metrics_storage[i, 0] = accuracy_rate(c)
            metrics_storage[i, 1] = error_rate(c)
            metrics_storage[i, 2] = sensitivity(c)
            metrics_storage[i, 3] = specificity(c)
            metrics_storage[i, 4] = precision(c)
            metrics_storage[i, 5] = f1_score(c)
            metrics_storage[i, 6] = geometric_mean(c)

            # --- Benchmark e Stima Tempo ---
            if i == benchmark_steps:
                elapsed = time.time() - start_time
                avg_iter = elapsed / benchmark_steps
                est_total = avg_iter * n_combs

                print(f"\n⏱️  Tempo medio: {avg_iter * 1000:.2f} ms/iter")
                print(f"⏳ STIMA TOTALE: {utils.format_duration(est_total)}")
                print("-" * 30)

            if i > 0 and i % 5000 == 0:
                print(f"Progresso: {i / n_combs:.1%} ({i}/{n_combs})", end="\r")

        print("\nCalcolo statistiche finali...")

        # Calcolo medie e deviazioni standard sulle colonne
        means = np.mean(metrics_storage, axis=0)
        stds = np.std(metrics_storage, axis=0)

        metric_names = ["accuracy", "error", "sensitivity", "specificity", "precision", "f1", "gmean"]
        summary = {}
        raw_metrics = {}

        for idx, name in enumerate(metric_names):
            summary[f"{name}_mean"] = float(means[idx])
            summary[f"{name}_std"] = float(stds[idx])
            raw_metrics[name] = metrics_storage[:, idx].tolist()

        return {
            "summary": summary,
            "raw_metrics": raw_metrics,
            "aggregated_cm": aggregated_cm,
            "n_iterations": n_combs
        }
