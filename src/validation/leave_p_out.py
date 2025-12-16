import math
import time
import numpy as np
from itertools import combinations
import src.utils as utils

from src.metrics import (
    confusion_counts, accuracy_rate, error_rate, sensitivity,
    specificity, precision, f1_score, geometric_mean
)
from src.validation.base import ValidationStrategy


class LeavePOutValidation(ValidationStrategy):
    """
    Implementa la validazione Leave-P-Out (LPO).
    Per ogni combinazione di p campioni, questi vengono usati come test set,
    mentre il resto come training set. Questo viene ripetuto per tutte le
    possibili combinazioni di p campioni.
    """

    def __init__(self, p: int = 1):
        if p < 1:
            raise ValueError("p deve essere almeno 1.")
        self.p = p

    def validate(self, model, X, y):
        """
        Esegue la validazione Leave-P-Out (LPO).

        Le distanze tra tutti i campioni vengono pre-calcolate una sola volta
        su tutto il dataset per ridurre drasticamente la complessità computazionale.
        Per questo motivo, non viene applicata una normalizzazione per ogni split,
        mantenendo coerenza con la matrice globale delle distanze ed evitando
        un costo computazionale proibitivo.

        :param model: Modello KNN da validare.
        :param X: Matrice delle feature.
        :param y: Vettore delle etichette.
        :return: Dizionario con metriche aggregate e dettagliate.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples = X.shape[0]

        if self.p >= n_samples:
            raise ValueError(f"p ({self.p}) deve essere < {n_samples}.")

        n_combs = math.comb(n_samples, self.p)
        print(f"Campioni: {n_samples} | Iterazioni Totali: {n_combs}")

        # 1. PRE-CALCOLO MATRICE
        # Calcoliamo TUTTE le distanze possibili ora, una volta sola.
        print("Calcolo matrice distanze globale (Pre-computing)...")
        start_mat = time.time()

        full_dist_matrix = np.zeros((n_samples, n_samples), dtype=np.float32)

        for i in range(n_samples):
            full_dist_matrix[i, :] = model.distance_metric.calculate(X[i], X)

        print(f"Matrice completata in {time.time() - start_mat:.2f}s")

        # Setup memoria
        metrics_storage = np.zeros((n_combs, 7), dtype=np.float32)
        aggregated_cm = np.zeros((2, 2), dtype=np.int64)

        all_indices = np.arange(n_samples)
        mask = np.ones(n_samples, dtype=bool)

        start_time = time.time()
        benchmark_steps = 200

        # 2. LOOP
        for i, test_idx_tuple in enumerate(combinations(all_indices, self.p)):
            test_indices = list(test_idx_tuple)

            # A. Setup Maschera Train
            mask[:] = True # Reset
            mask[test_indices] = False # Test indices a False

            # B. "Addestramento" Leggero
            # Invece di passare X (pesante), passiamo solo y al modello
            # perché le distanze X le abbiamo già nella matrice
            model.fit(X[mask], y[mask])

            # C. Predizione Veloce
            y_pred = []
            for t_idx in test_indices:
                # 1. SLICING: Estraiamo la riga delle distanze dalla matrice globale
                #    e teniamo solo le colonne che corrispondono al training set corrente
                row_dists = full_dist_matrix[t_idx, mask]

                # 2. PREDICT: Passiamo le distanze pronte al modello
                pred = model.predict_with_distances(row_dists)
                y_pred.append(pred)

            # D. Calcolo Metriche
            y_test_fold = y[test_indices]
            c = confusion_counts(y_test_fold, np.array(y_pred), pos_label=1)

            aggregated_cm[0, 0] += c.tp
            aggregated_cm[0, 1] += c.fn
            aggregated_cm[1, 0] += c.fp
            aggregated_cm[1, 1] += c.tn

            metrics_storage[i, 0] = accuracy_rate(c)
            metrics_storage[i, 1] = error_rate(c)
            metrics_storage[i, 2] = sensitivity(c)
            metrics_storage[i, 3] = specificity(c)
            metrics_storage[i, 4] = precision(c)
            metrics_storage[i, 5] = f1_score(c)
            metrics_storage[i, 6] = geometric_mean(c)

            # E. Feedback
            if i == benchmark_steps:
                elapsed = time.time() - start_time
                avg_iter = elapsed / benchmark_steps
                est_total = avg_iter * n_combs
                print(f"\nVELOCITÀ: {avg_iter * 1000:.3f} ms/iter")
                print(f"STIMA COMPLETAMENTO: {utils.format_duration(est_total)}")
                print("-" * 30)

            if i > 0 and i % 50000 == 0:
                print(f"Progresso: {i / n_combs:.1%} ({i}/{n_combs})", end="\r")

        print("\nCalcolo statistiche finali...")

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
