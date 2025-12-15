from collections import defaultdict
import numpy as np
from src.validation.base import ValidationStrategy
from src.metrics import evaluate_metrics, confusion_counts


class KFoldValidation(ValidationStrategy):
    def __init__(self, n_splits: int = 5, random_state: int | None = None):
        if n_splits < 2:
            raise ValueError("n_splits deve essere almeno 2.")
        self.n_splits = n_splits
        self.rng = np.random.default_rng(random_state)

    def validate(self, model, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples = X.shape[0]

        # Creazione e mescolamento indici
        indices = np.arange(n_samples)
        self.rng.shuffle(indices)

        # Divisione in k fold
        folds = np.array_split(indices, self.n_splits)

        metrics_history = defaultdict(list)

        # Matrice di confusione cumulativa
        aggregated_cm = None

        print(f"Avvio K-Fold ({self.n_splits} splits)...")

        for i in range(self.n_splits):
            # Split indici
            test_idx = folds[i]
            train_fold_list = [folds[j] for j in range(self.n_splits) if j != i]
            train_idx = np.concatenate(train_fold_list)

            # Split dati
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # --- Training & Inference ---
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Probabilità per AUC
            y_probs = model.predict_proba(X_test)
            y_score = y_probs[:, 1]

            # --- Valutazione Metriche ---
            fold_metrics = evaluate_metrics(
                y_true=y_test,
                y_pred=y_pred,
                y_score=y_score,
                metrics=["accuracy", "error", "sensitivity", "specificity",
                         "precision", "f1", "gmean", "auc"]
            )

            # Accumulo Metriche Scalari
            for name, value in fold_metrics.items():
                metrics_history[name].append(value)

            # Accumulo Matrice di Confusione
            c = confusion_counts(y_test, y_pred)
            cm_fold = np.array([[c.tp, c.fn], [c.fp, c.tn]])

            if aggregated_cm is None:
                aggregated_cm = cm_fold
            else:
                aggregated_cm += cm_fold

        summary = {}
        for name, values in metrics_history.items():
            summary[f"{name}_mean"] = float(np.mean(values))
            summary[f"{name}_std"] = float(np.std(values))

        return {
            "summary": summary,
            "raw_metrics": dict(metrics_history),
            "aggregated_cm": aggregated_cm,
            "folds_indices": folds
        }
