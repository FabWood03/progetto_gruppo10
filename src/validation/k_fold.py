import numpy as np
from collections import defaultdict
from src.metrics import evaluate_metrics, confusion_counts

from src.validation.base import ValidationStrategy
from src.validation.base import minmax_scale_train_test
from src.validation.base import median_impute_train_test


# from src.validation.base import ValidationStrategy, minmax_scale_train_test


class KFoldValidation(ValidationStrategy):
    """
    Implementazione della strategia K-Fold Cross-Validation ottimizzata.
    Divide il dataset in k fold, addestra il modello su k-1 fold e
    valuta le prestazioni sul fold rimanente, ripetendo per tutti i fold.
    1. Addestra il modello memorizzando i dati di training.
    2. Durante l'inference, calcola le distanze in modo vettorializzato
       per tutti i campioni di test in un singolo passaggio.
    3. Accumula le metriche per ogni fold e calcola la media e deviazione standard
       alla fine.
    """

    def __init__(self, n_splits: int = 5, random_state: int | None = None):
        if n_splits < 2:
            raise ValueError("n_splits deve essere almeno 2.")
        self.n_splits = n_splits
        self.rng = np.random.default_rng(random_state)

    def validate(self, model, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples = X.shape[0]

        # 1. GENERAZIONE FOLD
        indices = np.arange(n_samples)
        self.rng.shuffle(indices)
        folds = np.array_split(indices, self.n_splits)

        metrics_history = defaultdict(list)
        aggregated_cm = np.zeros((2, 2), dtype=np.int64)

        # 2. LOOP SUI FOLD
        for i in range(self.n_splits):
            test_idx = folds[i]

            mask = np.ones(n_samples, dtype=bool)
            mask[test_idx] = False

            train_idx = np.where(mask)[0]

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Imputazione dei NaN (solo sul training del fold)
            X_train, X_test = median_impute_train_test(X_train, X_test)

            # Normalizzazione corretta (solo sul training del fold)
            X_train, X_test = minmax_scale_train_test(X_train, X_test)

            # TRAINING
            model.fit(X_train, y_train)

            # INFERENCE
            y_pred = model.predict(X_test)
            probas = model.predict_proba(X_test)

            # Estrazione score per classe positiva (1)
            y_score = probas[:, 1] if probas.ndim > 1 and probas.shape[1] > 1 else probas.flatten()

            # 3. VALUTAZIONE
            fold_metrics = evaluate_metrics(
                y_true=y_test,
                y_pred=y_pred,
                y_score=y_score,
                metrics=["accuracy", "error", "sensitivity", "specificity",
                         "precision", "f1", "gmean", "auc"]
            )

            # Accumulo metriche
            for name, value in fold_metrics.items():
                metrics_history[name].append(value)

            # Accumulo matrice di confusione
            c = confusion_counts(y_test, y_pred)
            aggregated_cm += np.array([[c.tp, c.fn], [c.fp, c.tn]])

        # 4. AGGREGAZIONE STATISTICA
        summary = {
            f"{name}_mean": float(np.mean(values)) for name, values in metrics_history.items()
        }
        summary.update({
            f"{name}_std": float(np.std(values)) for name, values in metrics_history.items()
        })

        return {
            "summary": summary,
            "raw_metrics": dict(metrics_history),
            "aggregated_cm": aggregated_cm,
            "folds_indices": folds
        }
