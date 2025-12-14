import numpy as np

from src.metrics.evaluator import (
    confusion_counts,
    roc_curve_manual,
    calculate_auc,
    evaluate_metrics,
)
from .base import ValidationStrategy


class HoldoutValidation(ValidationStrategy):
    """
    Implementazione della strategia Holdout.
    Divide il dataset in un set di addestramento e uno di test,
    addestra il modello sul set di addestramento e valuta
    le prestazioni sul set di test.
    """

    def __init__(self, test_size: float = 0.2, random_state: int | None = None):
        if not (0.0 < test_size < 1.0):
            raise ValueError("Il parametro test_size deve essere compreso tra 0.0 e 1.0.")

        self.test_size = test_size
        self.rng = np.random.default_rng(random_state)

    def validate(self, model, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Esegue la validazione Holdout e restituisce i risultati grezzi.
        :return: Un dizionario contenente metriche, dati ROC e matrice di confusione.
        """
        # --- 1. PREPARAZIONE DATI ---
        X = np.asarray(X)
        y = np.asarray(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X e y hanno dimensioni diverse: {X.shape[0]} vs {y.shape[0]}")

        # Split Train/Test
        n_samples = X.shape[0]
        n_test = int(n_samples * self.test_size)
        indices = self.rng.permutation(n_samples)

        train_idx, test_idx = indices[n_test:], indices[:n_test]
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # --- 2. TRAINING & INFERENCE ---
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_probs = model.predict_proba(X_test)
        y_score = y_probs[:, 1]

        # --- 3. CALCOLO METRICHE ---
        # Calcolo curve ROC
        fpr, tpr, _ = roc_curve_manual(y_test, y_score)
        auc_value = calculate_auc(fpr, tpr)

        # Calcolo tutte le metriche scalari
        metrics = evaluate_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_score=y_score,
            metrics=["accuracy", "error", "sensitivity", "specificity",
                     "precision", "f1", "gmean"]
        )

        metrics["auc"] = auc_value

        # Dati per la matrice di confusione
        # Ricaviamo i conteggi raw per poterli plottare fuori
        c = confusion_counts(y_test, y_pred)
        cm_matrix = np.array([[c.tp, c.fn], [c.fp, c.tn]])

        return {
            "metrics": metrics,
            "roc_data": (fpr, tpr),
            "confusion_matrix": cm_matrix,
            "y_test": y_test,
            "y_pred": y_pred
        }
