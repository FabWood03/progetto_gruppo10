import numpy as np
from src.validation.base import minmax_scale_train_test

from src.metrics.evaluator import (
    confusion_counts,
    roc_curve_manual,
    calculate_auc,
    evaluate_metrics,
)
from .base import ValidationStrategy


class HoldoutValidation(ValidationStrategy):
    """
    Implementazione della strategia Holdout ottimizzata.
    Divide il dataset in un training set e un test set basato su una
    proporzione specificata. Addestra il modello sul training set e
    valuta le prestazioni sul test set.
    1. Addestra il modello memorizzando i dati di training.
    2. Durante l'inference, calcola le distanze in modo vettorializzato
         per tutti i campioni di test in un singolo passaggio.
    3. Calcola e restituisce le metriche di valutazione.
    """

    def __init__(self, test_size: float = 0.2, random_state: int | None = None):
        if not (0.0 < test_size < 1.0):
            raise ValueError("Il parametro test_size deve essere compreso tra 0.0 e 1.0.")

        self.test_size = test_size
        self.rng = np.random.default_rng(random_state)

    def validate(self, model, X: np.ndarray, y: np.ndarray) -> dict:
        """
            Esegue la validazione Holdout.

            Il dataset viene suddiviso in training e test set.
            La normalizzazione delle feature è applicata esclusivamente
            sui dati di training per evitare il data leakage.

            :param model: Modello KNN da validare.
            :param X: Matrice delle feature.
            :param y: Vettore delle etichette.
            :return: Dizionario con metriche e dati di valutazione.
         """
        # 1. PREPARAZIONE DATI
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples = X.shape[0]
        n_test = int(n_samples * self.test_size)
        indices = self.rng.permutation(n_samples)
        train_idx, test_idx = indices[n_test:], indices[:n_test]

        # 2. TRAINING & INFERENCE

        # Estrazione training e test set
        X_train = X[train_idx]
        y_train = y[train_idx]

        X_test = X[test_idx]
        y_test = y[test_idx]

        # Normalizzazione corretta (solo sul training)
        X_train, X_test = minmax_scale_train_test(X_train, X_test)

        # Il fit memorizza il training set normalizzato
        model.fit(X_train, y_train)

        # 3. INFERENCE
        y_pred = model.predict(X_test)

        # Otteniamo le probabilità per l'AUC
        probas = model.predict_proba(X_test)

        # Estraiamo lo score per la classe positiva (1).
        if probas.ndim > 1 and probas.shape[1] > 1:
            y_score = probas[:, 1]
        else:
            y_score = probas.flatten()

        # 4. CALCOLO METRICHE DI VALUTAZIONE
        fpr, tpr, _ = roc_curve_manual(y_test, y_score)
        auc_value = calculate_auc(fpr, tpr)

        metrics = evaluate_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_score=y_score,
            metrics=["accuracy", "error", "sensitivity", "specificity",
                     "precision", "f1", "gmean"]
        )
        metrics["auc"] = auc_value

        c = confusion_counts(y_test, y_pred)
        cm_matrix = np.array([[c.tp, c.fn], [c.fp, c.tn]])

        return {
            "metrics": metrics,
            "roc_data": (fpr, tpr),
            "confusion_matrix": cm_matrix,
            "y_test": y_test,
            "y_pred": y_pred
        }
