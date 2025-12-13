# validation/holdout.py

import numpy as np
from .base import ValidationStrategy
from knn.classifier import KNNClassifier  # Assunto che il classificatore sia accessibile


# ====================================================================
# METRICA DI DEBUG PROVVISORIA (DA RIMUOVERE)
# ====================================================================
def _debug_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcola l'accuratezza, da sostituire con il modulo metrics di Alessandro."""
    return np.mean(y_true == y_pred)


# ====================================================================


class HoldoutValidation(ValidationStrategy):
    """
    Implementazione della strategia di validazione Holdout.
    Esegue un singolo split del dataset in Train e Test.
    """

    def __init__(self, test_size: float = 0.2, random_state: int | None = None):
        """
        Inizializza la strategia Holdout con validazione dei parametri.
        """
        if not (0.0 < test_size < 1.0):
            raise ValueError("Il parametro test_size deve essere compreso tra 0.0 e 1.0.")

        self.test_size = test_size
        self.rng = np.random.default_rng(random_state)

    def validate(self, model: KNNClassifier, X: np.ndarray, y: np.ndarray) -> float:
        """
        Esegue la validazione Holdout: split, training, prediction e valutazione.
        """
        n_samples = X.shape[0]

        # Controllo che il modello sia un'istanza di KNNClassifier
        if not isinstance(model, KNNClassifier):
            raise TypeError("Il parametro 'model' deve essere un'istanza di KNNClassifier.")

        # Calcolo del numero di campioni per il test set
        n_test = int(n_samples * self.test_size)

        # Shuffle degli indici
        indices = self.rng.permutation(n_samples)

        # Split degli indici
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        # Creazione dei set di dati effettivi
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        # Training
        model.fit(X_train, y_train)

        # Prediction sul Test Set
        y_pred = model.predict(X_test)

        # Valutazione (Uso della metrica provvisoria)
        score = _debug_accuracy_score(y_test, y_pred)

        return score
