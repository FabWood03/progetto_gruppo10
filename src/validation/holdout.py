import numpy as np

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

    def validate(self, model, X: np.ndarray, y: np.ndarray) -> float:
        """
        Esegue la validazione Holdout.
        :param model: Il modello da validare (deve avere metodi .fit() e .predict()).
        :param X: I dati di input (array 2D).
        :param y: Le etichette di output (array 1D).
        :return: Metriche di valutazione.
        """
        # Conversione sicura in array NumPy
        X = np.asarray(X)
        y = np.asarray(y)

        # Controllo integrità dimensioni
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Errore dimensioni: X ha {X.shape[0]} righe, y ne ha {y.shape[0]}.")

        # Controllo validità modello
        if not (hasattr(model, "fit") and hasattr(model, "predict")):
            raise TypeError("Il modello fornito non è valido: mancano i metodi .fit() o .predict()")

        n_samples = X.shape[0]
        n_test = int(n_samples * self.test_size)

        # Shuffle efficiente
        indices = self.rng.permutation(n_samples)

        # Split e Creazione dataset
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Pipeline esecuzione
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calcolo metriche
        return float(np.mean(y_test == y_pred))
