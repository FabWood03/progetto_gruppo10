import unittest
import numpy as np

from src.validation.holdout import HoldoutValidation


class DummyModel:
    """
    Modello fittizio per testare la validazione.
    Implementa solo l'interfaccia necessaria.
    """

    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))

    def predict(self, X):
        # Predice sempre 0
        return np.zeros(X.shape[0], dtype=int)

    def predict_proba(self, X):
        # Probabilità fissa [0.7, 0.3]
        return np.tile([0.7, 0.3], (X.shape[0], 1))
