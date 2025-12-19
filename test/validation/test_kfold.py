import unittest
import numpy as np

from src.validation.k_fold import KFoldValidation


class DummyModel:
    """
    Modello fittizio per testare la K-Fold validation.
    """

    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))

    def predict(self, X):
        return np.zeros(X.shape[0], dtype=int)

    def predict_proba(self, X):
        return np.tile([0.6, 0.4], (X.shape[0], 1))


class TestKFoldValidation(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(123)
        self.X = rng.normal(size=(40, 6))
        self.y = rng.integers(0, 2, size=40)

        # Inseriamo NaN per testare imputazione
        self.X[0, 0] = np.nan
        self.X[1, 1] = np.nan
