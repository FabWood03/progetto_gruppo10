import unittest
import numpy as np
import math

from src.validation.leave_p_out import LeavePOutValidation


class DummyModel:
    """
    Modello fittizio per testare Leave-P-Out.
    Predice sempre 0, fit minimale.
    """

    def fit(self, X, y):
        self.X_train_ = np.array(X)
        self.y_train_ = np.array(y)

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class TestLeavePOutValidation(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(42)

        self.X = rng.normal(size=(10, 4))
        self.y = rng.integers(0, 2, size=10)

    # =========================
    # Costruttore
    # =========================
    def test_invalid_p_raises(self):
        with self.assertRaises(ValueError):
            LeavePOutValidation(p=0)

    # =========================
    # p >= n_samples
    # =========================

    def test_p_greater_equal_n_samples_raises(self):
        validator = LeavePOutValidation(p=10)
        model = DummyModel()

        with self.assertRaises(ValueError):
            validator.validate(model, self.X, self.y)