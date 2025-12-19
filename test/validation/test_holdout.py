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

class TestHoldoutValidation(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(42)

        self.X = rng.normal(size=(50, 5))
        self.y = rng.integers(0, 2, size=50)

        # Inseriamo NaN per testare imputazione
        self.X[0, 0] = np.nan
        self.X[1, 1] = np.nan

    # =========================
    # Costruttore
    # =========================
    def test_invalid_test_size_raises(self):
        with self.assertRaises(ValueError):
            HoldoutValidation(test_size=0.0)

        with self.assertRaises(ValueError):
            HoldoutValidation(test_size=1.0)

    # =========================
    # Validazione base
    # =========================
    def test_holdout_validation_output_structure(self):
        validator = HoldoutValidation(test_size=0.2, random_state=42)
        model = DummyModel()

        result = validator.validate(model, self.X, self.y)

        # Chiavi attese
        self.assertIn("metrics", result)
        self.assertIn("roc_data", result)
        self.assertIn("confusion_matrix", result)
        self.assertIn("y_test", result)
        self.assertIn("y_pred", result)
