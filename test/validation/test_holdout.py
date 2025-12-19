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
 # =========================
    # Shape e coerenza
    # =========================
    def test_holdout_shapes_are_consistent(self):
        validator = HoldoutValidation(test_size=0.3, random_state=1)
        model = DummyModel()

        result = validator.validate(model, self.X, self.y)

        y_test = result["y_test"]
        y_pred = result["y_pred"]
        cm = result["confusion_matrix"]

        self.assertEqual(len(y_test), len(y_pred))
        self.assertEqual(cm.shape, (2, 2))

    # =========================
    # Metriche presenti
    # =========================
    def test_metrics_keys_present(self):
        validator = HoldoutValidation(test_size=0.25, random_state=0)
        model = DummyModel()

        result = validator.validate(model, self.X, self.y)
        metrics = result["metrics"]

        for key in ["accuracy", "error", "sensitivity",
                    "specificity", "precision", "f1", "gmean", "auc"]:
            self.assertIn(key, metrics)

    # =========================
    # Riproducibilità
    # =========================
    def test_holdout_is_reproducible(self):
        model1 = DummyModel()
        model2 = DummyModel()

        validator1 = HoldoutValidation(test_size=0.2, random_state=42)
        validator2 = HoldoutValidation(test_size=0.2, random_state=42)

        res1 = validator1.validate(model1, self.X, self.y)
        res2 = validator2.validate(model2, self.X, self.y)

        np.testing.assert_array_equal(res1["y_test"], res2["y_test"])
        np.testing.assert_array_equal(res1["y_pred"], res2["y_pred"])


if __name__ == "__main__":
    unittest.main()
