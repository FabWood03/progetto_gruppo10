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

    # =========================
    # Numero iterazioni corretto
    # =========================

    def test_number_of_iterations(self):
        p = 2
        validator = LeavePOutValidation(p=p)
        model = DummyModel()

        result = validator.validate(model, self.X, self.y)

        expected = math.comb(len(self.X), p)
        self.assertEqual(result["n_iterations"], expected)

    # =========================
    # Struttura output
    # =========================

    def test_output_structure(self):
        validator = LeavePOutValidation(p=1)
        model = DummyModel()

        result = validator.validate(model, self.X, self.y)

        self.assertIn("summary", result)
        self.assertIn("raw_metrics", result)
        self.assertIn("aggregated_cm", result)
        self.assertIn("n_iterations", result)

    # =========================
    # Shape confusion matrix
    # =========================
    def test_aggregated_confusion_matrix_shape(self):
        validator = LeavePOutValidation(p=1)
        model = DummyModel()

        result = validator.validate(model, self.X, self.y)
        cm = result["aggregated_cm"]

        self.assertEqual(cm.shape, (2, 2))

    # =========================
    # Metriche summary presenti
    # =========================
    def test_summary_metrics_keys(self):
        validator = LeavePOutValidation(p=1)
        model = DummyModel()

        result = validator.validate(model, self.X, self.y)
        summary = result["summary"]

        metrics = [
            "accuracy", "error", "sensitivity",
            "specificity", "precision", "f1", "gmean"
        ]

        for m in metrics:
            self.assertIn(f"{m}_mean", summary)
            self.assertIn(f"{m}_std", summary)

    # =========================
    # Lunghezza raw metrics
    # =========================
    def test_raw_metrics_length(self):
        p = 1
        validator = LeavePOutValidation(p=p)
        model = DummyModel()

        result = validator.validate(model, self.X, self.y)
        n_iters = result["n_iterations"]

        for values in result["raw_metrics"].values():
            self.assertEqual(len(values), n_iters)

    # =========================
    # No data leakage
    # =========================
    def test_no_data_leakage(self):
        p = 3
        validator = LeavePOutValidation(p=p)
        model = DummyModel()

        validator.validate(model, self.X, self.y)

        # Ogni fit deve usare n - p campioni
        self.assertEqual(model.X_train_.shape[0], len(self.X) - p)

    # =========================
    # Stabilità numerica
    # =========================
    def test_no_nan_in_summary(self):
        validator = LeavePOutValidation(p=1)
        model = DummyModel()

        result = validator.validate(model, self.X, self.y)

        for value in result["summary"].values():
            self.assertFalse(np.isnan(value))


if __name__ == "__main__":
    unittest.main()