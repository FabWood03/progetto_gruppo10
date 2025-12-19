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
    # =========================
    # Costruttore
    # =========================

    def test_invalid_n_splits_raises(self):
        with self.assertRaises(ValueError):
            KFoldValidation(n_splits=1)

    # =========================
    # Struttura output
    # =========================

    def test_kfold_output_structure(self):
        validator = KFoldValidation(n_splits=4, random_state=42)
        model = DummyModel()

        result = validator.validate(model, self.X, self.y)

        self.assertIn("summary", result)
        self.assertIn("raw_metrics", result)
        self.assertIn("aggregated_cm", result)
        self.assertIn("folds_indices", result)

    # =========================
    # Metriche aggregate
    # =========================

    def test_summary_contains_mean_and_std(self):
        validator = KFoldValidation(n_splits=5, random_state=0)
        model = DummyModel()

        result = validator.validate(model, self.X, self.y)
        summary = result["summary"]

        for metric in ["accuracy", "error", "sensitivity",
                       "specificity", "precision", "f1", "gmean", "auc"]:
            self.assertIn(f"{metric}_mean", summary)
            self.assertIn(f"{metric}_std", summary)