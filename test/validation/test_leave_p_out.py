import unittest
import numpy as np
import math

from src.validation.leave_p_out import LeavePOutValidation


"""
Questo file testa la strategia Leave-P-Out.

Obiettivi:
1) Verificare correttezza parametro p
2) Verificare numero combinazioni (n choose p)
3) Verificare struttura output
4) Verificare assenza di data leakage
5) Verificare stabilità numerica delle metriche
"""


# =========================================================
# MODELLO FITTIZIO
# =========================================================

class DummyModel:
    """
    Modello minimale per isolare la logica
    della validazione Leave-P-Out.

    Predice sempre classe 0.
    """

    def fit(self, X, y):
        # Memorizzo training usato
        self.X_train_ = np.array(X)
        self.y_train_ = np.array(y)

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


# =========================================================
# TEST LEAVE-P-OUT
# =========================================================

class TestLeavePOutValidation(unittest.TestCase):

    def setUp(self):
        """
        Dataset sintetico riproducibile.
        """
        rng = np.random.default_rng(42)

        self.X = rng.normal(size=(10, 4))
        self.y = rng.integers(0, 2, size=10)

    # =========================
    # COSTRUTTORE
    # =========================

    def test_invalid_p_raises(self):
        """
        p deve essere >= 1.
        """
        with self.assertRaises(ValueError):
            LeavePOutValidation(p=0)

    # =========================
    # p >= n_samples
    # =========================

    def test_p_greater_equal_n_samples_raises(self):
        """
        Non ha senso lasciare fuori tutti i campioni.
        """
        validator = LeavePOutValidation(p=10)
        model = DummyModel()

        with self.assertRaises(ValueError):
            validator.validate(model, self.X, self.y)

    # =========================
    # NUMERO ITERAZIONI
    # =========================

    def test_number_of_iterations(self):
        """
        Leave-P-Out genera C(n, p) combinazioni.
        Verifico correttezza formula combinatoria.
        """
        p = 2
        validator = LeavePOutValidation(p=p)
        model = DummyModel()

        result = validator.validate(model, self.X, self.y)

        expected = math.comb(len(self.X), p)
        self.assertEqual(result["n_iterations"], expected)

    # =========================
    # STRUTTURA OUTPUT
    # =========================

    def test_output_structure(self):
        """
        Verifico che l'output contenga:
        - summary (metriche aggregate)
        - raw_metrics (metriche per iterazione)
        - aggregated_cm (matrice cumulativa)
        - n_iterations
        """
        validator = LeavePOutValidation(p=1)
        model = DummyModel()

        result = validator.validate(model, self.X, self.y)

        self.assertIn("summary", result)
        self.assertIn("raw_metrics", result)
        self.assertIn("aggregated_cm", result)
        self.assertIn("n_iterations", result)

    # =========================
    # MATRICE CONFUSIONE
    # =========================

    def test_aggregated_confusion_matrix_shape(self):
        """
        Deve essere sempre 2x2.
        """
        validator = LeavePOutValidation(p=1)
        model = DummyModel()

        result = validator.validate(model, self.X, self.y)
        cm = result["aggregated_cm"]

        self.assertEqual(cm.shape, (2, 2))

    # =========================
    # METRICHE SUMMARY
    # =========================

    def test_summary_metrics_keys(self):
        """
        Verifico presenza mean e std
        per tutte le metriche principali.
        """
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
    # RAW METRICS
    # =========================

    def test_raw_metrics_length(self):
        """
        Ogni metrica deve avere
        una misura per ogni iterazione.
        """
        p = 1
        validator = LeavePOutValidation(p=p)
        model = DummyModel()

        result = validator.validate(model, self.X, self.y)
        n_iters = result["n_iterations"]

        for values in result["raw_metrics"].values():
            self.assertEqual(len(values), n_iters)

    # =========================
    # NO DATA LEAKAGE
    # =========================

    def test_no_data_leakage(self):
        """
        In ogni iterazione il training deve avere n - p campioni.
        Serve a garantire che i campioni di test
        non entrino mai nel training.
        """
        p = 3
        validator = LeavePOutValidation(p=p)
        model = DummyModel()

        validator.validate(model, self.X, self.y)

        self.assertEqual(model.X_train_.shape[0], len(self.X) - p)

    # =========================
    # STABILITÀ NUMERICA
    # =========================

    def test_no_nan_in_summary(self):
        """
        Verifico che nessuna metrica aggregata
        produca NaN.
        """
        validator = LeavePOutValidation(p=1)
        model = DummyModel()

        result = validator.validate(model, self.X, self.y)

        for value in result["summary"].values():
            self.assertFalse(np.isnan(value))


if __name__ == "__main__":
    unittest.main()
