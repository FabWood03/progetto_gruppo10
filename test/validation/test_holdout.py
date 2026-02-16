import unittest
import numpy as np

from src.validation.holdout import HoldoutValidation


"""
Questo file testa la strategia di validazione Holdout.

Obiettivi:
1) Verificare corretto split train/test
2) Verificare integrazione modello + preprocessing
3) Verificare struttura output
4) Verificare correttezza metriche
5) Garantire riproducibilità
"""


# =========================================================
# MODELLO FITTIZIO
# =========================================================

class DummyModel:
    """
    Modello fittizio usato solo per test.

    Implementa:
    - fit
    - predict
    - predict_proba

    Serve per isolare la logica della validazione
    dalla complessità del modello reale.
    """

    def fit(self, X, y):
        # Memorizzo solo il numero di classi
        self.n_classes_ = len(np.unique(y))

    def predict(self, X):
        # Predice sempre classe 0
        return np.zeros(X.shape[0], dtype=int)

    def predict_proba(self, X):
        # Restituisce probabilità costanti
        return np.tile([0.7, 0.3], (X.shape[0], 1))


# =========================================================
# TEST HOLDOUT VALIDATION
# =========================================================

class TestHoldoutValidation(unittest.TestCase):

    def setUp(self):
        """
        Genero dataset sintetico casuale ma riproducibile.
        Inserisco NaN per testare imputazione.
        """
        rng = np.random.default_rng(42)

        self.X = rng.normal(size=(50, 5))
        self.y = rng.integers(0, 2, size=50)

        # Inserisco NaN per verificare imputazione
        self.X[0, 0] = np.nan
        self.X[1, 1] = np.nan

    # =========================
    # COSTRUTTORE
    # =========================

    def test_invalid_test_size_raises(self):
        """
        test_size deve essere tra 0 e 1 (esclusi).
        Verifico che valori estremi generino errore.
        """
        with self.assertRaises(ValueError):
            HoldoutValidation(test_size=0.0)

        with self.assertRaises(ValueError):
            HoldoutValidation(test_size=1.0)

    # =========================
    # STRUTTURA OUTPUT
    # =========================

    def test_holdout_validation_output_structure(self):
        """
        Verifico che la funzione validate restituisca
        tutte le chiavi attese.
        """
        validator = HoldoutValidation(test_size=0.2, random_state=42)
        model = DummyModel()

        result = validator.validate(model, self.X, self.y)

        # Struttura completa output
        self.assertIn("metrics", result)
        self.assertIn("roc_data", result)
        self.assertIn("confusion_matrix", result)
        self.assertIn("y_test", result)
        self.assertIn("y_pred", result)

    # =========================
    # SHAPE E COERENZA
    # =========================

    def test_holdout_shapes_are_consistent(self):
        """
        Verifico coerenza dimensionale:
        - y_test e y_pred stessa lunghezza
        - matrice di confusione 2x2
        """
        validator = HoldoutValidation(test_size=0.3, random_state=1)
        model = DummyModel()

        result = validator.validate(model, self.X, self.y)

        y_test = result["y_test"]
        y_pred = result["y_pred"]
        cm = result["confusion_matrix"]

        self.assertEqual(len(y_test), len(y_pred))
        self.assertEqual(cm.shape, (2, 2))

    # =========================
    # METRICHE PRESENTI
    # =========================

    def test_metrics_keys_present(self):
        """
        Verifico che tutte le metriche principali
        siano calcolate e presenti nel dizionario.
        """
        validator = HoldoutValidation(test_size=0.25, random_state=0)
        model = DummyModel()

        result = validator.validate(model, self.X, self.y)
        metrics = result["metrics"]

        for key in ["accuracy", "error", "sensitivity",
                    "specificity", "precision", "f1", "gmean", "auc"]:
            self.assertIn(key, metrics)

    # =========================
    # RIPRODUCIBILITÀ
    # =========================

    def test_holdout_is_reproducible(self):
        """
        Con stesso random_state,
        lo split deve essere identico.
        Questo garantisce riproducibilità scientifica.
        """
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
