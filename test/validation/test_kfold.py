import unittest
import numpy as np

from src.validation.k_fold import KFoldValidation


"""
Questo file testa la strategia di validazione K-Fold.

Obiettivi:
1) Verificare corretto numero di fold
2) Verificare struttura output
3) Verificare aggregazione metriche (mean/std)
4) Verificare aggregazione matrice di confusione
5) Garantire riproducibilità tramite random_state
"""


# =========================================================
# MODELLO FITTIZIO
# =========================================================

class DummyModel:
    """
    Modello fittizio usato per isolare la logica della K-Fold.
    Non interessa la qualità del modello,
    ma la correttezza della validazione.
    """

    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))

    def predict(self, X):
        # Predizione costante per semplicità
        return np.zeros(X.shape[0], dtype=int)

    def predict_proba(self, X):
        # Probabilità fisse per testare AUC
        return np.tile([0.6, 0.4], (X.shape[0], 1))


# =========================================================
# TEST KFOLD VALIDATION
# =========================================================

class TestKFoldValidation(unittest.TestCase):

    def setUp(self):
        """
        Dataset sintetico riproducibile.
        Inserisco NaN per verificare imputazione
        all'interno di ciascun fold.
        """
        rng = np.random.default_rng(123)
        self.X = rng.normal(size=(40, 6))
        self.y = rng.integers(0, 2, size=40)

        # Inserisco NaN per testare preprocessing
        self.X[0, 0] = np.nan
        self.X[1, 1] = np.nan

    # =========================
    # COSTRUTTORE
    # =========================

    def test_invalid_n_splits_raises(self):
        """
        n_splits deve essere almeno 2.
        Verifico che n_splits=1 generi errore.
        """
        with self.assertRaises(ValueError):
            KFoldValidation(n_splits=1)

    # =========================
    # STRUTTURA OUTPUT
    # =========================

    def test_kfold_output_structure(self):
        """
        Verifico che l'output contenga:
        - summary (metriche aggregate)
        - raw_metrics (metriche per fold)
        - aggregated_cm (matrice cumulativa)
        - folds_indices (indici usati)
        """
        validator = KFoldValidation(n_splits=4, random_state=42)
        model = DummyModel()

        result = validator.validate(model, self.X, self.y)

        self.assertIn("summary", result)
        self.assertIn("raw_metrics", result)
        self.assertIn("aggregated_cm", result)
        self.assertIn("folds_indices", result)

    # =========================
    # METRICHE AGGREGATE
    # =========================

    def test_summary_contains_mean_and_std(self):
        """
        Verifico che per ogni metrica vengano calcolati:
        - media
        - deviazione standard
        Questo è fondamentale per stimare la variabilità.
        """
        validator = KFoldValidation(n_splits=5, random_state=0)
        model = DummyModel()

        result = validator.validate(model, self.X, self.y)
        summary = result["summary"]

        for metric in ["accuracy", "error", "sensitivity",
                       "specificity", "precision", "f1", "gmean", "auc"]:
            self.assertIn(f"{metric}_mean", summary)
            self.assertIn(f"{metric}_std", summary)

    # =========================
    # MATRICE DI CONFUSIONE AGGREGATA
    # =========================

    def test_aggregated_confusion_matrix_shape(self):
        """
        La matrice aggregata deve essere 2x2
        (classificazione binaria).
        """
        validator = KFoldValidation(n_splits=3, random_state=1)
        model = DummyModel()

        result = validator.validate(model, self.X, self.y)
        cm = result["aggregated_cm"]

        self.assertEqual(cm.shape, (2, 2))

    # =========================
    # NUMERO FOLD CORRETTO
    # =========================

    def test_number_of_folds(self):
        """
        Verifico che il numero di fold generati
        sia coerente con n_splits.
        """
        n_splits = 4
        validator = KFoldValidation(n_splits=n_splits, random_state=10)
        model = DummyModel()

        result = validator.validate(model, self.X, self.y)
        folds = result["folds_indices"]

        self.assertEqual(len(folds), n_splits)

    # =========================
    # RIPRODUCIBILITÀ
    # =========================

    def test_kfold_is_reproducible(self):
        """
        Con stesso random_state,
        la suddivisione in fold deve essere identica.
        Questo garantisce riproducibilità scientifica.
        """
        model1 = DummyModel()
        model2 = DummyModel()

        validator1 = KFoldValidation(n_splits=5, random_state=99)
        validator2 = KFoldValidation(n_splits=5, random_state=99)

        res1 = validator1.validate(model1, self.X, self.y)
        res2 = validator2.validate(model2, self.X, self.y)

        for f1, f2 in zip(res1["folds_indices"], res2["folds_indices"]):
            np.testing.assert_array_equal(f1, f2)


if __name__ == "__main__":
    unittest.main()
