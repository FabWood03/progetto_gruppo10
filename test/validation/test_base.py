import unittest
import numpy as np

from src.validation.base import (
    minmax_scale_train_test,
    median_impute_train_test,
    ValidationStrategy
)


"""
Questo file testa le funzioni di preprocessing utilizzate
nelle strategie di validazione.

Obiettivi:
1) Garantire assenza di data leakage
2) Verificare correttezza matematica della normalizzazione
3) Verificare correttezza dell'imputazione
4) Controllare il comportamento di classi astratte
"""


# =========================================================
# MIN-MAX SCALING
# =========================================================

class TestMinMaxScaling(unittest.TestCase):

    def test_minmax_scale_basic(self):
        """
        Verifico che:
        - Il training venga scalato tra 0 e 1
        - Il test venga scalato usando SOLO le statistiche del training
        Questo evita data leakage.
        """
        X_train = np.array([
            [0.0, 10.0],
            [5.0, 20.0]
        ])
        X_test = np.array([
            [2.5, 15.0]
        ])

        X_train_s, X_test_s = minmax_scale_train_test(X_train, X_test)

        # Verifico che i valori del training siano tra 0 e 1
        self.assertTrue(np.all(X_train_s >= 0))
        self.assertTrue(np.all(X_train_s <= 1))

        # Verifico che il test venga scalato coerentemente
        # usando min e max del TRAINING
        self.assertAlmostEqual(X_test_s[0, 0], 0.5)
        self.assertAlmostEqual(X_test_s[0, 1], 0.5)

    def test_minmax_scale_constant_feature(self):
        """
        Caso limite: feature costante.
        max = min → denominatore zero.
        La colonna deve diventare tutta zero
        senza generare NaN o errore.
        """
        X_train = np.array([
            [1.0, 5.0],
            [1.0, 10.0]
        ])
        X_test = np.array([
            [1.0, 7.0]
        ])

        X_train_s, X_test_s = minmax_scale_train_test(X_train, X_test)

        # Colonna costante → tutti 0
        self.assertTrue(np.all(X_train_s[:, 0] == 0))
        self.assertTrue(np.all(X_test_s[:, 0] == 0))


# =========================================================
# MEDIAN IMPUTATION
# =========================================================

class TestMedianImputation(unittest.TestCase):

    def test_median_imputation_basic(self):
        """
        Verifico che:
        - I NaN vengano sostituiti con la mediana del TRAINING
        - Il test utilizzi la stessa mediana del training
        """
        X_train = np.array([
            [1.0, np.nan],
            [3.0, 4.0]
        ])
        X_test = np.array([
            [np.nan, 2.0]
        ])

        X_train_i, X_test_i = median_impute_train_test(X_train, X_test)

        # Median of column 0 = (1+3)/2 = 2.0
        self.assertEqual(X_test_i[0, 0], 2.0)

        # Median of column 1 = 4.0
        self.assertEqual(X_train_i[0, 1], 4.0)

    def test_no_nan_after_imputation(self):
        """
        Dopo l'imputazione non devono rimanere NaN.
        Questo garantisce stabilità nel modello.
        """
        X_train = np.array([
            [1.0, np.nan],
            [2.0, 3.0]
        ])
        X_test = np.array([
            [np.nan, np.nan]
        ])

        X_train_i, X_test_i = median_impute_train_test(X_train, X_test)

        self.assertFalse(np.isnan(X_train_i).any())
        self.assertFalse(np.isnan(X_test_i).any())


# =========================================================
# VALIDATION STRATEGY ABSTRACT CLASS
# =========================================================

class TestValidationStrategy(unittest.TestCase):

    def test_validation_strategy_is_abstract(self):
        """
        Verifico che la classe base non sia istanziabile.
        Essendo astratta, deve essere estesa da classi concrete
        come Holdout, KFold, LeavePOut.
        """
        with self.assertRaises(TypeError):
            ValidationStrategy()


if __name__ == "__main__":
    unittest.main()
