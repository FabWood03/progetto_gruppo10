import unittest
import numpy as np

from src.validation.base import (
    minmax_scale_train_test,
    median_impute_train_test,
    ValidationStrategy
)


class TestMinMaxScaling(unittest.TestCase):

    def test_minmax_scale_basic(self):
        X_train = np.array([
            [0.0, 10.0],
            [5.0, 20.0]
        ])
        X_test = np.array([
            [2.5, 15.0]
        ])

        X_train_s, X_test_s = minmax_scale_train_test(X_train, X_test)

        # Train scaled in [0, 1]
        self.assertTrue(np.all(X_train_s >= 0))
        self.assertTrue(np.all(X_train_s <= 1))

        # Test scaled using TRAIN statistics
        self.assertAlmostEqual(X_test_s[0, 0], 0.5)
        self.assertAlmostEqual(X_test_s[0, 1], 0.5)

    def test_minmax_scale_constant_feature(self):
        X_train = np.array([
            [1.0, 5.0],
            [1.0, 10.0]
        ])
        X_test = np.array([
            [1.0, 7.0]
        ])

        X_train_s, X_test_s = minmax_scale_train_test(X_train, X_test)

        # Colonna costante → tutto 0
        self.assertTrue(np.all(X_train_s[:, 0] == 0))
        self.assertTrue(np.all(X_test_s[:, 0] == 0))
class TestMedianImputation(unittest.TestCase):

    def test_median_imputation_basic(self):
        X_train = np.array([
            [1.0, np.nan],
            [3.0, 4.0]
        ])
        X_test = np.array([
            [np.nan, 2.0]
        ])

        X_train_i, X_test_i = median_impute_train_test(X_train, X_test)

        # Median of column 0 = 2.0
        self.assertEqual(X_test_i[0, 0], 2.0)

        # Median of column 1 = 4.0
        self.assertEqual(X_train_i[0, 1], 4.0)

    def test_no_nan_after_imputation(self):
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

class TestValidationStrategy(unittest.TestCase):

    def test_validation_strategy_is_abstract(self):
        with self.assertRaises(TypeError):
            ValidationStrategy()


if __name__ == "__main__":
    unittest.main()