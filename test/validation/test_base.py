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