import unittest
import numpy as np

from src.knn.classifier import KNNClassifier


class TestKNNClassifier(unittest.TestCase):

    def setUp(self):
        self.X = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0]
        ])
        self.y = np.array([0, 0, 1])

    # =========================
    # __init__
    # =========================
    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            KNNClassifier(k=0)

    def test_invalid_distance_name(self):
        with self.assertRaises(ValueError):
            KNNClassifier(distance="invalid")

    # =========================
    # fit
    # =========================
    def test_fit_invalid_X_dim(self):
        knn = KNNClassifier()
        with self.assertRaises(ValueError):
            knn.fit(np.array([1, 2, 3]), self.y)

    def test_fit_invalid_y_dim(self):
        knn = KNNClassifier()
        with self.assertRaises(ValueError):
            knn.fit(self.X, np.array([[0, 1, 1]]))

    def test_fit_length_mismatch(self):
        knn = KNNClassifier()
        with self.assertRaises(ValueError):
            knn.fit(self.X, np.array([0, 1]))

    def test_fit_k_greater_than_samples(self):
        knn = KNNClassifier(k=10)
        with self.assertRaises(ValueError):
            knn.fit(self.X, self.y)

    def test_fit_success(self):
        knn = KNNClassifier()
        knn.fit(self.X, self.y)
        self.assertIsNotNone(knn.X_train)
        self.assertIsNotNone(knn.y_train)

    # =========================
    # _compute_distances
    # =========================

    def test_compute_distances_without_fit(self):
        knn = KNNClassifier()
        with self.assertRaises(RuntimeError):
            knn._compute_distances(np.array([0, 0]))

    def test_compute_distances_shape(self):
        knn = KNNClassifier()
        knn.fit(self.X, self.y)
        dists = knn._compute_distances(np.array([0, 0]))
        self.assertEqual(dists.shape, (self.X.shape[0],))
