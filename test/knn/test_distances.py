import unittest
import numpy as np
import sys
import os

# Aggiunge la root al path per importare i moduli corretti
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.knn.distances import (
    DistanceFactory,
    EuclideanDistance,
    ManhattanDistance,
    ChebyshevDistance,
    CosineDistance
)


class TestDistances(unittest.TestCase):

    def setUp(self):
        """
        Dati comuni utilizzati in più test.
        """
        self.x = np.array([1.0, 2.0])

        self.matrix = np.array([
            [4.0, 6.0],   # campione 1
            [1.0, 2.0],   # campione 2 (identico a x)
            [0.0, 0.0]    # campione 3
        ])

    # =========================
    # FACTORY
    # =========================

    def test_factory_creation(self):
        """
        Verifica che il Factory restituisca l'istanza corretta
        e sollevi errori per metriche non supportate.
        """
        self.assertIsInstance(
            DistanceFactory.get_distance("euclidean"),
            EuclideanDistance
        )
        self.assertIsInstance(
            DistanceFactory.get_distance("manhattan"),
            ManhattanDistance
        )
        self.assertIsInstance(
            DistanceFactory.get_distance("cosine"),
            CosineDistance
        )

        with self.assertRaises(ValueError):
            DistanceFactory.get_distance("metrica_inventata")

    def test_factory_case_and_spaces(self):
        """
        Il nome della metrica deve essere case-insensitive
        e tollerare spazi.
        """
        dist = DistanceFactory.get_distance("  EuClIdEaN  ")
        self.assertIsInstance(dist, EuclideanDistance)

    def test_factory_invalid_type(self):
        """
        Il Factory deve fallire se il tipo non è stringa.
        """
        with self.assertRaises(AttributeError):
            DistanceFactory.get_distance(None)

    def test_factory_empty_string(self):
        """
        Stringa vuota non valida.
        """
        with self.assertRaises(ValueError):
            DistanceFactory.get_distance("")

        # =========================
        # DISTANZE CORRETTE
        # =========================

    def test_euclidean_distance(self):
        """
        Verifica matematica della distanza Euclidea.
        """
        strategy = EuclideanDistance()
        dists = strategy.calculate(self.x, self.matrix)

        expected = np.array([
            5.0,
            0.0,
            np.sqrt(5)
        ])

        np.testing.assert_allclose(
            dists,
            expected,
            rtol=1e-5,
            err_msg="Errore nel calcolo della distanza Euclidea"
        )

    def test_manhattan_distance(self):
        """
        Verifica matematica della distanza Manhattan.
        """
        strategy = ManhattanDistance()
        dists = strategy.calculate(self.x, self.matrix)

        expected = np.array([7.0, 0.0, 3.0])

        np.testing.assert_allclose(
            dists,
            expected,
            err_msg="Errore nel calcolo della distanza Manhattan"
        )

    def test_chebyshev_distance(self):
        """
        Verifica matematica della distanza Chebyshev.
        """
        strategy = ChebyshevDistance()
        dists = strategy.calculate(self.x, self.matrix)

        expected = np.array([4.0, 0.0, 2.0])

        np.testing.assert_allclose(
            dists,
            expected,
            err_msg="Errore nel calcolo della distanza Chebyshev"
        )

    def test_cosine_distance(self):
        """
        Verifica della distanza Coseno in casi noti.
        """
        strategy = CosineDistance()

        # Vettori ortogonali -> cos = 0 -> distanza = 1
        x_ortho = np.array([1.0, 0.0])
        mat_ortho = np.array([[0.0, 1.0]])
        res_ortho = strategy.calculate(x_ortho, mat_ortho)
        self.assertAlmostEqual(res_ortho[0], 1.0)

        # Vettori paralleli -> cos = 1 -> distanza = 0
        x_para = np.array([2.0, 2.0])
        mat_para = np.array([[1.0, 1.0]])
        res_para = strategy.calculate(x_para, mat_para)
        self.assertAlmostEqual(res_para[0], 0.0)


    # =========================
    # CASI LIMITE / ERRORI
    # =========================

    def test_dimension_mismatch(self):
        """
        Dimensioni incompatibili tra x e matrix.
        """
        strategy = EuclideanDistance()

        x = np.array([1.0, 2.0, 3.0])
        matrix = np.array([[1.0, 2.0]])

        with self.assertRaises(ValueError):
            strategy.calculate(x, matrix)

    def test_matrix_not_2d(self):
        """
        matrix deve essere bidimensionale.
        """
        strategy = ManhattanDistance()

        x = np.array([1.0, 2.0])
        matrix = np.array([3.0, 4.0])  # non 2D

        with self.assertRaises(ValueError):
            strategy.calculate(x, matrix)

    def test_empty_matrix(self):
        """
        matrix vuota: deve restituire un array vuoto.
        """
        strategy = EuclideanDistance()

        x = np.array([1.0, 2.0])
        matrix = np.empty((0, 2))

        result = strategy.calculate(x, matrix)
        self.assertEqual(result.shape[0], 0)

    def test_negative_values(self):
        """
        Supporto a valori negativi.
        """
        strategy = ManhattanDistance()

        x = np.array([-1.0, -2.0])
        matrix = np.array([[1.0, 2.0]])

        expected = np.array([6.0])
        np.testing.assert_allclose(strategy.calculate(x, matrix), expected)


if __name__ == '__main__':
    unittest.main()
