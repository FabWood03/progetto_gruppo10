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


if __name__ == '__main__':
    unittest.main()
