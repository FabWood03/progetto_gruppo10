from abc import ABC, abstractmethod

import numpy as np


class DistanceStrategy(ABC):
    """
    Interfaccia base (Strategy) per tutte le metriche di distanza.
    Ogni algoritmo di distanza deve implementare il metodo calculate.
    """

    @abstractmethod
    def calculate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass


class EuclideanDistance(DistanceStrategy):
    def calculate(self, x: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """
        Distanza euclidea (L2).
        È la distanza standard richiesta.
        """
        return np.sqrt(np.sum((matrix - x) ** 2, axis=1))


class ManhattanDistance(DistanceStrategy):
    def calculate(self, x: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """
        Distanza di Manhattan (L1).
        Molto adatta ai dati ordinali 1–10 del dataset.
        """
        return np.sum(np.abs(matrix - x), axis=1)


class ChebyshevDistance(DistanceStrategy):
    def calculate(self, x: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """
        Distanza di Chebyshev (L∞).
        Misura la massima differenza tra le feature.
        Utile in contesti dove conta la 'peggior' caratteristica.
        """
        return np.max(np.abs(matrix - x), axis=1)


class CosineDistance(DistanceStrategy):
    def calculate(self, x: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """
        Distanza basata su 1 - coseno.
        Confronta la 'forma' del profilo delle feature.
        Richiede che le feature siano normalizzate (come abbiamo fatto).
        """
        x_norm = np.linalg.norm(x)
        matrix_norm = np.linalg.norm(matrix, axis=1)

        denom = (x_norm * matrix_norm)

        # Evita divisione per zero
        denom[denom == 0] = 1e-10

        dot_product = np.dot(matrix, x)
        return 1.0 - (dot_product / denom)


class DistanceFactory:
    DISTANCE_FUNCTIONS = {
        "euclidean": EuclideanDistance,
        "manhattan": ManhattanDistance,
        "chebyshev": ChebyshevDistance,
        "cosine": CosineDistance,
    }

    @staticmethod
    def get_distance(name: str) -> DistanceStrategy:
        """
        Ritorna un'istanza della metrica di distanza richiesta.
        Solleva ValueError se il nome non è valido.
        """
        name = name.lower().strip()
        if name not in DistanceFactory.DISTANCE_FUNCTIONS:
            raise ValueError(f"Metrica '{name}' non supportata.")
        return DistanceFactory.DISTANCE_FUNCTIONS[name]()
