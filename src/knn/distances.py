from abc import ABC, abstractmethod

import numpy as np


class DistanceStrategy(ABC):
    """
    Interfaccia base (Strategy) per tutte le metriche di distanza.
    Ogni algoritmo di distanza deve implementare il metodo calculate.
    """

    @abstractmethod
    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        pass


class EuclideanDistance(DistanceStrategy):
    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Distanza euclidea (L2).
        È la distanza standard richiesta.
        """
        diff = x - y
        return float(np.sqrt(np.sum(diff * diff)))


class ManhattanDistance(DistanceStrategy):
    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Distanza di Manhattan (L1).
        Molto adatta ai dati ordinali 1–10 del dataset.
        """
        return float(np.sum(np.abs(x - y)))


class ChebyshevDistance(DistanceStrategy):
    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Distanza di Chebyshev (L∞).
        Misura la massima differenza tra le feature.
        Utile in contesti dove conta la 'peggior' caratteristica.
        """
        return float(np.max(np.abs(x - y)))


class CosineDistance(DistanceStrategy):
    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Distanza basata su 1 - coseno.
        Confronta la 'forma' del profilo delle feature.
        Richiede che le feature siano normalizzate (come abbiamo fatto).
        """
        x_norm = np.linalg.norm(x)
        y_norm = np.linalg.norm(y)

        if x_norm == 0.0 or y_norm == 0.0:
            return 1.0

        cos_sim = float(np.dot(x, y) / (x_norm * y_norm))
        return 1.0 - cos_sim


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
            valid_keys = list(DistanceFactory.DISTANCE_FUNCTIONS.keys())
            raise ValueError(f"Metrica '{name}' non supportata. Disponibili: {valid_keys}")

        # Istanzia la classe corrispondente e la ritorna
        metric_class = DistanceFactory.DISTANCE_FUNCTIONS[name]
        return metric_class()
