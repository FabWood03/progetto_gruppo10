import numpy as np
from .distances import DistanceFactory


class KNNClassifier:
    """
    Implementazione manuale del classificatore K-NN.
    """

    def __init__(self, k: int = 3, distance: str = "euclidean", random_state: int | None = None):
        if k <= 0:
            raise ValueError("k deve essere > 0")

        self.k = k
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.distance_metric = DistanceFactory.get_distance(distance)
        self.distance_name = distance

        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Addestra il modello memorizzando il training set.
        :param X: caratteristiche del training set
        :param y: etichette del training set
        :return: None
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2: raise ValueError(f"X deve essere 2D")
        if y.ndim != 1: raise ValueError(f"y deve essere 1D")
        if X.shape[0] != y.shape[0]: raise ValueError("X e y diverse lunghezze")
        if self.k > X.shape[0]: raise ValueError("k > n_samples")

        self.X_train = X
        self.y_train = y

    def _compute_distances(self, x: np.ndarray) -> np.ndarray:
        """
        Calcola le distanze tra un campione e tutti i campioni di training.
         1 vs N calcolo vettorializzato.
        :param x: campione di input
        :return: array di distanze
        """
        if self.X_train is None: raise RuntimeError("KNN non addestrato")
        # Calcolo vettorializzato (1 vs N)
        return self.distance_metric.calculate(x, self.X_train)

    def _vote(self, neighbor_labels: np.ndarray) -> int:
        """
        Esegue il voto tra le etichette dei vicini.
        In caso di pareggio, sceglie casualmente tra le etichette con il
        conteggio massimo.
        :param neighbor_labels: etichetta dei k vicini
        :return: etichetta predetta
        """
        unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
        max_count = counts.max()
        best_labels = unique_labels[counts == max_count]

        if best_labels.size == 1:
            return int(best_labels[0])

        idx = self.rng.integers(low=0, high=best_labels.size)
        return int(best_labels[idx])

    # --- METODI PER VALIDAZIONE ---
    def predict_with_distances(self, distances: np.ndarray) -> int:
        """Predice la classe usando distanze precalcolate."""
        if self.y_train is None: raise RuntimeError("KNN non addestrato")
        neighbor_idxs = np.argsort(distances)[:self.k]
        return self._vote(self.y_train[neighbor_idxs])

    def predict_proba_with_distances(self, distances: np.ndarray) -> np.ndarray:
        """
        Predice le probabilità usando distanze precalcolate.
        """
        if self.y_train is None: raise RuntimeError("KNN non addestrato")

        # 1. Trova i k vicini
        neighbor_idxs = np.argsort(distances)[:self.k]
        neighbor_labels = self.y_train[neighbor_idxs]

        # 2. Calcola probabilità
        classes = np.unique(self.y_train)

        proba = np.zeros(len(classes))
        for i, c in enumerate(classes):
            proba[i] = np.sum(neighbor_labels == c) / self.k

        return proba

    def predict_one(self, x: np.ndarray) -> int:
        dists = self._compute_distances(x)
        return self.predict_with_distances(dists)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.X_train is None: raise RuntimeError("KNN non addestrato")
        x = np.asarray(x)
        if x.ndim == 1: return np.array([self.predict_one(x)])
        return np.array([self.predict_one(s) for s in x])

    def predict_proba_one(self, x: np.ndarray) -> np.ndarray:
        dists = self._compute_distances(x)
        return self.predict_proba_with_distances(dists)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.X_train is None: raise RuntimeError("KNN non addestrato")
        x = np.asarray(x)
        if x.ndim == 1: return np.array([self.predict_proba_one(x)])
        return np.array([self.predict_proba_one(s) for s in x])
