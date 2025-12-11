import numpy as np
from .distances import DistanceFactory


class KNNClassifier:
    """
    Implementazione manuale del classificatore K-NN.
    Costruiremo i metodi progressivamente.
    """

    def __init__(self, k: int = 3, distance: str = "euclidean", random_state: int | None = None):
        """
        Costruttore della classe KNNClassifier.
        """
        if k <= 0:
            raise ValueError("k deve essere maggiore di zero")

        self.k = k
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.distance_metric = DistanceFactory.get_distance(distance)
        self.distance_name = distance

        self.x_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Addestra il classificatore memorizzando X e y.

        Parametri:
        - X: array numpy 2D (n_samples, n_features)
        - y: array numpy 1D (n_samples,)
        """

        # Conversione in array numpy (sicurezza)
        X = np.asarray(X)
        y = np.asarray(y)

        # Validazioni robuste
        if X.ndim != 2:
            raise ValueError(f"X deve essere 2D, shape ricevuta: {X.shape}")

        if y.ndim != 1:
            raise ValueError(f"y deve essere 1D, shape ricevuta: {y.shape}")

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Numero di campioni diverso tra X ({X.shape[0]}) e y ({y.shape[0]})"
            )

        if self.k > X.shape[0]:
            raise ValueError(
                f"k={self.k} non può essere maggiore del numero di campioni ({X.shape[0]})"
            )

        # Salvataggio dei dati
        self.X_train = X
        self.y_train = y

    def _compute_distances(self, x: np.ndarray) -> np.ndarray:
        """
        Calcola le distanze tra il campione x e tutti i campioni del training set.
        Ritorna un array numpy monodimensionale di distanze.
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("KNNClassifier non addestrato. Chiama fit(X, y) prima di predict().")

        x = np.asarray(x)

        # x deve essere un vettore 1D
        if x.ndim != 1:
            raise ValueError(f"x deve essere un vettore 1D, shape ricevuta: {x.shape}")

        # stessa dimensione delle feature
        if x.shape[0] != self.X_train.shape[1]:
            raise ValueError(
                f"Dimensione di x ({x.shape[0]}) diversa da n_features del training ({self.X_train.shape[1]})"
            )

        distances = []
        for row in self.X_train:
            d = self.distance_metric.calculate(x, row)
            distances.append(d)

        return np.array(distances, dtype=float)

    def _vote(self, neighbor_labels: np.ndarray) -> int:
        """
        Determina la classe finale tramite voto di maggioranza.

        Se c'è un pareggio tra due o più classi,
        viene scelta casualmente una delle classi con maggior frequenza.
        """
        neighbor_labels = np.asarray(neighbor_labels)

        # Trova etichette uniche e conteggi
        unique_labels, counts = np.unique(neighbor_labels, return_counts=True)

        max_count = counts.max()

        # Classi che hanno ottenuto il massimo numero di voti
        best_labels = unique_labels[counts == max_count]

        # Se c'è una sola classe vincente → la ritorniamo
        if best_labels.size == 1:
            return int(best_labels[0])

        # Altrimenti scegliamo casualmente una delle etichette con max voto
        idx = self.rng.integers(low=0, high=best_labels.size)
        return int(best_labels[idx])

    def predict_one(self, x: np.ndarray) -> int:
        """
        Predice la classe per un singolo campione x.

        Passi:
        - calcolo delle distanze tra x e tutto il training
        - selezione dei k vicini più vicini
        - voto di maggioranza con gestione pareggi
        """
        # Calcola distanze verso tutti i campioni del training
        distances = self._compute_distances(x)

        # Seleziona gli indici dei k campioni più vicini
        neighbor_idxs = np.argsort(distances)[:self.k]

        # Estrae le loro etichette
        neighbor_labels = self.y_train[neighbor_idxs]

        # Restituisce il risultato del voto
        return self._vote(neighbor_labels)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice le classi per uno o più campioni X.

        - Se X è un vettore 1D → predice un singolo campione.
        - Se X è 2D → predice tutti i campioni riga per riga.
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("KNNClassifier non addestrato. Chiama fit(X, y) prima di predict().")

        X = np.asarray(X)

        # Caso: un singolo campione (vettore 1D)
        if X.ndim == 1:
            return np.array([self.predict_one(X)], dtype=int)

        # Caso: più campioni (matrice 2D)
        if X.ndim != 2:
            raise ValueError(f"X deve essere 1D o 2D. Shape ricevuta: {X.shape}")

        predictions = [self.predict_one(sample) for sample in X]

        return np.array(predictions, dtype=int)
