import numpy as np
from .distances import DISTANCE_FUNCTIONS

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
        if distance not in DISTANCE_FUNCTIONS:
            raise ValueError(f"Distanza '{distance}' non supportata. Disponibili: {list(DISTANCE_FUNCTIONS.keys())}")

        self.k = k
        self.distance_name = distance
        self.distance_fn = DISTANCE_FUNCTIONS[distance]
        self.random_state = random_state

        # Random generator per gestione pareggi
        self.rng = np.random.default_rng(random_state)

        # Verranno valorizzati in fit()
        self.X_train = None
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
            d = self.distance_fn(x, row)  # usa la distanza scelta nel costruttore
            distances.append(d)

        return np.array(distances, dtype=float)



    
