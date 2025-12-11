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
