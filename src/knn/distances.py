import numpy as np

def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Distanza euclidea (L2). 
    È la distanza standard richiesta.
    """
    diff = x - y
    return float(np.sqrt(np.sum(diff * diff)))

def manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Distanza di Manhattan (L1).
    Molto adatta ai dati ordinali 1–10 del dataset.
    """
    return float(np.sum(np.abs(x - y)))


