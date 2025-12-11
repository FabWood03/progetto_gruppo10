import numpy as np

def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Distanza euclidea (L2). 
    È la distanza standard richiesta dal professore.
    """
    diff = x - y
    return float(np.sqrt(np.sum(diff * diff)))
