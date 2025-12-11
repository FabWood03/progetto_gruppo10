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

def chebyshev_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Distanza di Chebyshev (L∞).
    Misura la massima differenza tra le feature.
    Utile in contesti dove conta la 'peggior' caratteristica.
    """
    return float(np.max(np.abs(x - y)))

def cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
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

DISTANCE_FUNCTIONS = {
    "euclidean": euclidean_distance,
    "manhattan": manhattan_distance,
    "chebyshev": chebyshev_distance,
    "cosine": cosine_distance,
}




