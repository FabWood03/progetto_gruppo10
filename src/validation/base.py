from abc import ABC, abstractmethod

import numpy as np

def minmax_scale_train_test(X_train, X_test):
    """
    Applica una normalizzazione Min-Max utilizzando esclusivamente
    i dati di training, per evitare il data leakage.

    La stessa trasformazione viene poi applicata al test set.

    :param X_train: Matrice delle feature di training.
    :param X_test: Matrice delle feature di test.
    :return: Tuple (X_train_normalizzato, X_test_normalizzato).
    """
    min_vals = X_train.min(axis=0)
    max_vals = X_train.max(axis=0)

    denom = max_vals - min_vals
    denom[denom == 0] = 1.0  # evita divisioni per zero

    X_train_scaled = (X_train - min_vals) / denom
    X_test_scaled = (X_test - min_vals) / denom

    return X_train_scaled, X_test_scaled



class ValidationStrategy(ABC):
    @abstractmethod
    def validate(self, model, X, y):
        """
        Metodo astratto per la validazione del modello.
        :param model: Il modello da validare.
        :param X: I dati di input.
        :param y: Le etichette di output.
        :return: Risultati della validazione.
        """
        pass
