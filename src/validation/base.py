from abc import ABC, abstractmethod


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
