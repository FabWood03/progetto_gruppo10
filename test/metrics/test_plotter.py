import unittest
import numpy as np
import tempfile
from pathlib import Path

from src.metrics.plotter import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_metric_distribution
)


"""
Questo file testa il modulo di visualizzazione (plotter).

L’obiettivo non è verificare il contenuto grafico pixel per pixel,
ma garantire:

1) Corretta generazione e salvataggio dei file
2) Validazione degli input
3) Robustezza anche con input vuoti
4) Stabilità del sistema di plotting
"""


# =========================================================
# CONFUSION MATRIX PLOT
# =========================================================

class TestPlotConfusionMatrix(unittest.TestCase):

    def test_confusion_matrix_saved(self):
        """
        Verifico che la funzione generi correttamente
        il file della matrice di confusione.
        """
        cm = np.array([[5, 1], [2, 7]])
        labels = ["Neg", "Pos"]

        with tempfile.TemporaryDirectory() as tmp:
            save_path = Path(tmp) / "cm.png"

            plot_confusion_matrix(
                cm=cm,
                labels=labels,
                title="Test CM",
                save_path=save_path,
                normalize=False
            )

            # Verifico che il file sia stato effettivamente creato
            self.assertTrue(save_path.exists())

    def test_confusion_matrix_invalid_shape(self):
        """
        Se la matrice non è 2D quadrata,
        deve essere sollevato ValueError.
        """
        cm = np.array([1, 2, 3])
        labels = ["A", "B"]

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                plot_confusion_matrix(
                    cm=cm,
                    labels=labels,
                    title="Invalid",
                    save_path=Path(tmp) / "cm.png"
                )

    def test_confusion_matrix_label_mismatch(self):
        """
        Se il numero di etichette non coincide
        con la dimensione della matrice,
        deve essere sollevato errore.
        """
        cm = np.array([[1, 0], [0, 1]])
        labels = ["OnlyOneLabel"]

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                plot_confusion_matrix(
                    cm=cm,
                    labels=labels,
                    title="Invalid",
                    save_path=Path(tmp) / "cm.png"
                )


# =========================================================
# ROC CURVE PLOT
# =========================================================

class TestPlotROC(unittest.TestCase):

    def test_roc_curve_saved(self):
        """
        Verifico che la curva ROC venga salvata correttamente.
        Non controllo l'immagine, ma la creazione del file.
        """
        fpr = np.array([0.0, 0.5, 1.0])
        tpr = np.array([0.0, 0.8, 1.0])
        auc = 0.7

        with tempfile.TemporaryDirectory() as tmp:
            save_path = Path(tmp) / "roc.png"

            plot_roc_curve(
                fpr=fpr,
                tpr=tpr,
                auc_value=auc,
                title="ROC Test",
                save_path=save_path
            )

            self.assertTrue(save_path.exists())


# =========================================================
# METRIC DISTRIBUTION PLOT
# =========================================================

class TestMetricDistribution(unittest.TestCase):

    def test_metric_distribution_saved(self):
        """
        Verifico che l'istogramma venga salvato correttamente.
        """
        values = [0.8, 0.85, 0.9, 0.88]

        with tempfile.TemporaryDirectory() as tmp:
            save_path = Path(tmp) / "hist.png"

            plot_metric_distribution(
                values=values,
                metric_name="Accuracy",
                title="Accuracy Distribution",
                save_path=save_path,
                bins=5
            )

            self.assertTrue(save_path.exists())

    def test_metric_distribution_empty_values(self):
        """
        Caso limite: lista valori vuota.
        La funzione non deve crashare
        e deve comunque generare un file valido.
        """
        values = []

        with tempfile.TemporaryDirectory() as tmp:
            save_path = Path(tmp) / "hist.png"

            plot_metric_distribution(
                values=values,
                metric_name="Accuracy",
                title="Empty",
                save_path=save_path
            )

            self.assertTrue(save_path.exists())


if __name__ == "__main__":
    unittest.main()
