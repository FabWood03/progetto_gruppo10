import unittest
import numpy as np
import tempfile
from pathlib import Path

from src.metrics.plotter import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_metric_distribution
)


class TestPlotConfusionMatrix(unittest.TestCase):

    def test_confusion_matrix_saved(self):
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

            self.assertTrue(save_path.exists())

    def test_confusion_matrix_invalid_shape(self):
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
class TestPlotROC(unittest.TestCase):

    def test_roc_curve_saved(self):
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