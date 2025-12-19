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