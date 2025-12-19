import unittest
import numpy as np

from src.metrics.evaluator import (
    confusion_counts,
    accuracy_rate,
    error_rate,
    sensitivity,
    specificity,
    precision,
    f1_score,
    geometric_mean,
    roc_curve_manual,
    calculate_auc,
    evaluate_metrics
)


class TestConfusionCounts(unittest.TestCase):

    def test_confusion_counts_basic(self):
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 0, 0]

        c = confusion_counts(y_true, y_pred)

        self.assertEqual(c.tp, 1)
        self.assertEqual(c.tn, 2)
        self.assertEqual(c.fp, 0)
        self.assertEqual(c.fn, 1)

    def test_confusion_counts_length_mismatch(self):
        with self.assertRaises(ValueError):
            confusion_counts([1, 0], [1])

    def test_confusion_counts_empty(self):
        with self.assertRaises(ValueError):
            confusion_counts([], [])
