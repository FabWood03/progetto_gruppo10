import unittest
import numpy as np

from src.metrics.evaluator import (
    ConfusionCounts,
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
class TestScalarMetrics(unittest.TestCase):

    def setUp(self):
        self.c = ConfusionCounts(tp=8, tn=6, fp=2, fn=4)

    def test_accuracy(self):
        self.assertAlmostEqual(accuracy_rate(self.c), (8 + 6) / 20)

    def test_error_rate(self):
        self.assertAlmostEqual(error_rate(self.c), 1 - accuracy_rate(self.c))

    def test_sensitivity(self):
        self.assertAlmostEqual(sensitivity(self.c), 8 / (8 + 4))

    def test_specificity(self):
        self.assertAlmostEqual(specificity(self.c), 6 / (6 + 2))

    def test_precision(self):
        self.assertAlmostEqual(precision(self.c), 8 / (8 + 2))

    def test_f1_score(self):
        p = precision(self.c)
        r = sensitivity(self.c)
        self.assertAlmostEqual(f1_score(self.c), 2 * p * r / (p + r))

    def test_geometric_mean(self):
        sens = sensitivity(self.c)
        spec = specificity(self.c)
        self.assertAlmostEqual(geometric_mean(self.c), np.sqrt(sens * spec))

class TestZeroDivisionCases(unittest.TestCase):

    def test_zero_division_precision(self):
        c = ConfusionCounts(tp=0, tn=5, fp=0, fn=5)
        self.assertEqual(precision(c), 0.0)

    def test_zero_division_sensitivity(self):
        c = ConfusionCounts(tp=0, tn=5, fp=2, fn=0)
        self.assertEqual(sensitivity(c), 0.0)

    def test_zero_division_specificity(self):
        c = ConfusionCounts(tp=5, tn=0, fp=0, fn=5)
        self.assertEqual(specificity(c), 0.0)