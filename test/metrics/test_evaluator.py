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

class TestROCAndAUC(unittest.TestCase):

    def test_roc_curve_shapes(self):
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.4, 0.35, 0.8]

        fpr, tpr, thresholds = roc_curve_manual(y_true, y_score)

        self.assertEqual(len(fpr), len(tpr))
        self.assertEqual(len(tpr), len(thresholds))

    def test_auc_perfect_classifier(self):
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.2, 0.8, 0.9]

        fpr, tpr, _ = roc_curve_manual(y_true, y_score)
        auc = calculate_auc(fpr, tpr)

        self.assertAlmostEqual(auc, 0.5)

    def test_auc_random_classifier(self):
        y_true = [0, 1, 0, 1]
        y_score = [0.5, 0.5, 0.5, 0.5]

        fpr, tpr, _ = roc_curve_manual(y_true, y_score)
        auc = calculate_auc(fpr, tpr)

        self.assertAlmostEqual(auc, 0.0)

class TestEvaluateMetrics(unittest.TestCase):

    def test_evaluate_selected_metrics(self):
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 0, 0]

        metrics = evaluate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            metrics=["accuracy", "precision"]
        )

        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertNotIn("f1", metrics)

    def test_evaluate_with_auc(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        y_score = [0.1, 0.2, 0.8, 0.9]

        metrics = evaluate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score,
            metrics=["auc"]
        )

        self.assertIn("auc", metrics)
        self.assertAlmostEqual(metrics["auc"], 0.5)

    def test_auc_without_score_raises(self):
        with self.assertRaises(ValueError):
            evaluate_metrics(
                y_true=[0, 1],
                y_pred=[0, 1],
                metrics=["auc"]
            )

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            evaluate_metrics(
                y_true=[0, 1],
                y_pred=[0, 1],
                metrics=["invalid_metric"]
            )


if __name__ == "__main__":
    unittest.main()