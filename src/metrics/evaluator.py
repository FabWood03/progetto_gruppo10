from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Sequence

import numpy as np


@dataclass(frozen=True)
class ConfusionCounts:
    """
    Struttura dati per memorizzare i conteggi della Matrice di Confusione.

    :param tp: Veri Positivi (True Positives).
    :param tn: Veri Negativi (True Negatives).
    :param fp: Falsi Positivi (False Positives).
    :param fn: Falsi Negativi (False Negatives).
    """
    tp: int
    tn: int
    fp: int
    fn: int


def confusion_counts(y_true: Sequence[int], y_pred: Sequence[int], pos_label: int = 1) -> ConfusionCounts:
    """
    Calcola TP, TN, FP, FN per classificazione binaria.

    :param y_true: Etichette vere (ground truth).
    :param y_pred: Etichette previste dal modello.
    :param pos_label: Valore della classe positiva (default: 1).
    :return: Conteggi della matrice di confusione.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"y_true e y_pred devono avere la stessa lunghezza "
                         f"({y_true.shape[0]} != {y_pred.shape[0]}).")
    if y_true.size == 0:
        raise ValueError("y_true è vuoto, impossibile calcolare metriche.")

    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    tn = np.sum((y_true != pos_label) & (y_pred != pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label))

    return ConfusionCounts(tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))


def _safe_div(num: float, den: float, zero_value: float = 0.0) -> float:
    """
    Esegue la divisione in modo sicuro, restituendo `zero_value` (default 0.0) se il denominatore è zero.
    """
    # Sopprimiamo il RuntimeWarning di NumPy su divisione per zero
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(num, den)

    return float(result) if den != 0 else float(zero_value)


def accuracy_rate(c: ConfusionCounts) -> float:
    """Calcola l'Accuracy = (TP + TN) / Totale."""
    total = c.tp + c.tn + c.fp + c.fn
    return _safe_div(c.tp + c.tn, total)


def error_rate(c: ConfusionCounts) -> float:
    """Calcola l'Error Rate = 1 - Accuracy."""
    return 1.0 - accuracy_rate(c)


def sensitivity(c: ConfusionCounts) -> float:
    """Calcola la Sensitivity (Recall) = TP / (TP + FN)."""
    return _safe_div(c.tp, c.tp + c.fn)


def specificity(c: ConfusionCounts) -> float:
    """Calcola la Specificity = TN / (TN + FP)."""
    return _safe_div(c.tn, c.tn + c.fp)


def geometric_mean(c: ConfusionCounts) -> float:
    """Calcola la G-Mean = sqrt(Sensitivity * Specificity)."""
    sens = sensitivity(c)
    spec = specificity(c)
    return float(np.sqrt(sens * spec))


def precision(c: ConfusionCounts) -> float:
    """Calcola la Precision = TP / (TP + FP)."""
    return _safe_div(c.tp, c.tp + c.fp)


def f1_score(c: ConfusionCounts) -> float:
    """Calcola l'F1-Score (media armonica tra Precision e Sensitivity)."""
    p = precision(c)
    r = sensitivity(c)
    return _safe_div(2 * p * r, p + r)


def roc_curve_manual(
        y_true: Sequence[int],
        y_score: Sequence[float],
        pos_label: int = 1,
        neg_label: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Costruisce la curva ROC (FPR vs TPR) variando soglie sullo score continuo.

    :param y_true: Etichette vere.
    :param y_score: Score continui (probabilità, distanza, etc.).
    :param pos_label: Valore da considerare come classe positiva.
    :param neg_label: Valore da considerare come classe negativa.
    :return: (fpr, tpr, thresholds).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    # Ordina score decrescente
    desc_idx = np.argsort(-y_score)
    y_true_sorted = y_true[desc_idx]
    y_score_sorted = y_score[desc_idx]

    # Soglie uniche + infinito iniziale
    thresholds = np.r_[np.inf, np.unique(y_score_sorted)[::-1]]

    # Cumulativi positivi/negativi
    tp_cumsum = np.cumsum(y_true_sorted == pos_label)
    fp_cumsum = np.cumsum(y_true_sorted == neg_label)

    tot_pos = tp_cumsum[-1] if tp_cumsum.size > 0 else 0
    tot_neg = fp_cumsum[-1] if fp_cumsum.size > 0 else 0

    # TPR/FPR per soglia
    tpr = np.zeros_like(thresholds, dtype=float)
    fpr = np.zeros_like(thresholds, dtype=float)

    # Indici per ogni soglia
    for i, thr in enumerate(thresholds[1:], start=1):
        idx = np.searchsorted(-y_score_sorted, -thr, side='right')
        tpr[i] = _safe_div(tp_cumsum[idx - 1] if idx > 0 else 0, tot_pos)
        fpr[i] = _safe_div(fp_cumsum[idx - 1] if idx > 0 else 0, tot_neg)

    return fpr, tpr, thresholds


def calculate_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """
    Calcola l'Area Sotto la Curva ROC (AUC) tramite la regola del trapezio (integrazione).
    """
    # Assicura che FPR sia ordinato per l'integrazione
    sorted_indices = np.argsort(fpr)
    return float(np.trapezoid(tpr[sorted_indices], fpr[sorted_indices]))


def evaluate_metrics(
        y_true: Sequence[int],
        y_pred: Sequence[int],
        y_score: Optional[Sequence[float]] = None,
        metrics: Optional[List[str]] = None,
        pos_label: int = 1,
        neg_label: int = 0) -> Dict[str, float]:
    """
    Calcola un insieme specificato di metriche di valutazione del classificatore.

    Metriche disponibili: 'accuracy', 'error', 'sensitivity', 'specificity',
    'precision', 'f1', 'gmean', 'auc'.

    :param y_true: Etichette vere.
    :param y_pred: Etichette predette.
    :param y_score: Score continuo (necessario solo per 'auc').
    :param metrics: Lista di metriche da calcolare (default: tutte).
    :param pos_label: Valore da considerare come classe positiva.
    :param neg_label: Valore da considerare come classe negativa.
    :return: Dizionario {nome_metrica: valore}.
    """
    allowed_metrics = {"accuracy", "error", "sensitivity", "specificity",
                       "precision", "f1", "gmean", "auc"}
    if metrics is None:
        metrics = list(allowed_metrics)
    unknown = set(metrics) - allowed_metrics
    if unknown:
        raise ValueError(f"Metriche non riconosciute: {unknown}")

    c = confusion_counts(y_true, y_pred, pos_label)
    out: Dict[str, float] = {}

    metric_funcs = {
        "accuracy": accuracy_rate, "error": error_rate,
        "sensitivity": sensitivity, "specificity": specificity,
        "precision": precision, "f1": f1_score,
        "gmean": geometric_mean
    }

    for name in metrics:
        if name != "auc" and name in metric_funcs:
            out[name] = metric_funcs[name](c)

    if "auc" in metrics:
        if y_score is None:
            raise ValueError("AUC richiesta ma y_score è None.")
        fpr, tpr, _ = roc_curve_manual(y_true, y_score, pos_label, neg_label)
        out["auc"] = calculate_auc(fpr, tpr)

    return {k: out[k] for k in sorted(out)}
