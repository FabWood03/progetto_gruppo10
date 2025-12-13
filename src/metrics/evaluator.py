from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class ConfusionCounts:
    """
    La classe viene utilizzata come struttura dati di supporto per il calcolo
    delle metriche di valutazione del classificatore, evitando la gestione
    separata dei singoli valori TP, TN, FP e FN.
    :param tp : Veri Positivi (True Positives).
    :param tn : Veri Negativi (True Negatives).
    :param fp : Falsi Positivi (False Positives).
    :param fn : Falsi Negativi (False Negatives).

    """
    tp: int
    tn: int
    fp: int
    fn: int


def confusion_counts(y_true, y_pred, pos_label: int = 4) -> ConfusionCounts:
    """
    Calcola TP, TN, FP, FN in uno scenario di classificazione binaria.

    Note:
        In molti setup del dataset Breast Cancer Wisconsin:
        - classe 2 = benigno (negativo)
        - classe 4 = maligno (positivo)
        `pos_label` indica il valore della classe positiva.

    :param y_true: Etichette vere (ground truth).
    :param y_pred: Etichette previste dal modello.
    :param pos_label: Valore da considerare come classe positiva.
    :return: Conte della matrice di confusione (tp, tn, fp, fn).

    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    """
        In caso di lunghezze diverse, il calcolo dei conteggi TP, TN, FP, FN
        risulterebbe logicamente errato o potrebbe generare comportamenti
        indesiderati dovuti al broadcasting di NumPy.
        
    """
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            "y_true e y_pred devono avere la stessa lunghezza "
            f"(ottenuti {y_true.shape[0]} e {y_pred.shape[0]})."
        )

    # Un dataset vuoto rende prive di significato le metriche di valutazione
    if y_true.shape[0] == 0:
        raise ValueError("Impossibile calcolare le metriche: y_true è vuoto.")

    tp = int(np.sum((y_true == pos_label) & (y_pred == pos_label)))
    tn = int(np.sum((y_true != pos_label) & (y_pred != pos_label)))
    fp = int(np.sum((y_true != pos_label) & (y_pred == pos_label)))
    fn = int(np.sum((y_true == pos_label) & (y_pred != pos_label)))

    return ConfusionCounts(tp=tp, tn=tn, fp=fp, fn=fn)


def _safe_div(num: float, den: float, zero_value: float = 0.0) -> float:
    """
    Controlla che la divisione avviene in modo sicuro, evitando errori di divisione per zero.
    :param num: Numeratore.
    :param den: Denominatore.
    :param zero_value: Valore restituito quando den == 0.
    :return: num/den se den != 0 altrimenti zero_value.
    """
    return float(num / den) if den != 0 else float(zero_value)


def accuracy_rate(c: ConfusionCounts) -> float:
    """
    Calcola l'Accuracy = (TP + TN) / (TP + TN + FP + FN).

    :param c: Conte della matrice di confusione.
    :return: Accuratezza in [0, 1].
    """
    return _safe_div(c.tp + c.tn, c.tp + c.tn + c.fp + c.fn)


def error_rate(c: ConfusionCounts) -> float:
    """
    Calcola l'Error Rate = 1 - Accuracy.

    :param c: Conte della matrice di confusione.
    :return: Tasso di errore in [0, 1].

    """
    return 1.0 - accuracy_rate(c)


def sensitivity(c: ConfusionCounts) -> float:
    """
    Calcola la Sensitivity (Recall della classe positiva) = TP / (TP + FN).

    :param c: Conte della matrice di confusione.
    :return: Sensibilità in [0, 1].

    """
    return _safe_div(c.tp, c.tp + c.fn)


def specificity(c: ConfusionCounts) -> float:
    """
    Calcola la Specificity (Recall della classe negativa) = TN / (TN + FP).

    :param c: Conte della matrice di confusione.
    :return: Specificità in [0, 1].

    """
    return _safe_div(c.tn, c.tn + c.fp)


def geometric_mean(c: ConfusionCounts) -> float:
    """
    Calcola la G-Mean = sqrt(Sensitivity * Specificity).

    :param c: Conte della matrice di confusione.
    :return: Media geometrica in [0, 1].

    """
    return float(np.sqrt(sensitivity(c) * specificity(c)))

def precision(c: ConfusionCounts) -> float:
    """
    Calcola la Precision = TP / (TP + FP).

    :param c: Conte della matrice di confusione.
    :return: Precision in [0, 1].

    """
    return _safe_div(c.tp, c.tp + c.fp)


def f1_score(c: ConfusionCounts) -> float:
    """
    Calcola l'F1-Score = 2 * (Precision * Recall) / (Precision + Recall).

    :param c: Conte della matrice di confusione.
    :return: F1-Score in [0, 1].

    """
    p = precision(c)
    r = sensitivity(c)
    return _safe_div(2 * p * r, p + r)


def roc_curve_manual(
    y_true,
    y_score,
    pos_label: int = 4,
    neg_label: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Costruisce la curva ROC (FPR, TPR) variando soglie su uno score continuo.

    Uso tipico in questo progetto: passare uno score continuo del KNN, ad es.:
    score = (#vicini_positivi) / k.

    Schema di implementazione:
    1) Ordina i campioni per score decrescente.
    2) Usa i valori di score unici come soglie (+inf per includere il punto iniziale).
    3) Per ogni soglia, assegna:
       - positivo se score >= soglia
       - negativo altrimenti
       quindi calcola TPR e FPR.

    :param y_true: Etichette vere (ground truth).
    :param y_score: Score continui (più alto => più probabile positivo).
    :param pos_label: Valore da considerare come classe positiva.
    :param neg_label: Valore da considerare come classe negativa (usato nelle predizioni a soglia).
    :return: (fpr, tpr, thresholds).

    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    # Ordinamento decrescente per costruzione standard della ROC
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]

    # Soglie candidate per la costruzione della ROC:
    # +inf consente di includere il punto iniziale (TPR = 0, FPR = 0),
    # mentre i valori distinti di y_score rappresentano tutte le soglie significative.
    thresholds = np.r_[np.inf, np.unique(y_score)]

    # Liste per memorizzare i valori di True Positive Rate (TPR)
    # e False Positive Rate (FPR) per ciascuna soglia
    tpr_list: List[float] = []
    fpr_list: List[float] = []

    # Numero totale di campioni positivi reali (P)
    # e di campioni negativi reali (N)
    P = int(np.sum(y_true == pos_label))
    N = int(np.sum(y_true != pos_label))

    for thr in thresholds:
        # Predizione binaria ottenuta applicando la soglia corrente:
        # un campione è classificato come positivo se y_score >= soglia,
        # altrimenti viene classificato come negativo
        y_pred_thr = np.where(y_score >= thr, pos_label, neg_label)

        # Calcolo dei conteggi della matrice di confusione
        c = confusion_counts(y_true, y_pred_thr, pos_label=pos_label)

        # Calcolo dei tassi:
        # TPR (True Positive Rate) = TP / P
        # FPR (False Positive Rate) = FP / N
        tpr_list.append(_safe_div(c.tp, P))
        fpr_list.append(_safe_div(c.fp, N))

    # Restituisce FPR, TPR e l'insieme delle soglie utilizzate
    return np.array(fpr_list), np.array(tpr_list), np.array(thresholds)


def calculate_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """
    Calcola l'AUC con integrazione ai trapezi.

    :param fpr: Vettore di False Positive Rate.
    :param tpr: Vettore di True Positive Rate.
    :return: Area sotto la curva ROC (AUC) in [0, 1].

    """
    sorted_indices = np.argsort(fpr)
    return float(np.trapezoid(tpr[sorted_indices], fpr[sorted_indices]))

def evaluate_metrics(
    y_true,
    y_pred,
    y_score: Optional[np.ndarray] = None,
    metrics: Optional[List[str]] = None,
    pos_label: int = 4,
    neg_label: int = 2
) -> Dict[str, float]:
    """
    Calcola un insieme di metriche di valutazione del classificatore.

    Metriche disponibili:
    - accuracy
    - error
    - sensitivity
    - specificity
    - precision
    - f1
    - gmean
    - auc (richiede y_score)

    :param y_true: Etichette vere.
    :param y_pred: Etichette predette.
    :param y_score: Score continuo (necessario per AUC).
    :param metrics: Lista di metriche da calcolare.
    :param pos_label: Classe positiva.
    :param neg_label: Classe negativa.
    :return: Dizionario {nome_metrica: valore}.

    """
    #Questa variabile mi serve a evitare errori legati a nomi di metriche errati.
    allowed_metrics = {
        "accuracy", "error",
        "sensitivity", "specificity",
        "precision", "f1",
        "gmean", "auc"}

    if metrics is None:
        metrics = ["accuracy", "error", "sensitivity", "specificity","precision","f1", "gmean","auc"]

    unknown = set(metrics) - allowed_metrics
    if unknown:
        raise ValueError(f"Metriche non riconosciute: {unknown}")

    c = confusion_counts(y_true, y_pred, pos_label=pos_label)
    out: Dict[str, float] = {}

    if "accuracy" in metrics:
        out["accuracy"] = accuracy_rate(c)
    if "error" in metrics:
        out["error"] = error_rate(c)
    if "sensitivity" in metrics:
        out["sensitivity"] = sensitivity(c)
    if "specificity" in metrics:
        out["specificity"] = specificity(c)
    if "precision" in metrics:
        out["precision"] = precision(c)
    if "f1" in metrics:
        out["f1"] = f1_score(c)
    if "gmean" in metrics:
        out["gmean"] = geometric_mean(c)
    if "auc" in metrics:
        if y_score is None:
            raise ValueError("AUC richiesta ma y_score è None.")
        if len(y_score) != len(y_true):
            raise ValueError(
                "y_score deve avere la stessa lunghezza di y_true "
                f"(ottenuti {len(y_score)} e {len(y_true)})."
            )

        fpr, tpr, _ = roc_curve_manual(
            y_true, y_score, pos_label=pos_label, neg_label=neg_label
        )
        out["auc"] = calculate_auc(fpr, tpr)

    return out
