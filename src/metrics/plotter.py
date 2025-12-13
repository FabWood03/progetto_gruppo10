from __future__ import annotations
from typing import List
import os
import numpy as np
import matplotlib.pyplot as plt

def _ensure_output_dir(save_path: str) -> None:
    """
    Crea la directory di output associata al percorso del file, se non esiste.

    :param save_path: Percorso del file di output.
    :return: None.
    """
    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """
    Restituisce una copia della matrice con righe normalizzate
    (ogni riga somma a 1, quando possibile).

    :param matrix: Matrice di input.
    :return: Matrice normalizzata per righe.
    """
    matrix = matrix.astype(float, copy=True)
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix),
        where=row_sums != 0
    )


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    title: str,
    save_path: str,
    normalize: bool = False
) -> None:
    """
    Visualizza e salva la matrice di confusione come immagine.

    Le righe rappresentano le classi reali, mentre le colonne
    rappresentano le classi predette.

    :param cm: Matrice di confusione (tipicamente 2x2).
    :param labels: Etichette delle classi da mostrare sugli assi.
    :param title: Titolo del grafico.
    :param save_path: Percorso del file di output.
    :param normalize: Se True, normalizza le righe della matrice.
    :return: None.
    """
    # Assicura l'esistenza della directory di output
    _ensure_output_dir(save_path)

    # Controllo di coerenza tra dimensione della matrice e numero di etichette
    n_labels = len(labels)
    if cm.shape[0] != n_labels or cm.shape[1] != n_labels:
        raise ValueError(
            f"`cm` shape {cm.shape} must match number of labels ({n_labels})."
        )

    # Se richiesto, normalizza le righe della matrice
    cm_to_plot = _normalize_rows(cm) if normalize else cm.astype(float, copy=False)

    # Creazione della figura e visualizzazione della matrice
    fig, ax = plt.subplots()
    im = ax.imshow(cm_to_plot, cmap="Blues")

    # Impostazione titolo, assi ed etichette
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(n_labels))
    ax.set_yticks(range(n_labels))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Inserimento dei valori numerici all'interno delle celle
    for i in range(cm_to_plot.shape[0]):
        for j in range(cm_to_plot.shape[1]):
            value = f"{cm_to_plot[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, value, ha="center", va="center")

    # Aggiunge la colorbar per migliorare la leggibilità
    fig.colorbar(im, ax=ax)

    # Ottimizza il layout e salva l'immagine
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)

    # Chiude la figura per liberare memoria
    plt.close(fig)


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_value: float,
    title: str,
    save_path: str
) -> None:
    """
    Visualizza e salva la curva ROC.

    :param fpr: Vettore di False Positive Rate.
    :param tpr: Vettore di True Positive Rate.
    :param auc_value: Valore dell'AUC da mostrare in legenda.
    :param title: Titolo del grafico.
    :param save_path: Percorso del file di output.
    :return: None.
    """
    # Assicura l'esistenza della directory di output
    _ensure_output_dir(save_path)

    fig, ax = plt.subplots()

    # Curva ROC
    ax.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}")

    # Linea di riferimento per classificatore casuale
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random")

    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_metric_distribution(
    values: List[float],
    metric_name: str,
    title: str,
    save_path: str,
    bins: int = 10
) -> None:
    """
    Visualizza e salva la distribuzione di una metrica di valutazione
    calcolata su più esperimenti (ad esempio su diversi fold di cross-validation).

    La funzione rappresenta graficamente i valori ottenuti tramite le funzioni di evaluator.py,
    permettendo di analizzare la variabilità delle prestazioni del modello.
    :param values: Lista dei valori della metrica.
    :param metric_name: Nome della metrica (asse x).
    :param title: Titolo del grafico.
    :param save_path: Percorso del file di output.
    :param bins: Numero di bin dell'istogramma.
    :return: None.
    """
    # Assicura l'esistenza della directory di output
    _ensure_output_dir(save_path)

    # Creazione della figura e visualizzazione della matrice
    fig, ax = plt.subplots()

    ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(metric_name)
    ax.set_ylabel("Frequency")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
