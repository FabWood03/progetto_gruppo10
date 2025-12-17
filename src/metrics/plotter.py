from __future__ import annotations

import itertools
from pathlib import Path
from typing import Sequence, Union

import matplotlib.pyplot as plt
import numpy as np

# Tipo per i percorsi: accetta stringhe od oggetti Path
PathType = Union[str, Path]


def _prepare_plot(save_path: PathType, title: str) -> tuple[plt.Figure, plt.Axes]:
    """
    Helper per inizializzare figura, assi e creare la directory di output.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.set_title(title)
    return fig, ax


def _save_and_close(fig: plt.Figure, save_path: PathType) -> None:
    """
    Salva la figura ottimizzando il layout e libera la memoria.
    """
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(
        cm: np.ndarray,
        labels: Sequence[str],
        title: str,
        save_path: PathType,
        normalize: bool = False
) -> None:
    """
    Genera e salva la matrice di confusione.
    Normalizza opzionalmente i valori per riga.
    """
    cm = np.asarray(cm)

    # Validazione dimensioni
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1] or len(labels) != cm.shape[0]:
        raise ValueError("Dimensioni della matrice o etichette non coerenti.")

    # Calcolo valori per il plot (normalizzati o grezzi)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        # np.divide gestisce divisione per zero in modo sicuro
        cm_plot = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    else:
        cm_plot = cm.astype(float)

    # Creazione Plot
    fig, ax = _prepare_plot(save_path, title)

    # Mappa colori
    im = ax.imshow(cm_plot, interpolation='nearest', cmap="Blues")
    fig.colorbar(im, ax=ax)

    # Configurazione assi
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    # Inserimento testo nelle celle (ottimizzato con itertools)
    thresh = cm_plot.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        val = cm_plot[i, j]
        label_text = f"{val:.2f}" if normalize else f"{int(val)}"

        # Colore testo adattivo (bianco su scuro, nero su chiaro)
        text_color = "white" if val > thresh else "black"

        ax.text(j, i, label_text, horizontalalignment="center", color=text_color)

    _save_and_close(fig, save_path)


def plot_roc_curve(
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc_value: float,
        title: str,
        save_path: PathType
) -> None:
    """
    Genera e salva la curva ROC con il valore AUC.
    """
    fig, ax = _prepare_plot(save_path, title)

    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc_value:.3f}')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')

    # FIX: Passiamo argomenti separati (float) invece di una lista
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")

    _save_and_close(fig, save_path)


def plot_metric_distribution(
        values: list[float] | np.ndarray,
        metric_name: str,
        title: str,
        save_path: PathType,
        bins: int = 10
) -> None:
    """
    Genera istogramma per la distribuzione di una metrica (es. su K-Fold).
    """
    fig, ax = _prepare_plot(save_path, title)

    ax.hist(values, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)

    ax.set_xlabel(metric_name)
    ax.set_ylabel('Frequency')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    _save_and_close(fig, save_path)
