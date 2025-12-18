import sys
import traceback
from pathlib import Path
from time import time

import numpy as np

# --- Setup Path ---
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.preprocessing.loader import DataLoader
from src.knn.classifier import KNNClassifier
from src.validation.holdout import HoldoutValidation
from src.validation.k_fold import KFoldValidation
from src.validation.leave_p_out import LeavePOutValidation
from src.metrics.plotter import plot_confusion_matrix, plot_roc_curve, plot_metric_distribution


class Config:
    """Configurazione parametri e percorsi."""
    # Modello
    K_NEIGHBORS = 5
    METRIC = "euclidean"
    RANDOM_SEED = 42

    # Validazione
    TEST_SPLIT = 0.2  # Holdout
    N_SPLITS = 5  # K-Fold
    LPO_P = 1  # Leave-P-Out (Consigliato: 1)

    # Modalità: "holdout", "kfold", "leavepout", "all"
    VALIDATION_MODE = "all"

    # Paths
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "outputs"
    INPUT_FILE = DATA_DIR / "version_1.csv"
    OUTPUT_CLEAN_FILE = DATA_DIR / "version_1_clean.csv"


def _get_model() -> KNNClassifier:
    """Factory per istanziare il modello con la config corrente."""
    return KNNClassifier(
        k=Config.K_NEIGHBORS,
        distance=Config.METRIC,
        random_state=Config.RANDOM_SEED
    )


def print_scalar_metrics(metrics: dict, duration: float):
    print(f"\n{'=' * 40}")
    print(f"{'METRICA':<20} | {'VALORE':<10}")
    print(f"{'-' * 40}")
    for key, val in metrics.items():
        print(f"{key.replace('_', ' ').capitalize():<20} | {val:.4f}")
    print(f"{'-' * 40}")
    print(f"{'Duration':<20} | {duration:.4f} s")
    print(f"{'=' * 40}\n")


def print_aggregate_metrics(summary: dict, duration: float):
    print(f"\n{'=' * 60}")
    print(f"{'METRICA (CV)':<20} | {'MEDIA':<10} | {'STD DEV':<10}")
    print(f"{'-' * 60}")

    metric_names = sorted(list(set(
        k.replace("_mean", "").replace("_std", "") for k in summary.keys()
    )))

    for m in metric_names:
        mean = summary.get(f"{m}_mean", 0.0)
        std = summary.get(f"{m}_std", 0.0)
        print(f"{m.replace('_', ' ').capitalize():<20} | {mean:.4f}     | +/- {std:.4f}")

    print(f"{'-' * 60}")
    print(f"{'Total Time':<20} | {duration:.4f} s")
    print(f"{'=' * 60}\n")


def save_holdout_plots(results: dict):
    out_dir = Config.OUTPUT_DIR / "holdout_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = ["Benigno (0)", "Maligno (1)"]

    plot_confusion_matrix(
        results["confusion_matrix"], labels, f"Matrice di confusione (K={Config.K_NEIGHBORS})",
        str(out_dir / "holdout_cm_raw.png"), normalize=False
    )
    plot_confusion_matrix(
        results["confusion_matrix"], labels, f"Matrice di confusione normalizzata (K={Config.K_NEIGHBORS})",
        str(out_dir / "holdout_cm_norm.png"), normalize=True
    )
    fpr, tpr = results["roc_data"]
    plot_roc_curve(
        fpr, tpr, results["metrics"]["auc"], f"Curva ROC (K={Config.K_NEIGHBORS})",
        str(out_dir / "holdout_roc.png")
    )
    print(f"Grafici Holdout salvati in: {out_dir}")


def save_cv_plots(results: dict, method_name: str):
    """Gestisce salvataggio plot per K-Fold e Leave-P-Out."""
    out_dir = Config.OUTPUT_DIR / f"{method_name}_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = ["Benigno (0)", "Maligno (1)"]

    plot_confusion_matrix(
        results["aggregated_cm"], labels, f"CM aggregata ({method_name})",
        str(out_dir / "aggregated_cm.png"), normalize=True
    )

    if "accuracy" in results.get("raw_metrics", {}):
        plot_metric_distribution(
            results["raw_metrics"]["accuracy"], "Accuracy", f"Distribuzione dell'accuratezza ({method_name})",
            str(out_dir / "accuracy_dist.png")
        )
    print(f"Grafici {method_name} salvati in: {out_dir}")


def run_holdout(X: np.ndarray, y: np.ndarray):
    print(f"\n=== AVVIO HOLDOUT (Split {Config.TEST_SPLIT:.0%}) ===")
    validator = HoldoutValidation(test_size=Config.TEST_SPLIT, random_state=Config.RANDOM_SEED)

    try:
        start = time()
        results = validator.validate(_get_model(), X, y)
        duration = time() - start

        print_scalar_metrics(results["metrics"], duration)
        save_holdout_plots(results)
    except Exception:
        print("Errore Holdout:")
        traceback.print_exc()


def run_kfold(X: np.ndarray, y: np.ndarray):
    print(f"\n=== AVVIO K-FOLD ({Config.N_SPLITS} splits) ===")
    validator = KFoldValidation(n_splits=Config.N_SPLITS, random_state=Config.RANDOM_SEED)

    try:
        start = time()
        results = validator.validate(_get_model(), X, y)
        duration = time() - start

        print_aggregate_metrics(results["summary"], duration)
        save_cv_plots(results, "kfold")
    except Exception:
        print("Errore K-Fold:")
        traceback.print_exc()


def run_lpo(X: np.ndarray, y: np.ndarray):
    print(f"\n=== AVVIO LEAVE-P-OUT (p={Config.LPO_P}) ===")
    validator = LeavePOutValidation(p=Config.LPO_P)

    try:
        start = time()
        results = validator.validate(_get_model(), X, y)
        duration = time() - start

        print_aggregate_metrics(results["summary"], duration)
        save_cv_plots(results, "leavepout")
    except Exception:
        print("Errore Leave-P-Out:")
        traceback.print_exc()


def main():
    if not Config.INPUT_FILE.exists():
        print(f"ERRORE: File non trovato: {Config.INPUT_FILE}")
        return

    try:
        loader = DataLoader(str(Config.INPUT_FILE))
        X, y, df_clean = loader.load()

        print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

        modes = {
            "holdout": run_holdout,
            "kfold": run_kfold,
            "leavepout": run_lpo
        }

        if Config.VALIDATION_MODE == "all":
            for mode in modes.values():
                mode(X, y)
        elif Config.VALIDATION_MODE in modes:
            modes[Config.VALIDATION_MODE](X, y)
        else:
            print(f"Modalità '{Config.VALIDATION_MODE}' non valida.")

        df_clean.to_csv(Config.OUTPUT_CLEAN_FILE, index=False)

    except Exception as e:
        print(f"Errore inatteso: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
