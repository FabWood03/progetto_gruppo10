import sys
import traceback
import numpy as np
from pathlib import Path
from time import time

# --- SETUP AMBIENTE ---
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.preprocessing.loader import DataLoader
from src.knn.classifier import KNNClassifier
from src.validation.holdout import HoldoutValidation
from src.validation.k_fold import KFoldValidation
from src.metrics.plotter import plot_confusion_matrix, plot_roc_curve, plot_metric_distribution


# --- CONFIGURAZIONE GLOBALE ---
class Config:
    """Parametri di configurazione centralizzati."""
    # Parametri Modello
    K_NEIGHBORS = 5
    METRIC = "euclidean"
    RANDOM_SEED = 42

    # Parametri Validazione
    TEST_SPLIT = 0.2  # Per Holdout
    N_SPLITS = 5  # Per K-Fold

    # Selettore Modalità: "holdout", "kfold", "leavepout", "all"
    VALIDATION_MODE = "all"

    # Paths
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_HOLDOUT_PLOTS = BASE_DIR / "outputs" / "holdout_plots"
    OUTPUT_KFOLD_PLOTS = BASE_DIR / "outputs" / "kfold_plots"
    INPUT_FILE = DATA_DIR / "version_1.csv"
    OUTPUT_CLEAN_FILE = DATA_DIR / "version_1_clean.csv"


# --- HELPER DI OUTPUT (STAMPA & GRAFICI) ---
def print_scalar_metrics(metrics: dict, duration: float):
    """Stampa report per validazione Holdout."""
    print(f"\n{'=' * 40}")
    print(f"{'METRICA':<20} | {'VALORE':<10}")
    print(f"{'-' * 40}")
    for key, val in metrics.items():
        clean_name = key.replace('_', ' ').capitalize()
        print(f"{clean_name:<20} | {val:.4f}")
    print(f"{'-' * 40}")
    print(f"{'Time':<20} | {duration:.4f} s")
    print(f"{'=' * 40}\n")


def print_aggregate_metrics(summary: dict, duration: float):
    """Stampa report aggregato per la K-fold Cross Validation (Media +/- Std)."""
    print(f"\n{'=' * 60}")
    print(f"{'METRICA (CV)':<20} | {'MEDIA':<10} | {'STD DEV':<10}")
    print(f"{'-' * 60}")

    metric_names = sorted(list(set(
        k.replace("_mean", "").replace("_std", "") for k in summary.keys()
    )))

    for m in metric_names:
        mean_val = summary.get(f"{m}_mean", 0.0)
        std_val = summary.get(f"{m}_std", 0.0)
        clean_name = m.replace('_', ' ').capitalize()
        print(f"{clean_name:<20} | {mean_val:.4f}     | +/- {std_val:.4f}")

    print(f"{'-' * 60}")
    print(f"{'Total Time':<20} | {duration:.4f} s")
    print(f"{'=' * 60}\n")


def save_holdout_plots(results: dict, k: int):
    """Salva i grafici per la validazione Holdout."""
    Config.OUTPUT_HOLDOUT_PLOTS.mkdir(parents=True, exist_ok=True)
    labels = ["Benign (0)", "Malignant (1)"]

    plot_confusion_matrix(
        cm=results["confusion_matrix"], labels=labels,
        title=f"Confusion Matrix (K={k})",
        save_path=str(Config.OUTPUT_HOLDOUT_PLOTS / "holdout_cm_raw.png"), normalize=False
    )
    plot_confusion_matrix(
        cm=results["confusion_matrix"], labels=labels,
        title=f"Confusion Matrix Normalized (K={k})",
        save_path=str(Config.OUTPUT_HOLDOUT_PLOTS / "holdout_cm_norm.png"), normalize=True
    )
    fpr, tpr = results["roc_data"]
    plot_roc_curve(
        fpr=fpr, tpr=tpr, auc_value=results["metrics"]["auc"],
        title=f"ROC Curve (K={k})",
        save_path=str(Config.OUTPUT_HOLDOUT_PLOTS / "holdout_roc.png")
    )
    print(f"Grafici Holdout salvati in: {Config.OUTPUT_HOLDOUT_PLOTS}")


def save_kfold_plots(results: dict, k: int):
    """Salva i grafici per la K-Fold Cross Validation."""
    Config.OUTPUT_KFOLD_PLOTS.mkdir(parents=True, exist_ok=True)
    labels = ["Benign (0)", "Malignant (1)"]

    # 1. Matrice Aggregata (Somma di tutti i fold)
    plot_confusion_matrix(
        cm=results["aggregated_cm"], labels=labels,
        title=f"Aggregated Confusion Matrix (CV K={k})",
        save_path=str(Config.OUTPUT_KFOLD_PLOTS / "kfold_aggregated_cm.png"),
        normalize=True
    )

    # 2. Distribuzione Accuracy
    if "accuracy" in results["raw_metrics"]:
        plot_metric_distribution(
            values=results["raw_metrics"]["accuracy"],
            metric_name="Accuracy",
            title=f"Accuracy Distribution ({Config.N_SPLITS} Folds)",
            save_path=str(Config.OUTPUT_KFOLD_PLOTS / "kfold_accuracy_dist.png"),
            bins=Config.N_SPLITS
        )
    print(f"Grafici K-Fold salvati in: {Config.OUTPUT_KFOLD_PLOTS}")


# --- PIPELINE DI VALIDAZIONE ---
def run_holdout_pipeline(X: np.ndarray, y: np.ndarray):
    """Pipeline: Holdout Split 80/20."""
    print(f"\n=== AVVIO HOLDOUT (Split {Config.TEST_SPLIT:.0%}) ===")

    model = KNNClassifier(k=Config.K_NEIGHBORS, distance=Config.METRIC, random_state=Config.RANDOM_SEED)
    validator = HoldoutValidation(test_size=Config.TEST_SPLIT, random_state=Config.RANDOM_SEED)

    try:
        start = time()
        results = validator.validate(model, X, y)
        duration = time() - start

        print_scalar_metrics(results["metrics"], duration)
        save_holdout_plots(results, Config.K_NEIGHBORS)

    except Exception:
        print("Errore Holdout:")
        traceback.print_exc()


def run_kfold_pipeline(X: np.ndarray, y: np.ndarray):
    """Pipeline: K-Fold Cross Validation."""
    print(f"\n=== AVVIO K-FOLD ({Config.N_SPLITS} splits) ===")

    model = KNNClassifier(k=Config.K_NEIGHBORS, distance=Config.METRIC, random_state=Config.RANDOM_SEED)
    validator = KFoldValidation(n_splits=Config.N_SPLITS, random_state=Config.RANDOM_SEED)

    try:
        start = time()
        results = validator.validate(model, X, y)
        duration = time() - start

        print_aggregate_metrics(results["summary"], duration)
        save_kfold_plots(results, Config.K_NEIGHBORS)

    except Exception:
        print("Errore K-Fold:")
        traceback.print_exc()


# --- MAIN ---
def main():
    print("--- Pipeline Iniziata ---")

    if not Config.INPUT_FILE.exists():
        print(f"File non trovato: {Config.INPUT_FILE}")
        return

    try:
        # 1. Caricamento
        loader = DataLoader(path=str(Config.INPUT_FILE))

        X, y, df_clean = loader.load()

        print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        unique, counts = np.unique(y, return_counts=True)
        print(f"Target: {dict(zip(unique, counts))}")

        # 2. Esecuzione
        if Config.VALIDATION_MODE == "kfold":
            run_kfold_pipeline(X, y)
        elif Config.VALIDATION_MODE == "holdout":
            run_holdout_pipeline(X, y)
        elif Config.VALIDATION_MODE == "leavepout":
            print("Leave-P-Out non implementato.")
        elif Config.VALIDATION_MODE == "all":
            run_holdout_pipeline(X, y)
            run_kfold_pipeline(X, y)
        else:
            print(f"Modalità '{Config.VALIDATION_MODE}' non riconosciuta.")

        # 3. Salvataggio Dataset Pulito
        df_clean.to_csv(Config.OUTPUT_CLEAN_FILE, index=False)
        print(f"\nDataset pulito salvato in: {Config.OUTPUT_CLEAN_FILE}")

    except Exception as e:
        print(f"Errore inatteso: {e}")
        traceback.print_exc()

    print("--- Pipeline Terminata ---")


if __name__ == "__main__":
    main()
