import sys
import traceback
import numpy as np
from pathlib import Path
from time import time

# --- 1. SETUP AMBIENTE ---
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.preprocessing.loader import DataLoader
from src.knn.classifier import KNNClassifier
from src.validation.holdout import HoldoutValidation
from src.metrics.plotter import plot_confusion_matrix, plot_roc_curve


# --- 2. CONFIGURAZIONE GLOBALE ---
class Config:
    """Parametri di configurazione centralizzati."""
    K_NEIGHBORS = 5
    METRIC = "euclidean"
    TEST_SPLIT = 0.2
    RANDOM_SEED = 42

    # Paths relativi
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_PLOTS = BASE_DIR / "outputs" / "plots"
    INPUT_FILE = DATA_DIR / "version_1.csv"
    OUTPUT_CLEAN_FILE = DATA_DIR / "version_1_clean.csv"


# --- 3. FUNZIONI DI UTILITÀ (OUTPUT) ---
def print_metrics(metrics: dict, duration: float):
    """Stampa report metriche e tempo di esecuzione."""
    print(f"\n{'=' * 40}")
    print(f"{'METRICA':<20} | {'VALORE':<10}")
    print(f"{'-' * 40}")

    for key, val in metrics.items():
        # Formatta: "sensitivity" -> "Sensitivity"
        clean_name = key.replace('_', ' ').capitalize()
        print(f"{clean_name:<20} | {val:.4f}")

    print(f"{'-' * 40}")
    print(f"{'Execution Time':<20} | {duration:.4f} sec")
    print(f"{'=' * 40}\n")


def save_validation_plots(results: dict, k: int):
    """Gestisce il salvataggio di tutti i grafici."""
    Config.OUTPUT_PLOTS.mkdir(parents=True, exist_ok=True)

    # Etichette corrette per il dataset (0=Benigno, 1=Maligno)
    labels = ["Benign (0)", "Malignant (1)"]

    # 1. Matrice Confusione (Conteggi)
    plot_confusion_matrix(
        cm=results["confusion_matrix"],
        labels=labels,
        title=f"Confusion Matrix (K={k})",
        save_path=str(Config.OUTPUT_PLOTS / "holdout_cm_raw.png"),
        normalize=False
    )

    # 2. Matrice Confusione (Normalizzata)
    plot_confusion_matrix(
        cm=results["confusion_matrix"],
        labels=labels,
        title=f"Confusion Matrix Normalized (K={k})",
        save_path=str(Config.OUTPUT_PLOTS / "holdout_cm_norm.png"),
        normalize=True
    )

    # 3. ROC Curve
    fpr, tpr = results["roc_data"]
    plot_roc_curve(
        fpr=fpr,
        tpr=tpr,
        auc_value=results["metrics"]["auc"],
        title=f"ROC Curve (K={k})",
        save_path=str(Config.OUTPUT_PLOTS / "holdout_roc.png")
    )
    print(f"Grafici salvati in: {Config.OUTPUT_PLOTS}")


# --- 4. CORE LOGIC ---
def run_holdout_pipeline(X: np.ndarray, y: np.ndarray):
    """Esegue la validazione misurando le performance."""
    print("\n=== AVVIO VALIDAZIONE HOLDOUT ===")
    print(f"Config: KNN(k={Config.K_NEIGHBORS}, dist='{Config.METRIC}') | Split: {Config.TEST_SPLIT:.0%}")

    # Setup Modello
    model = KNNClassifier(
        k=Config.K_NEIGHBORS,
        distance=Config.METRIC,
        random_state=Config.RANDOM_SEED
    )

    validator = HoldoutValidation(
        test_size=Config.TEST_SPLIT,
        random_state=Config.RANDOM_SEED
    )

    try:
        # Misurazione tempo di inferenza
        start_time = time()

        # Validazione (ritorna dict con metriche e dati raw)
        results = validator.validate(model, X, y)

        elapsed_time = time() - start_time

        # Output
        print_metrics(results["metrics"], elapsed_time)
        save_validation_plots(results, Config.K_NEIGHBORS)

    except Exception:
        print("Errore critico durante la validazione:")
        traceback.print_exc()


def main():
    """Entry point principale."""
    print("--- Pipeline Iniziata ---")

    # 1. Caricamento Dati
    if not Config.INPUT_FILE.exists():
        print(f"File non trovato: {Config.INPUT_FILE}")
        return

    try:
        loader = DataLoader(path=str(Config.INPUT_FILE))
        X, y, df_clean = loader.load()

        print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

        # --- FIX: Calcolo distribuzione esplicito per evitare warning IDE ---
        unique_vals, counts = np.unique(y, return_counts=True)
        dist_target = dict(zip(unique_vals, counts))
        print(f"Distribuzione Target: {dist_target}")
        # ------------------------------------------------------------------

        # 2. Esecuzione Test
        run_holdout_pipeline(X, y)

        # 3. Salvataggio Dataset Pulito
        df_clean.to_csv(Config.OUTPUT_CLEAN_FILE, index=False)
        print(f"\nDataset pulito salvato in: {Config.OUTPUT_CLEAN_FILE}")

    except Exception as e:
        print(f"Errore inatteso: {e}")
        traceback.print_exc()

    print("--- Pipeline Terminata ---")


if __name__ == "__main__":
    main()
