import sys
import numpy as np
import traceback
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Import moduli locali
from src.preprocessing.loader import DataLoader
from src.knn.classifier import KNNClassifier
from src.validation.holdout import HoldoutValidation
from src.metrics.plotter import plot_confusion_matrix, plot_roc_curve


def print_metrics_report(metrics: dict):
    """Stampa una tabella pulita delle metriche."""
    print("\n" + "=" * 40)
    print(f"{'METRICA':<20} | {'VALORE':<10}")
    print("-" * 40)
    for name, value in metrics.items():
        # Formattazione: prima lettera maiuscola, 4 decimali
        print(f"{name.replace('_', ' ').capitalize():<20} | {value:.4f}")
    print("=" * 40 + "\n")


def execute_holdout_validation(x: np.ndarray, y: np.ndarray, output_dir: Path) -> None:
    """
    Esegue la pipeline di validazione Holdout, stampa i risultati e salva i grafici.
    """
    print("\n=== AVVIO VALIDAZIONE HOLDOUT ===")

    # 1. Configurazione
    k_neighbors = 5
    metric = "euclidean"
    split_ratio = 0.2
    seed = 42

    print(f"Configurazione: KNN(k={k_neighbors}, dist='{metric}') | Split: {split_ratio:.0%}")

    # 2. Istanziazione
    knn_model = KNNClassifier(k=k_neighbors, distance=metric, random_state=seed)
    validator = HoldoutValidation(test_size=split_ratio, random_state=seed)

    try:
        # 3. Esecuzione (Il metodo ora RITORNA i dati, non stampa)
        results = validator.validate(knn_model, x, y)

        # 4. Reporting Metriche
        print_metrics_report(results["metrics"])

        # 5. Generazione Grafici
        # Creiamo la cartella output se non esiste
        output_dir.mkdir(parents=True, exist_ok=True)

        # Plot Matrice di Confusione
        plot_confusion_matrix(
            cm=results["confusion_matrix"],
            labels=["Benign (0)", "Malignant (1)"],
            title=f"Confusion Matrix (KNN k={k_neighbors})",
            save_path=str(output_dir / "holdout_confusion_matrix.png"),
            normalize=False
        )

        plot_confusion_matrix(
            cm=results["confusion_matrix"],
            labels=["Malignant (4)", "Benign (2)"],
            title="Confusion Matrix (Normalized)",
            save_path=str(output_dir / "test1_confusion_matrix_normalized.png"),
            normalize=True
        )

        # Plot Curva ROC
        fpr, tpr = results["roc_data"]
        plot_roc_curve(
            fpr=fpr,
            tpr=tpr,
            auc_value=results["metrics"]["auc"],
            title=f"ROC Curve (KNN k={k_neighbors})",
            save_path=str(output_dir / "holdout_roc_curve.png")
        )

        print(f"✅ Grafici salvati in: {output_dir}")

    except Exception as e:
        print(f"❌ Errore durante la validazione:")
        traceback.print_exc()


def main():
    """
    Entry point dello script.
    """
    # Definizione percorsi relativi alla posizione dello script
    data_dir = BASE_DIR / "data"
    output_plots_dir = BASE_DIR / "outputs" / "plots"

    input_csv = data_dir / "version_1.csv"
    output_clean_csv = data_dir / "version_1_clean.csv"

    print("--- Pipeline Iniziata ---")

    # 1. Caricamento Dati
    try:
        if not input_csv.exists():
            raise FileNotFoundError(f"File non trovato: {input_csv}")

        loader = DataLoader(path=str(input_csv))
        X, y, df_clean = loader.load()

        print(f"Dataset caricato: {X.shape[0]} righe, {X.shape[1]} colonne")
        unique, counts = np.unique(y, return_counts=True)
        print(f"Distribuzione target: {dict(zip(unique, counts))}")

    except Exception as e:
        print(f"❌ ERRORE CRITICO nel caricamento dati: {e}")
        return

    # 2. Esecuzione Validazione
    execute_holdout_validation(X, y, output_dir=output_plots_dir)

    # 3. Salvataggio Dataset Pulito
    try:
        df_clean.to_csv(output_clean_csv, index=False)
        print(f"\nDataset pulito salvato in: {output_clean_csv}")
    except Exception as e:
        print(f"⚠️ Errore nel salvataggio del CSV: {e}")

    print("\n--- Pipeline Terminata ---")


if __name__ == "__main__":
    main()
