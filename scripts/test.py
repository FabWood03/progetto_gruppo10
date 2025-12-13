import sys
import numpy as np

# Aggiunge la directory corrente al path per permettere l'import dei moduli locali
sys.path.append(".")

from preprocessing.loader import DataLoader
from knn.classifier import KNNClassifier
from validation.holdout import HoldoutValidation


def execute_holdout_validation(X: np.ndarray, y: np.ndarray) -> None:
    """
    Esegue la pipeline di validazione Holdout sul dataset.

    Istanzia il classificatore KNN e il validatore Holdout, esegue il training
    e il testing su uno split 80/20, stampando l'accuratezza finale.
    :param X: Matrice delle feature.
    :param y: Vettore delle etichette target.
    """
    print("\n=== VALIDAZIONE HOLDOUT ===")

    # Configurazione parametri
    k_neighbors = 5
    metric = "euclidean"
    split_ratio = 0.2
    seed = 42

    knn_model = KNNClassifier(k=k_neighbors, distance=metric, random_state=seed)
    validator = HoldoutValidation(test_size=split_ratio, random_state=seed)

    print(f"Configurazione: KNN(k={k_neighbors}, dist='{metric}') | Split Test: {split_ratio:.0%}")

    try:
        accuracy = validator.validate(knn_model, X, y)
        print(f"Risultato (Accuratezza): {accuracy:.4f}")
    except Exception as e:
        print(f"Errore durante la validazione: {e}")


def execute_manual_knn_tests(X: np.ndarray, y: np.ndarray) -> None:
    """
    Esegue test rapidi di inferenza su un sottoinsieme di dati variando la metrica.

    Itera su diverse metriche di distanza (Euclidea, Manhattan, Chebyshev, Cosine)
    per verificare che il modello produca predizioni senza errori a runtime.

    :param X: Matrice delle feature.
    :param y: Vettore delle etichette target.
    """
    print("\n=== TEST COMPARATIVO METRICHE ===")

    # Subset per test rapido (primi 10 campioni)
    sample_X = X[:10]
    sample_y = y[:10]
    metrics = ["euclidean", "manhattan", "chebyshev", "cosine"]

    for metric in metrics:
        try:
            # Istanzia e addestra il modello per la metrica corrente
            clf = KNNClassifier(k=5, distance=metric, random_state=42)
            clf.fit(X, y)

            # Inferenza
            preds = clf.predict(sample_X)
            acc = np.mean(preds == sample_y)

            print(f"Metrica: {metric:<10} | Accuratezza (Top 10): {acc:.1f}")
        except Exception as e:
            print(f"Metrica: {metric:<10} | Errore: {e}")


def main():
    """
    Entry point dello script.
    Carica i dati, esegue la validazione e salva il dataset pulito.
    """
    input_csv = "../data/version_1.csv"
    output_csv = "../data/version_1_clean.csv"

    print("--- Avvio Pipeline ---")

    try:
        loader = DataLoader(path=input_csv)
        X, y, df_clean = loader.load()

        print(f"Dataset caricato: {X.shape[0]} campioni, {X.shape[1]} feature")
        print(f"Distribuzione target: {np.unique(y, return_counts=True)}")

    except Exception as e:
        print(f"ERRORE CRITICO: Impossibile caricare '{input_csv}'.\nDettagli: {e}")
        return

    # Esecuzione pipeline
    execute_holdout_validation(X, y)
    execute_manual_knn_tests(X, y)

    # Salvataggio output
    try:
        df_clean.to_csv(output_csv, index=False)
        print(f"\nDataset pulito salvato in: {output_csv}")
    except Exception as e:
        print(f"Errore nel salvataggio del CSV: {e}")


if __name__ == "__main__":
    main()
