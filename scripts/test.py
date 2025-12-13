import numpy as np

from preprocessing.loader import DataLoader
from knn.classifier import KNNClassifier
from validation.holdout import HoldoutValidation


# =========================================================================
# 2. FUNZIONI DI ESECUZIONE
# =========================================================================

def execute_holdout_validation(X: np.ndarray, y: np.ndarray):
    """Esegue la pipeline di validazione Holdout."""
    print("\n=== VALIDAZIONE HOLDOUT ===")

    # Configurazione
    knn_model = KNNClassifier(k=5, distance="euclidean", random_state=42)
    holdout_validator = HoldoutValidation(test_size=0.2, random_state=42)

    print(f"Modello: KNN (k={knn_model.k}, dist={knn_model.distance_name})")
    print(f"Split: {holdout_validator.test_size * 100:.0f}% Test")

    # Esecuzione
    try:
        accuracy_score = holdout_validator.validate(knn_model, X, y)
        print("\n--- Risultato Holdout ---")
        print(f"Accuratezza: {accuracy_score:.4f}")

    except Exception as e:
        print(f"Errore Holdout: {e}")


def execute_manual_knn_tests(X: np.ndarray, y: np.ndarray):
    """Esegue i test rapidi del KNN con diverse metriche di distanza."""
    print("\n" + "=" * 50)
    print("=== TEST METRICHE KNN ===")
    print("=" * 50)

    sample_X = X[:10]
    sample_y = y[:10]
    distances_to_test = ["euclidean", "manhattan", "chebyshev", "cosine"]

    for dist_name in distances_to_test:
        try:
            clf = KNNClassifier(k=5, distance=dist_name, random_state=42)
            clf.fit(X, y)

            preds = clf.predict(sample_X)
            accuracy = np.mean(preds == sample_y)

            print(f"\n--- {dist_name.capitalize()} ---")
            print("Predizioni:", preds)
            # print("True labels:", sample_y) # Rimosso per minimalismo
            print(f"Accuratezza (Top 10): {accuracy:.1f}")

        except Exception as e:
            print(f"Errore test {dist_name}: {e}")


# =========================================================================
# 3. FUNZIONE MAIN PRINCIPALE
# =========================================================================

def main():
    DATA_PATH = "../data/version_1.csv"

    print("--- 1. Caricamento Dati ---")

    try:
        data_loader = DataLoader(path=DATA_PATH)
        X, y, df_clean = data_loader.load()

        print(f"Shape: {X.shape}")

    except Exception as e:
        print(f"ERRORE CRITICO: Impossibile caricare i dati da '{DATA_PATH}'.")
        print(f"Errore: {e}")
        return

    execute_holdout_validation(X, y)
    execute_manual_knn_tests(X, y)

    # Salvataggio dei dati puliti
    OUTPUT_PATH = "../data/version_1_clean.csv"
    df_clean.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
