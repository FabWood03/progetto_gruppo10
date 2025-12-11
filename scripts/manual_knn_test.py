"""
Script di test manuale per verificare il corretto funzionamento del KNN.

Eseguire con:
    python scripts/manual_knn_test.py
"""

from src.preprocessing.loader import DataLoader
from src.knn.classifier import KNNClassifier

def main():
    print("=== TEST MANUALE KNN ===")

    # 1. Caricamento e preprocessing
    loader = DataLoader("../data/version_1.csv")


    X, y, df_clean = loader.load()

    OUTPUT_PATH = "../data/version_1_clean.csv"  # Definisci un nuovo percorso
    df_clean.to_csv(OUTPUT_PATH, index=False)
    print(f"\nCSV pulito salvato in: {OUTPUT_PATH}")

    print("\n--- Dataset Info ---")
    print(f"Shape X: {X.shape}")
    print(f"Shape y: {y.shape}")
    print(f"Prime etichette: {y[:10]}")

    # -----------------------------
    # Test con distanza Euclidea
    # -----------------------------
    print("\n=== Test con distanza: Euclidean ===")
    clf_euc = KNNClassifier(k=5, distance="euclidean", random_state=42)
    clf_euc.fit(X, y)

    preds_euc = clf_euc.predict(X[:10])
    print("Predictions (Euclidean):", preds_euc)
    print("True labels:",            y[:10])

    # -----------------------------
    # Test con distanza Manhattan
    # -----------------------------
    print("\n=== Test con distanza: Manhattan ===")
    clf_man = KNNClassifier(k=5, distance="manhattan", random_state=42)
    clf_man.fit(X, y)

    preds_man = clf_man.predict(X[:10])
    print("Predictions (Manhattan):", preds_man)
    print("True labels:",            y[:10])

    # -----------------------------
    # Test con distanza Chebyshev
    # -----------------------------
    print("\n=== Test con distanza: Chebyshev ===")
    clf_cheb = KNNClassifier(k=5, distance="chebyshev", random_state=42)
    clf_cheb.fit(X, y)

    preds_cheb = clf_cheb.predict(X[:10])
    print("Predictions (Chebyshev):", preds_cheb)
    print("True labels:",            y[:10])

    # -----------------------------
    # Test con distanza Cosine
    # -----------------------------
    print("\n=== Test con distanza: Cosine ===")
    clf_cos = KNNClassifier(k=5, distance="cosine", random_state=42)
    clf_cos.fit(X, y)

    preds_cos = clf_cos.predict(X[:10])
    print("Predictions (Cosine):", preds_cos)
    print("True labels:",          y[:10])

if __name__ == "__main__":
    main()

