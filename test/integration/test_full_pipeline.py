import unittest
from pathlib import Path
import numpy as np

from src.preprocessing.loader import DataLoader
from src.knn.classifier import KNNClassifier
from src.validation.holdout import HoldoutValidation
from src.validation.k_fold import KFoldValidation
from src.validation.leave_p_out import LeavePOutValidation


"""
Questo file testa l'integrazione completa della pipeline:

Dataset -> Preprocessing -> KNN -> Validazione -> Metriche

È un test end-to-end che verifica che tutte le componenti
lavorino correttamente insieme.
"""


class TestFullPipelineIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Carico il dataset reale una sola volta per tutti i test.
        Questo simula l'esecuzione reale della pipeline.
        """
        base_dir = Path(__file__).resolve().parents[2]

        data_path = base_dir / "data" / "version_1.csv"
        assert data_path.exists(), f"Dataset not found: {data_path}"

        # Caricamento e preprocessing
        loader = DataLoader(str(data_path))
        cls.X, cls.y, _ = loader.load()

        # Inizializzo modello reale
        cls.model = KNNClassifier(
            k=5,
            distance="euclidean",
            random_state=42
        )

    # =====================================================
    # HOLDOUT
    # =====================================================

    def test_holdout_pipeline_runs(self):
        """
        Verifico che la pipeline completa funzioni
        con validazione Holdout.
        """
        validator = HoldoutValidation(test_size=0.2, random_state=42)
        result = validator.validate(self.model, self.X, self.y)

        # Struttura output
        self.assertIn("metrics", result)
        self.assertIn("confusion_matrix", result)

        # Matrice corretta
        self.assertEqual(result["confusion_matrix"].shape, (2, 2))

        # Nessuna metrica deve essere NaN
        for value in result["metrics"].values():
            self.assertFalse(np.isnan(value))

    # =====================================================
    # K-FOLD
    # =====================================================

    def test_kfold_pipeline_runs(self):
        """
        Verifico esecuzione completa con K-Fold.
        Controllo:
        - summary
        - matrice aggregata
        - stabilità numerica
        """
        validator = KFoldValidation(n_splits=5, random_state=42)
        result = validator.validate(self.model, self.X, self.y)

        self.assertIn("summary", result)
        self.assertIn("aggregated_cm", result)

        self.assertEqual(result["aggregated_cm"].shape, (2, 2))

        for value in result["summary"].values():
            self.assertFalse(np.isnan(value))

    # =====================================================
    # LEAVE-P-OUT
    # =====================================================

    def test_leave_p_out_pipeline_runs(self):
        """
        Verifico che Leave-P-Out funzioni
        con modello reale.
        """
        validator = LeavePOutValidation(p=1)
        result = validator.validate(self.model, self.X, self.y)

        self.assertIn("summary", result)
        self.assertIn("raw_metrics", result)

        self.assertEqual(result["aggregated_cm"].shape, (2, 2))

        # Deve eseguire almeno una iterazione
        self.assertGreater(result["n_iterations"], 0)


if __name__ == "__main__":
    unittest.main()
