import unittest
from pathlib import Path
import numpy as np

from src.preprocessing.loader import DataLoader
from src.knn.classifier import KNNClassifier
from src.validation.holdout import HoldoutValidation
from src.validation.k_fold import KFoldValidation
from src.validation.leave_p_out import LeavePOutValidation

class TestFullPipelineIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        base_dir = Path(__file__).resolve().parents[2]

        data_path = base_dir / "data" / "version_1.csv"
        assert data_path.exists(), f"Dataset not found: {data_path}"

        loader = DataLoader(str(data_path))
        cls.X, cls.y, _ = loader.load()

        cls.model = KNNClassifier(
            k=5,
            distance="euclidean",
            random_state=42
        )

    def test_holdout_pipeline_runs(self):
        validator = HoldoutValidation(test_size=0.2, random_state=42)
        result = validator.validate(self.model, self.X, self.y)

        self.assertIn("metrics", result)
        self.assertIn("confusion_matrix", result)
        self.assertEqual(result["confusion_matrix"].shape, (2, 2))

        for value in result["metrics"].values():
            self.assertFalse(np.isnan(value))

    def test_kfold_pipeline_runs(self):
        validator = KFoldValidation(n_splits=5, random_state=42)
        result = validator.validate(self.model, self.X, self.y)

        self.assertIn("summary", result)
        self.assertIn("aggregated_cm", result)
        self.assertEqual(result["aggregated_cm"].shape, (2, 2))

        for value in result["summary"].values():
            self.assertFalse(np.isnan(value))

    def test_leave_p_out_pipeline_runs(self):
        validator = LeavePOutValidation(p=1)
        result = validator.validate(self.model, self.X, self.y)

        self.assertIn("summary", result)
        self.assertIn("raw_metrics", result)
        self.assertEqual(result["aggregated_cm"].shape, (2, 2))
        self.assertGreater(result["n_iterations"], 0)

if __name__ == "__main__":
    unittest.main()
