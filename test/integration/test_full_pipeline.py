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
        base_dir = Path(__file__).resolve().parent.parent.parent
        data_path = base_dir / "data" / "version_1_clean.csv"

        loader = DataLoader(str(data_path))
        cls.X, cls.y, _ = loader.load()

        cls.model = KNNClassifier(
            k=5,
            distance="euclidean",
            random_state=42
        )