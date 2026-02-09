import unittest
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.main import main

class TestMainEndToEnd(unittest.TestCase):
    def setUp(self):
        self.base_dir = Path(__file__).resolve().parents[2]

        self.outputs_dir = self.base_dir / "outputs"
        self.data_dir = self.base_dir / "data"

        self.clean_data = self.data_dir / "version_1_clean.csv"

        if self.clean_data.exists():
            self.clean_data.unlink()

    def test_main_runs_end_to_end(self):
        sys.argv = [
            "main.py",
            "--no-interactive",
            "--mode", "holdout"
        ]

        main()

        self.assertTrue(self.clean_data.exists())

if __name__ == "__main__":
    unittest.main()