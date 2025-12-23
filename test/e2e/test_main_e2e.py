import unittest
import sys
from pathlib import Path
import pandas as pd

# Importiamo il main reale
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from scripts.test import main

class TestMainEndToEnd(unittest.TestCase):
    def setUp(self):
        self.base_dir = Path(__file__).resolve().parents[2]

        self.outputs_dir = self.base_dir / "outputs"
        self.data_dir = self.base_dir / "data"

        self.final_results = self.outputs_dir / "final_results.csv"
        self.clean_data = self.data_dir / "version_1_clean.csv"

        # Rimuoviamo eventuali file precedenti
        if self.final_results.exists():
            self.final_results.unlink()

        if self.clean_data.exists():
            self.clean_data.unlink()

    def test_main_runs_end_to_end(self):
        # Esecuzione completa del programma
        main()

        # Verifica che i file finali siano stati creati
        self.assertTrue(self.final_results.exists())
        self.assertTrue(self.clean_data.exists())

        # Verifica contenuto CSV finale
        df = pd.read_csv(self.final_results)
        self.assertFalse(df.empty)

        for col in ["method", "metric", "mean", "std"]:
            self.assertIn(col, df.columns)

if __name__ == "__main__":
    unittest.main()