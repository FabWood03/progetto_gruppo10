import unittest
import sys
from pathlib import Path

# Aggiungo la root del progetto al path
# Serve per simulare l’esecuzione reale da terminale
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.main import main


"""
Questo test verifica l’esecuzione completa del programma
simulando l’avvio da linea di comando.

È un test end-to-end reale.
"""


class TestMainEndToEnd(unittest.TestCase):

    def setUp(self):
        """
        Preparo ambiente pulito prima del test.
        Elimino eventuali file generati precedentemente.
        """
        self.base_dir = Path(__file__).resolve().parents[2]

        self.outputs_dir = self.base_dir / "outputs"
        self.data_dir = self.base_dir / "data"

        self.clean_data = self.data_dir / "version_1_clean.csv"

        # Rimuovo eventuale file precedente
        if self.clean_data.exists():
            self.clean_data.unlink()

    def test_main_runs_end_to_end(self):
        """
        Simulo l'esecuzione da terminale:
        python main.py --no-interactive --mode holdout

        Verifico che:
        - Il programma venga eseguito
        - Il dataset pulito venga generato
        """

        # Simulazione argomenti CLI
        sys.argv = [
            "main.py",
            "--no-interactive",
            "--mode", "holdout"
        ]

        # Eseguo main()
        main()

        # Verifico effetto concreto sul filesystem
        self.assertTrue(self.clean_data.exists())


if __name__ == "__main__":
    unittest.main()
