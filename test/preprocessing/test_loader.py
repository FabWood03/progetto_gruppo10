import unittest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from src.preprocessing.loader import (
    load_raw_dataset,
    clean_dataset,
    rename_columns,
    remove_unwanted_columns,
    DataLoader
)


class TestDataFunctions(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "A": ["1", "2", "?"],
            "B": ["3", "?", "5"],
            "target": [2.0, 4.0, 2.0]
        })

    # =========================
    # load_raw_dataset
    # =========================
    def test_load_raw_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.csv"
            self.df.to_csv(path, index=False)

            df_loaded = load_raw_dataset(str(path))
            self.assertIsInstance(df_loaded, pd.DataFrame)
            self.assertEqual(len(df_loaded), 3)

    # =========================
    # clean_dataset
    # =========================
    def test_clean_dataset_converts_to_numeric(self):
        cleaned = clean_dataset(self.df.copy())
        self.assertTrue(pd.isna(cleaned.loc[2, "A"]))
        self.assertTrue(pd.isna(cleaned.loc[1, "B"]))

    # =========================
    # rename_columns
    # =========================
    def test_rename_columns(self):
        mapping = {"A": "A_new"}
        renamed = rename_columns(self.df, mapping)
        self.assertIn("A_new", renamed.columns)
        self.assertNotIn("A", renamed.columns)

    # =========================
    # remove_unwanted_columns
    # =========================
    def test_remove_unwanted_columns(self):
        reduced = remove_unwanted_columns(self.df, ["B", "C"])
        self.assertNotIn("B", reduced.columns)
        self.assertIn("A", reduced.columns)

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "uniformity_cellsize_xx": ["1", "2", "3"],
            "clump_thickness_ty": ["4", "5", "?"],
            "bareNucleix_wrong": ["7", "8", "9"],
            "Sample code number": [100, 101, 102],
            "classtype_v1": [2.0, 4.0, 2.0]
        })

    def _write_csv(self):
        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name) / "data.csv"
        self.df.to_csv(path, index=False)
        return tmp, path

    # =========================
    # Pipeline completa
    # =========================
    def test_dataloader_pipeline_success(self):
        tmp, path = self._write_csv()

        loader = DataLoader(path=str(path))
        X, y, df_out = loader.load()

        # Target binario
        self.assertTrue(set(y).issubset({0, 1}))

        # Colonne rimosse
        self.assertNotIn("Sample code number", df_out.columns)

        # Colonne rinominate
        self.assertIn("Uniformity of Cell Size", df_out.columns)
        self.assertIn("Clump Thickness", df_out.columns)

        # Shape coerente
        self.assertEqual(X.shape[0], y.shape[0])

        tmp.cleanup()

    # =========================
    # Target non mappabile
    # =========================
    def test_dataloader_invalid_target_raises(self):
        self.df.loc[1, "classtype_v1"] = 3.0
        tmp, path = self._write_csv()

        loader = DataLoader(path=str(path))

        with self.assertRaises(ValueError):
            loader.load()

        tmp.cleanup()


if __name__ == "__main__":
    unittest.main()