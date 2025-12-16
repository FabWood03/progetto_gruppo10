import pandas as pd


def load_raw_dataset(path: str):
    """
    Carica il dataset grezzo dal file CSV.
    Non applica alcuna trasformazione.
    """
    df = pd.read_csv(path)
    return df


def clean_dataset(df: pd.DataFrame, exclude_columns=None):
    """
    Pulisce il dataset:
    - sostituisce '?' con NaN
    - converte a numerico
    - imputa SOLO le feature (esclude il target)
    """
    exclude_columns = exclude_columns or []

    df = df.replace("?", pd.NA)
    df = df.apply(pd.to_numeric, errors="coerce")

    feature_cols = [c for c in df.columns if c not in exclude_columns]
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    return df



def rename_columns(df: pd.DataFrame, mapping: dict):
    """
    Rinomina colonne con nomi errati o non standard.
    """
    return df.rename(columns=mapping)


def remove_unwanted_columns(df: pd.DataFrame, columns_to_remove: list):
    """
    Rimuove le colonne specificate se presenti.
    """
    cols = [c for c in columns_to_remove if c in df.columns]
    return df.drop(columns=cols)


def normalize_features(df: pd.DataFrame, exclude_columns=None):
    df = df.copy()
    exclude_columns = exclude_columns or []

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in exclude_columns]

    for col in numeric_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if min_val != max_val:
            df[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df[col] = 0.0

    return df


class DataLoader:
    """
    Classe che esegue l'intera pipeline di preprocessing del dataset.
    Tutti i parametri sono configurabili in modo da rendere il codice robusto.
    """
    #(nuovo)
    def __init__(
        self,
        path: str,
        target_column: str = "classtype_v1",
        positive_label: float = 4.0,
        negative_label: float = 2.0,
        columns_to_rename: dict = None,
        columns_to_remove: list = None,
        normalize: bool = True
):

        self.path = path
        self.target_column = target_column
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.normalize = normalize 


        self.columns_to_rename = columns_to_rename or {
            "uniformity_cellsize_xx": "Uniformity of Cell Size",
            "clump_thickness_ty": "Clump Thickness",
            "bareNucleix_wrong": "Bare Nuclei"
        }

        self.columns_to_remove = columns_to_remove or [
            "Sample code number",
            "Blood Pressure",
            "Heart Rate"
        ]


    def load(self):
        # 1. Caricamento dataset
        df = load_raw_dataset(self.path)

        # 2. Elimina righe con target mancante
        df = df.dropna(subset=[self.target_column])

        # 3. Rinominare colonne
        df = rename_columns(df, self.columns_to_rename)

        # 4. Rimozione colonne inutili
        df = remove_unwanted_columns(df, self.columns_to_remove)

        # 5. Cleaning SOLO delle feature (target escluso)
        df = clean_dataset(df, exclude_columns=[self.target_column])

        #(nuovo)
        raw_target = df[self.target_column].copy()

        df[self.target_column] = df[self.target_column].map({
            self.negative_label: 0,
            self.positive_label: 1
        })

        if df[self.target_column].isna().any():
            bad_values = set(raw_target[df[self.target_column].isna()].unique())
            raise ValueError(f"Valori target non mappabili trovati: {bad_values}")


        # 6. Normalizzazione SOLO se richiesta (evita leakage)
        if self.normalize:
            df = normalize_features(df, exclude_columns=[self.target_column])

        # 7. Split X/y
        X = df.drop(columns=[self.target_column]).values
        y = df[self.target_column].values

        return X, y, df