import pandas as pd

def load_raw_dataset(path: str):
    """
    Carica il dataset grezzo dal file CSV.
    Non applica alcuna trasformazione.
    """
    df = pd.read_csv(path)
    return df


def clean_dataset(df: pd.DataFrame):
    """
    Pulisce il dataset:
    - sostituisce valori non numerici con NaN
    - converte tutto in float
    - gestisce valori mancanti con la mediana
    """
    df = df.replace("?", pd.NA)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(df.median())
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

    def __init__(
        self,
        path: str,
        target_column: str = "classtype_v1",
        positive_label: float = 4.0,
        negative_label: float = 2.0,
        columns_to_rename: dict = None,
        columns_to_remove: list = None
    ):
        self.path = path
        self.target_column = target_column
        self.positive_label = positive_label
        self.negative_label = negative_label

        self.columns_to_rename = columns_to_rename or {
            "uniformity_cellsize_xx": "Uniformity of Cell Size",
            "clump_thickness_ty": "Clump Thickness",
            "bareNucleix_wrong": "Bare Nuclei"
        }

        self.columns_to_remove = columns_to_remove or [
            "Sample code number"
        ]

    def load(self):
        # 1. Caricamento dataset
        df = load_raw_dataset(self.path)

        # 2. Cleaning
        df = clean_dataset(df)

        # 3. Rinominare colonne
        df = rename_columns(df, self.columns_to_rename)


        # 4. Rimozione colonne inutili
        df = remove_unwanted_columns(df, self.columns_to_remove)




        # 5. Conversione del target (prima della normalizzazione)
        df[self.target_column] = df[self.target_column].map({
            self.negative_label: 0,
            self.positive_label: 1
        })

        # Controllo errori nei valori target
        if df[self.target_column].isna().any():
            valori_originali = set(df[self.target_column])
            raise ValueError(f"Valori target non validi trovati: {valori_originali}")

        # 6. Normalizzazione SOLO delle feature (non del target)
        df = normalize_features(df, exclude_columns=[self.target_column])

        # 7. Split X/y
        X = df.drop(columns=[self.target_column]).values
        y = df[self.target_column].values

        return X, y


