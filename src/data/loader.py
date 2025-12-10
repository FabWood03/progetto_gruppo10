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


def rename_columns(df: pd.DataFrame):
    """
    Rinomina colonne con nomi errati o non standard.
    """
    df = df.rename(columns={
        "uniformity_cellsize_xx": "Uniformity of Cell Size",
        "clump_thickness_ty": "Clump Thickness",
        "bareNucleix_wrong": "Bare Nuclei"
    })
    return df


def remove_unwanted_columns(df: pd.DataFrame):
    """
    Rimuove colonne non utili o dannose per il modello.
    """
    if "Sample code number" in df.columns:
        df = df.drop(columns=["Sample code number"])
    return df


def normalize_features(df: pd.DataFrame):
    """
    Normalizza tutte le feature numeriche usando min-max scaling.
    Non applicata sulla colonna target.
    """
    df = df.copy()
    df = (df - df.min()) / (df.max() - df.min())
    return df


def split_features_target(
    df: pd.DataFrame,
    target_column: str = "classtype_v1",
    positive_label: float = 4.0,
    negative_label: float = 2.0
):
    """
    Separa X e y in modo robusto:
    - verifica che il target esista
    - mappa {negative_label, positive_label} -> {0,1}
    """
    
    if target_column not in df.columns:
        raise ValueError(
            f"Colonna target '{target_column}' non trovata. Colonne disponibili: {list(df.columns)}"
        )

    df = df.copy()

    df[target_column] = df[target_column].map({
        negative_label: 0,
        positive_label: 1
    })

    y = df[target_column].values
    X = df.drop(columns=[target_column]).values

    return X, y


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
        df = df.rename(columns=self.columns_to_rename)

        # 4. Rimozione colonne inutili
        for col in self.columns_to_remove:
            if col in df.columns:
                df = df.drop(columns=[col])

        # 5. Normalizzazione
        df = normalize_features(df)

        # 6. Separazione X/y
        X, y = split_features_target(
            df,
            target_column=self.target_column,
            positive_label=self.positive_label,
            negative_label=self.negative_label
        )

        return X, y



