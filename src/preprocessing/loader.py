import pandas as pd


def load_raw_dataset(path: str):
    """
    Carica il dataset grezzo dal file CSV.
    Non applica alcuna trasformazione.
    """
    df = pd.read_csv(path)
    return df


def clean_dataset(df: pd.DataFrame, exclude_columns=None):
    exclude_columns = exclude_columns or []

    df = df.replace("?", pd.NA)
    df = df.apply(pd.to_numeric, errors="coerce")

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
            "Sample code number",
            "Blood Pressure",
            "Heart Rate"
        ]


    def load(self):

        """
    Esegue la pipeline di preprocessing del dataset e prepara i dati
    per la fase di classificazione.

    :return:
        - X: matrice delle feature preprocessate
        - y: vettore delle etichette binarie
        - df: dataframe preprocessato completo
        """
    
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


        # 7. Split X/y
        X = df.drop(columns=[self.target_column]).values
        y = df[self.target_column].values

        return X, y, df