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
    # Rimpiazza valori tipo "?" o stringhe anomale
    df = df.replace("?", pd.NA)

    # Converte tutto a numerico (ciò che non si converte diventa NaN)
    df = df.apply(pd.to_numeric, errors='coerce')

    # Sostituisce i valori mancanti con la mediana della colonna
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
    In particolare, rimuove gli ID che non devono essere usati come feature.
    """
    if "Sample code number" in df.columns:
        df = df.drop(columns=["Sample code number"])
    return df

def normalize_features(df: pd.DataFrame):
    """
    Normalizza tutte le feature numeriche usando min-max scaling.
    Non applicata sulla colonna target.
    """
    # Per sicurezza lavoriamo su una copia
    df = df.copy()
    
    # Min-max scaling: (x - min) / (max - min)
    df = (df - df.min()) / (df.max() - df.min())
    
    return df

def split_features_target(df: pd.DataFrame):
    """
    Separa le feature X dal target y.
    Mappa la classe da {2,4} a {0,1}.
    """
    df = df.copy()

    # Mappatura della classe
    df["classtype_v1"] = df["classtype_v1"].map({2.0: 0, 4.0: 1})

    # y = target
    y = df["classtype_v1"].values

    # X = tutto tranne il target
    X = df.drop(columns=["classtype_v1"]).values

    return X, y
