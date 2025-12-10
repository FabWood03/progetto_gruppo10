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
