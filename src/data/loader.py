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
