import pandas as pd

def load_raw_dataset(path: str):
    """
    Carica il dataset grezzo dal file CSV.
    Non applica alcuna trasformazione.
    """
    df = pd.read_csv(path)
    return df

