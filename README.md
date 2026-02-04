# Progetto gruppo 10

## Run con docker

## Run con argparse

Il programma può essere eseguito sia in **modalità interattiva**, sia in **modalità batch** utilizzando gli argomenti da linea di comando tramite `argparse`.  

### Sintassi generale

```bash
python src/main.py [--k K] [--metric METRIC] [--mode MODE] [--test-split SPLIT] [--n-splits N] [--p P] [--no-interactive]
```

### Parametri

| Argomento          | Tipo  | Descrizione                                                                 | Default (da config.ini) |
|:-------------------|:-----:|-----------------------------------------------------------------------------|-------------------------|
| `--k`              |  int  | Numero di vicini per KNN                                                    | 5                       |
| `--metric`         |  str  | Metrica di distanza: `euclidean`, `manhattan`, `chebyshev`                  | euclidean               |
| `--mode`           |  str  | Modalità di validazione: `holdout`, `kfold`, `leavepout`, `all`             | all                     |
| `--test-split`     | float | Percentuale del test set (solo holdout)                                     | 0.2                     |
| `--n-splits`       |  int  | Numero di fold (solo k-fold)                                                | 5                       |
| `--p`              |  int  | Valore P (solo leave-p-out)                                                 | 1                       |
| `--no-interactive` | flag  | Disabilita input interattivo: prende tutti i valori dai default o dagli arg | –                       |

### Esempi d'uso
1. **Esecuzione standard:** verranno richiesti i parametri mancanti tramite prompt.

```bash
python src/main.py --mode holdout
```

2. **Override dei default:** esegue una validazione K-Fold con K=5 e metrica Manhattan, senza chiedere input.

```bash
python src/main.py --mode kfold --k 5 --metric manhattan --n-splits 10 --no-interactive
```

3. **Esecuzione batch:** utilizza esclusivamente le impostazioni salvate in config.ini.

```bash
python src/main.py --no-interactive
```