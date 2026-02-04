# KNN Classifier – Project Guide

**Progetto Gruppo 10**

Questo progetto implementa una pipeline completa per la **classificazione binaria** basata su **K-Nearest Neighbors (KNN)**, con particolare attenzione al **preprocessing dei dati**, alla **gestione delle distanze** e alla **valutazione corretta delle prestazioni**.

## Contenuto del progetto

Questa documentazione descrive in dettaglio:

- preprocessing del dataset
- implementazione del classificatore KNN
- metriche di distanza
- strategie di validazione
- risultati sperimentali
- modalità di esecuzione (Docker e argparse)

## Installazione ed esecuzione tramite Docker

### Requisiti di Sistema

- **Docker** (versione ≥ 20.x)

Non è richiesta l’installazione locale di Python o librerie aggiuntive.

### Clonare il repository

Aprire un terminale ed eseguire:

```bash
git clone https://github.com/FabWood03/progetto_gruppo10.git
cd progetto_gruppo10
```

> Questo passaggio è necessario **solo se il progetto non è già presente in locale**.
> 
> 
> Se la cartella `progetto_gruppo10/` è già disponibile sul proprio computer, è possibile passare direttamente alla fase di build dell’immagine Docker.
> 

### Spostarsi nella cartella del progetto

È **fondamentale** posizionarsi nella directory root del progetto, ovvero nella cartella che contiene il file `Dockerfile`.

Esempio:

```bash
cd percorso/della/cartella/progetto_gruppo10
```

Per verificare di essere nella cartella corretta, eseguire:

```bash
ls
```

### Costruire l’immagine Docker

Dalla directory root del progetto, eseguire:

```bash
docker build -t progetto_gruppo10 .
```

Questo comando crea l’immagine Docker necessaria all’esecuzione del progetto.

### Installazione di Docker (se non presente)

Nel caso Docker non sia installato sul proprio sistema, è possibile scaricarlo e installarlo dal sito ufficiale:

🔗 https://www.docker.com/get-started/

Dopo l’installazione, assicurarsi che Docker sia correttamente avviato prima di procedere con la build dell’immagine.

È possibile verificare che Docker sia in esecuzione aprendo un terminale e digitando:

```bash
docker --version
```

oppure:

```bash
docker info
```

Se Docker è correttamente avviato, i comandi restituiranno le informazioni sulla versione e sullo stato del sistema Docker. In caso contrario, verrà mostrato un messaggio di errore.

### Avviare il container

Il container deve essere avviato tramite **terminale**, in quanto l’esecuzione richiede il montaggio esplicito delle cartelle `data/` e `outputs/` come volumi esterni.

Il montaggio dei volumi è necessario per rendere disponibili i dataset di input e per garantire la persistenza dei risultati generati dal programma.

**Sempre dalla stessa directory (`progetto_gruppo10`)**, eseguire:

```bash
docker run --rm -it -v"$(pwd)/data:/app/data" -v"$(pwd)/outputs:/app/outputs" progetto_gruppo10
```

L’esecuzione del programma produce **risultati sperimentali e metriche di valutazione**, salvati automaticamente nella cartella `outputs/` sul filesystem locale.

Tutti i file di output vengono generati automaticamente a runtime e **non sono inclusi nell’immagine Docker**, garantendo la persistenza dei risultati tramite volume Docker.

## Esecuzione con argparse

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
## Struttura del Progetto

## Dataset

Il dataset utilizzato presenta le seguenti caratteristiche:

- feature numeriche e ordinali (valori discreti, tipicamente 1–10)
- una colonna target per classificazione binaria
- valori mancanti indicati con `"?"`
- colonne non informative o con nomi non standard

Dopo il preprocessing, il dataset finale è composto da **615 campioni**.

## Preprocessing dei dati

Il preprocessing è gestito dalla classe `DataLoader` (`src/preprocessing/loader.py`) ed è strutturato come una pipeline sequenziale e configurabile.

### Pipeline di preprocessing

1. **Caricamento del dataset grezzo**
    - Lettura diretta del file CSV senza trasformazioni iniziali.
2. **Pulizia del target**
    - Rimozione delle righe con target mancante.
    - Rimappatura del target in:
        - `0` → classe negativa
        - `1` → classe positiva
    - Blocco dell’esecuzione in presenza di valori non mappabili.
3. **Rinomina delle colonne**
    - Correzione di nomi errati o non standard tramite mapping esplicito.
4. **Rimozione delle colonne inutili**
    - Eliminazione di feature non rilevanti per il task di classificazione.
5. **Pulizia delle feature**
    - Conversione di `"?"` in `NaN`.
    - Conversione forzata delle feature in formato numerico.

Output del preprocessing:

- `X`: matrice delle feature
- `y`: vettore delle etichette
- `df_clean`: dataframe pulito (salvabile su file)

## Gestione del Data Leakage

Poiché KNN è un algoritmo **basato sulle distanze**, una gestione scorretta del preprocessing può introdurre facilmente **data leakage**.

Per evitarlo:

### Imputazione dei valori mancanti

- I valori `NaN` vengono imputati usando la **mediana calcolata esclusivamente sul training set**.
- La stessa mediana viene applicata al test set.
- L’operazione viene ripetuta indipendentemente per:
    - Holdout
    - ogni fold del K-Fold
    - ogni iterazione del Leave-P-Out

### Normalizzazione Min-Max

- Le feature sono normalizzate con **Min-Max scaling**.
- Min e Max sono calcolati **solo sul training set**.
- Il test set viene trasformato usando gli stessi parametri.

## Classificatore KNN

Il classificatore è implementato manualmente nella classe `KNNClassifier` (`src/knn/classifier.py`).

Caratteristiche principali:

- implementazione completa senza librerie ML esterne
- memorizzazione del training set (no training parametrico)
- calcolo delle distanze vettorializzato (1 vs N)
- supporto a predizioni singole e batch
- gestione dei pareggi tramite random tie-breaking con seed
- supporto alla predizione delle probabilità di classe

## Metriche di distanza

Le metriche di distanza sono implementate utilizzando il **pattern Strategy** (`src/knn/distances.py`).

Metriche disponibili:

- **Euclidean Distance (L2)**
- **Manhattan Distance (L1)** – adatta a feature ordinali
- **Chebyshev Distance (L∞)** – basata sulla massima differenza tra feature

La selezione della metrica avviene tramite una `DistanceFactory`, che centralizza la logica ed evita duplicazioni di codice.

## Strategie di validazione

La pipeline supporta tre strategie di validazione, tutte compatibili con l’assenza di data leakage:
1. **Holdout Validation**: effettua un partizionamento statico del dataset (default 80/20) per una valutazione rapida del modello.
2. **K-Fold Cross Validation**: suddivide il dataset in k fold per una validazione iterativa. Ideale per massimizzare l'uso dei dati disponibili e ottenere una valutazione statistica oggettiva delle metriche di classificazione.
3. **Leave-P-Out**: algoritmo di validazione che testa il modello su tutte le possibili combinazioni di p campioni.

## Risultati sperimentali

I risultati riportati sono ottenuti con:

- `K = 5`
- preprocessing completo
- dataset di 615 campioni

### Manhattan Distance (K = 5)

**Holdout**

- Accuracy: 0.9675
- F1-score: 0.9556
- G-Mean: 0.9649
- AUC: 0.9946

**K-Fold (5 fold)**

- Accuracy: 0.9659 ± 0.0166
- F1-score: 0.9479 ± 0.0289
- G-Mean: 0.9631 ± 0.0208
- AUC: 0.9828 ± 0.0156

### Chebyshev Distance (K = 5)

**Holdout**

- Accuracy: 0.9756
- F1-score: 0.9670
- G-Mean: 0.9761
- AUC: 0.9973

**K-Fold (5 fold)**

- Accuracy: 0.9528 ± 0.0174
- F1-score: 0.9286 ± 0.0306
- G-Mean: 0.9485 ± 0.0210
- AUC: 0.9763 ± 0.0137

## Valutazione delle prestazioni

Le prestazioni del classificatore KNN sono state valutate utilizzando un insieme di **metriche specifiche per la classificazione binaria**, al fine di ottenere una valutazione completa e bilanciata del modello.

Le metriche considerate sono le seguenti:

- **Accuracy**
    
    Percentuale di campioni correttamente classificati rispetto al totale.
    
- **Error**
    
    Percentuale di campioni classificati in modo errato, calcolata come complemento dell’accuracy.
    
- **Precision**
    
    Proporzione di predizioni positive corrette rispetto al totale delle predizioni positive.
    
- **Sensitivity (Recall)**
    
    Capacità del modello di individuare correttamente i campioni appartenenti alla classe positiva.
    
- **Specificity**
    
    Capacità del modello di individuare correttamente i campioni appartenenti alla classe negativa.
    
- **F1-score**
    
    Media armonica tra precision e sensitivity, utile in presenza di classi sbilanciate.
    
- **G-Mean**
    
    Media geometrica tra sensitivity e specificity, indicata per valutare le prestazioni complessive su entrambe le classi.
    

L’utilizzo combinato di queste metriche consente di analizzare in modo approfondito il comportamento del classificatore, evitando valutazioni basate esclusivamente sull’accuracy.

## Considerazioni Finali
Il progetto ha mostrato come l’efficacia del classificatore KNN dipenda fortemente da un preprocessing corretto, dalla scelta appropriata della metrica di distanza e dall’utilizzo di strategie di validazione adeguate. In particolare, l’analisi delle metriche evidenzia che l’accuracy da sola non è sufficiente a descrivere le prestazioni del modello, rendendo necessario l’impiego di misure come F1-score, G-Mean e AUC per una valutazione più affidabile e completa.