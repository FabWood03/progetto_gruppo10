from collections import defaultdict
import numpy as np

from src.metrics import evaluate_metrics, confusion_counts
from src.validation.base import ValidationStrategy
from src.validation.base import minmax_scale_train_test
from src.validation.base import median_impute_train_test




class KFoldValidation(ValidationStrategy):
    """
    Implementazione della strategia K-Fold Cross-Validation ottimizzata.
    Divide il dataset in k fold, addestra il modello su k-1 fold e
    valuta le prestazioni sul fold rimanente, ripetendo per tutti i fold.
    1. Addestra il modello memorizzando i dati di training.
    2. Durante l'inference, calcola le distanze in modo vettorializzato
       per tutti i campioni di test in un singolo passaggio.
    3. Accumula le metriche per ogni fold e calcola la media e deviazione standard
       alla fine.
    """

    def __init__(self, n_splits: int = 5, random_state: int | None = None):
        if n_splits < 2:
            raise ValueError("n_splits deve essere almeno 2.")
        self.n_splits = n_splits
        self.rng = np.random.default_rng(random_state)

    def validate(self, model, X, y):
        """
        Esegue la validazione K-Fold ottimizzata.

        Per ogni fold, il dataset viene suddiviso in training e test set.
        La normalizzazione delle feature è applicata separatamente
        per ciascun fold, utilizzando esclusivamente i dati di training,
        al fine di evitare il data leakage.

        :param model: Modello KNN da validare.
        :param X: Matrice delle feature.
        :param y: Vettore delle etichette.
        :return: Dizionario con metriche aggregate e dettagliate.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples = X.shape[0]

        # Creazione e mescolamento indici
        indices = np.arange(n_samples)
        self.rng.shuffle(indices)

        # Divisione in k fold
        folds = np.array_split(indices, self.n_splits)

        metrics_history = defaultdict(list)
        aggregated_cm = np.zeros((2, 2), dtype=np.int64)

        for i in range(self.n_splits):
            # 1. Definizione Training e Test Set per questo fold
            test_idx = folds[i]

            # Uniamo tutti gli altri fold per il training
            train_idx = np.concatenate([folds[j] for j in range(self.n_splits) if j != i])

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Imputazione dei NaN (solo sul training del fold)
            X_train, X_test = median_impute_train_test(X_train, X_test)
            
            # Normalizzazione corretta (solo sul training del fold)
            X_train, X_test = minmax_scale_train_test(X_train, X_test)

            # 2. Addestramento (Memorizzazione)
            model.fit(X_train, y_train)

            # 3. Inference Ottimizzata (Single Pass Distance Calculation)
            y_pred_fold = []
            y_score_fold = []

            # Iteriamo sui campioni di test del fold corrente
            for sample in X_test:
                # Calcolo distanze vettorializzato
                dists = model.distance_metric.calculate(sample, X_train)

                # Predizione Classe
                pred_label = model.predict_with_distances(dists)
                y_pred_fold.append(pred_label)

                # Predizione Probabilità
                probas = model.predict_proba_with_distances(dists)

                # Assumiamo classificazione binaria: prendiamo probabilità classe 1
                # Se probas ha length 2 (classe 0 e 1), prendiamo index 1.
                # Se c'è solo una classe nel train fold, gestiamo l'edge case.
                if len(probas) > 1:
                    y_score_fold.append(probas[1])
                else:
                    # Se il modello ha visto solo una classe (es. solo 0), probas ha len 1
                    # Se la classe unica è 1, la prob è probas[0], altrimenti 0
                    unique_cls = np.unique(y_train)
                    if unique_cls[0] == 1:
                        y_score_fold.append(probas[0])
                    else:
                        y_score_fold.append(0.0)

            y_pred_fold = np.array(y_pred_fold)
            y_score_fold = np.array(y_score_fold)

            # 4. Valutazione Metriche
            fold_metrics = evaluate_metrics(
                y_true=y_test,
                y_pred=y_pred_fold,
                y_score=y_score_fold,
                metrics=["accuracy", "error", "sensitivity", "specificity",
                         "precision", "f1", "gmean", "auc"]
            )

            # Accumulo Metriche Scalari
            for name, value in fold_metrics.items():
                metrics_history[name].append(value)

            # Accumulo Matrice di Confusione
            c = confusion_counts(y_test, y_pred_fold)
            cm_fold = np.array([[c.tp, c.fn], [c.fp, c.tn]])
            aggregated_cm += cm_fold

        # 5. Calcolo Statistiche Finali
        summary = {}
        for name, values in metrics_history.items():
            summary[f"{name}_mean"] = float(np.mean(values))
            summary[f"{name}_std"] = float(np.std(values))

        return {
            "summary": summary,
            "raw_metrics": dict(metrics_history),
            "aggregated_cm": aggregated_cm,
            "folds_indices": folds
        }
