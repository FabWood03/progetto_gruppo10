import sys
import gc
import traceback
import configparser
from pathlib import Path
from time import time
import numpy as np

# --- Setup Path ---
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.preprocessing.loader import DataLoader
from src.knn.classifier import KNNClassifier
from src.validation.holdout import HoldoutValidation
from src.validation.k_fold import KFoldValidation
from src.validation.leave_p_out import LeavePOutValidation
from src.metrics.plotter import plot_confusion_matrix, plot_roc_curve, plot_metric_distribution
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="KNN Classifier Pipeline"
    )

    # --- Model ---
    parser.add_argument("--k", type=int, help="Numero di vicini")
    parser.add_argument(
        "--metric",
        choices=["euclidean", "manhattan", "chebyshev"],
        help="Metrica di distanza"
    )

    # --- Validazione ---
    parser.add_argument(
        "--mode",
        choices=["holdout", "kfold", "leavepout", "all"],
        help="Modalità di validazione"
    )

    parser.add_argument("--test-split", type=float, help="Percentuale test (holdout)")
    parser.add_argument("--n-splits", type=int, help="Numero fold (k-fold)")
    parser.add_argument("--p", type=int, help="Valore P (leave-p-out)")

    # --- Flags ---
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Disabilita input interattivo (batch mode)"
    )

    return parser.parse_args()


# --- Global Config Loader ---
def load_config():
    config = configparser.ConfigParser()
    config_path = BASE_DIR / 'config.ini'
    if not config_path.exists():
        raise FileNotFoundError(f"File di configurazione non trovato in {config_path}")
    config.read(config_path)
    return config


# --- Gestore Percorsi Dinamico ---
class Paths:
    """I percorsi vengono inizializzati leggendo il file .ini"""

    def __init__(self, config):
        self.DATA_DIR = BASE_DIR / "data"
        self.OUTPUT_DIR = BASE_DIR / config['PATHS']['output_dir']
        self.INPUT_FILE = BASE_DIR / config['PATHS']['input_file']
        self.OUTPUT_CLEAN_FILE = BASE_DIR / config['PATHS']['output_clean']


# --- Gestore Input Utente ---
class InputHandler:
    @staticmethod
    def get_int(prompt, min_val=None, max_val=None, default=None):
        while True:
            d_str = f" (default={default})" if default is not None else ""
            user_input = input(f"{prompt}{d_str}: ").strip()
            if not user_input and default is not None:
                return int(default)
            try:
                val = int(user_input)
                if (min_val is not None and val < min_val) or (max_val is not None and val > max_val):
                    print(f" > Errore: Valore fuori range ({min_val}-{max_val}).")
                    continue
                return val
            except ValueError:
                print(" > Errore: Inserisci un numero intero.")

    @staticmethod
    def get_float(prompt, min_val=0.0, max_val=1.0, default=None):
        while True:
            d_str = f" (default={default})" if default is not None else ""
            user_input = input(f"{prompt}{d_str}: ").strip()
            if not user_input and default is not None:
                return float(default)
            try:
                val = float(user_input)
                if val <= min_val or val >= max_val:
                    print(f" > Errore: Valore fuori range ({min_val}-{max_val}).")
                    continue
                return val
            except ValueError:
                print(" > Errore: Inserisci un numero decimale.")

    @staticmethod
    def get_choice(prompt, options, default=None):
        options_str = "/".join(options)
        d_str = f" (default={default})" if default else ""
        while True:
            user_input = input(f"{prompt} [{options_str}]{d_str}: ").strip().lower()
            if not user_input and default:
                return default
            if user_input in options:
                return user_input
            print(f" > Errore: Scelta non valida.")


# --- Funzioni di Utility ---
def print_separator(title=""):
    print(f"\n{title:=^60}")


def print_metrics(metrics: dict, duration: float, mode="scalar"):
    print(f"\n{'-' * 40}")
    print(f"{'METRICA':<25} | {'VALORE':<15}")
    print(f"{'-' * 40}")
    if mode == "scalar":
        for k, v in metrics.items():
            print(f"{k.replace('_', ' ').title():<25} | {v:.4f}")
    else:
        metric_names = sorted(list(set(k.replace("_mean", "").replace("_std", "") for k in metrics.keys())))
        for m in metric_names:
            mean = metrics.get(f"{m}_mean", 0.0)
            std = metrics.get(f"{m}_std", 0.0)
            print(f"{m.title():<25} | {mean:.4f} (+/- {std:.4f})")
    print(f"{'-' * 40}")
    print(f"{'Tempo Esecuzione':<25} | {duration:.4f} s")
    print(f"{'-' * 40}\n")


def save_plots(results: dict, mode: str, k: int, paths_obj):
    out_dir = paths_obj.OUTPUT_DIR / f"{mode}_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = ["Benigno (0)", "Maligno (1)"]

    # Confusion matrix
    cm_key = "confusion_matrix" if mode == "holdout" else "aggregated_cm"
    if cm_key in results:
        plot_confusion_matrix(
            results[cm_key],
            labels,
            f"CM {mode.title()} (K={k})",
            str(out_dir / "cm_norm.png"),
            normalize=True
        )

    # ROC solo per holdout
    if mode == "holdout" and "roc_data" in results:
        fpr, tpr = results["roc_data"]
        plot_roc_curve(fpr, tpr, results["metrics"].get("auc", 0.0),
                       f"ROC {mode.title()}", str(out_dir / "roc.png"))

    # Distribuzione metriche per K-Fold / LPO
    if mode in ["kfold", "leavepout"] and "raw_metrics" in results:
        for metric_name, values in results["raw_metrics"].items():
            plot_metric_distribution(
                values,
                metric_name,
                f"Distribuzione {metric_name.title()} ({mode.title()}, K={k})",
                str(out_dir / f"{metric_name}_dist.png")
            )

    print(f" [INFO] Grafici salvati in: {out_dir}")


# --- Runner ---
def execute_validation(mode, X, y, model_params, val_params, paths_obj):
    print_separator(f" AVVIO {mode.upper()} ")
    model = KNNClassifier(k=model_params['k'], distance=model_params['metric'], random_state=42)

    if mode == "holdout":
        validator = HoldoutValidation(test_size=val_params['split'], random_state=42)
    elif mode == "kfold":
        validator = KFoldValidation(n_splits=val_params['splits'], random_state=42)
    elif mode == "leavepout":
        validator = LeavePOutValidation(p=val_params['p'])
    else:
        return

    try:
        gc.collect()
        start_time = time()
        results = validator.validate(model, X, y)
        duration = time() - start_time
        metrics_key = "metrics" if mode == "holdout" else "summary"
        print_metrics(results[metrics_key], duration, mode="scalar" if mode == "holdout" else "aggregate")
        save_plots(results, mode, model_params['k'], paths_obj)
    except Exception:
        print(f" [!] Errore durante {mode}:")
        traceback.print_exc()


# --- MAIN ---
def main():
    args = parse_args()

    try:
        # 0. Caricamento configurazione .ini
        config = load_config()
        paths = Paths(config)

        if not paths.INPUT_FILE.exists():
            print(f"ERRORE: File non trovato: {paths.INPUT_FILE}")
            return

        print_separator(" KNN CLASSIFIER PIPELINE (.INI CONFIG LOADED) ")

        # 1. Caricamento dati
        print(" [1/4] Caricamento Dataset...")
        loader = DataLoader(str(paths.INPUT_FILE))
        X, y, df_clean = loader.load()
        X = np.ascontiguousarray(X)
        df_clean.to_csv(paths.OUTPUT_CLEAN_FILE, index=False)
        print(f"   - Dataset: {X.shape[0]} samples. Clean salvato in {paths.OUTPUT_CLEAN_FILE}")

        # 2. Configurazione con default da .ini
        print("\n [2/4] Configurazione Parametri")
        model_params = {
            "k": (
                args.k if args.k is not None else
                InputHandler.get_int(
                    "   - Numero vicini (K)",
                    1, X.shape[0],
                    config['MODEL_DEFAULTS']['k']
                ) if not args.no_interactive else int(config['MODEL_DEFAULTS']['k'])
            ),
            "metric": (
                args.metric if args.metric is not None else
                InputHandler.get_choice(
                    "   - Metrica distanza",
                    ["euclidean", "manhattan", "chebyshev"],
                    config['MODEL_DEFAULTS']['metric']
                ) if not args.no_interactive else config['MODEL_DEFAULTS']['metric']
            )
        }

        print(f" [INFO] Parametri modello: K={model_params['k']}, Distance={model_params['metric']}")

        val_mode = (
            args.mode if args.mode is not None else
            InputHandler.get_choice(
                "\n   - Modalità validazione",
                ["holdout", "kfold", "leavepout", "all"],
                "all"
            )
        )

        val_params = {
            "split": args.test_split if args.test_split is not None
            else config.getfloat('VALIDATION_DEFAULTS', 'test_split'),

            "splits": args.n_splits if args.n_splits is not None
            else config.getint('VALIDATION_DEFAULTS', 'n_splits'),

            "p": args.p if args.p is not None
            else config.getint('VALIDATION_DEFAULTS', 'lpo_p')
        }

        should_run_lpo = True

        # --- HOLDOUT ---
        if val_mode in ["holdout", "all"]:
            if args.test_split is not None:
                val_params["split"] = args.test_split
            elif not args.no_interactive:
                val_params["split"] = InputHandler.get_float(
                    "   - [Holdout] % Test Set",
                    0.0, 1.0,
                    val_params["split"]
                )

        # --- K-FOLD ---
        if val_mode in ["kfold", "all"]:
            if args.n_splits is not None:
                val_params["splits"] = args.n_splits
            elif not args.no_interactive:
                val_params["splits"] = InputHandler.get_int(
                    "   - [K-Fold] Numero Fold",
                    2, 50,
                    val_params["splits"]
                )

        # --- LEAVE-P-OUT ---
        if val_mode in ["leavepout", "all"]:
            if args.p is not None:
                val_params["p"] = args.p
            elif not args.no_interactive:
                val_params["p"] = InputHandler.get_int(
                    "   - [Leave-P-Out] Valore P",
                    1, 10,
                    val_params["p"]
                )

            # controllo costo computazionale
            if val_params["p"] > 2 and X.shape[0] > 100:
                if not args.no_interactive:
                    if InputHandler.get_choice(
                            "     ATTENZIONE: P pesante. Procedere?",
                            ["s", "n"],
                            "n"
                    ) == "n":
                        should_run_lpo = False
                else:
                    print(" [WARN] Leave-P-Out pesante, ma eseguito in batch mode.")

        # 3. Esecuzione
        print("\n [3/4] Avvio Elaborazione...")
        if val_mode in ["holdout", "all"]:
            execute_validation("holdout", X, y, model_params, val_params, paths)
        if val_mode in ["kfold", "all"]:
            execute_validation("kfold", X, y, model_params, val_params, paths)
        if (val_mode in ["leavepout", "all"]) and should_run_lpo:
            execute_validation("leavepout", X, y, model_params, val_params, paths)

        print("\n [4/4] Salvataggio risultati e grafici / Fine")
        print_separator(" FINITO ")

    except Exception as e:
        print(f" [!] Errore: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
