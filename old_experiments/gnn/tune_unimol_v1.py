#!/usr/bin/env python3
"""
Optuna hyper‑parameter search for Uni‑Mol‑v2 (84 M) regression tasks.

Usage
-----
python tune_unimol.py --target <TARGET_NAME> --trials <N_TRIALS>
"""

import argparse
import json
import shutil
import uuid
from pathlib import Path
from typing import Dict
import traceback

import joblib
import optuna

from unimol_tools import MolTrain, MolPredict


def parse_arguments() -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(
        description="Hyper‑parameter optimisation for Uni‑Mol‑v2 84 M."
    )
    parser.add_argument(
        "--target",
        required=True,
        type=str,
        help="Column name (target) to predict, e.g. 'Tg'.",
    )
    parser.add_argument(
        "--trials",
        required=True,
        type=int,
        help="Number of Optuna trials to execute.",
    )
    parser.add_argument(
        "--data",
        default="gnn/unimol_datasets/from_host/{target}.csv",
        type=str,
        help=(
            "Path template for the dataset CSV. "
            "Use '{target}' as a placeholder for the target name."
        ),
    )
    parser.add_argument(
        "--storage",
        default=None,
        type=str,
        help="Optional Optuna storage URL for persistent studies.",
    )
    return parser.parse_args()


def create_experiment_directory(root_directory: Path, trial_number: int) -> Path:
    """Create a dedicated directory for a single Optuna trial."""
    experiment_directory = root_directory / f"trial_{trial_number:04d}_{uuid.uuid4().hex}"
    experiment_directory.mkdir(parents=True, exist_ok=True)
    return experiment_directory


def load_mae(metric_file: Path) -> float:
    """Load the mean absolute error written by MolTrain."""
    metrics: Dict[str, float] = joblib.load(metric_file)
    return float(metrics["mae"])


def objective(
    trial: optuna.trial.Trial,
    target_name: str,
    data_path: Path,
    experiment_root: Path,
) -> float:
    """Single Optuna trial."""
    # epoch_count: int = trial.suggest_int("epochs", 2, 40)
    # learning_rate: float = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    # batch_size: int = trial.suggest_categorical("batch_size", [8, 16])
    epoch_count: int = trial.suggest_int("epochs", 30, 70)
    learning_rate: float = trial.suggest_float("learning_rate", 5e-5, 1.5e-3, log=True)
    batch_size: int = trial.suggest_categorical("batch_size", [8, 16])
    replacement_strategy: str = trial.suggest_categorical("replacement_strategy", ['', '_C', '_At'])

    experiment_directory: Path = create_experiment_directory(
        experiment_root, trial.number
    )

    trainer = MolTrain(
        task="regression",
        data_type="molecule",
        epochs=epoch_count,
        learning_rate=learning_rate,
        batch_size=batch_size,
        kfold=5,
        model_name="unimolv2",
        model_size="84m",
        early_stopping=1e9,
        metrics="mae",
        conf_cache_level=2,
        save_path=str(experiment_directory),
    )

    raw_data_path = str(data_path)
    raw_data_path = raw_data_path.replace('.csv', f'{replacement_strategy}.csv')
    trainer.fit(data=raw_data_path)

    metric_file: Path = experiment_directory / "metric.result"
    mae_score: float = load_mae(metric_file)

    # Optional: clean up to save disk space
    shutil.rmtree(experiment_directory, ignore_errors=True)

    return mae_score  # Optuna minimises by default

def safe_objective(*args, **kwargs):
    try:
        return objective(*args, **kwargs)
    except:
        traceback.print_exc()
        return 100

'''
Tg:
    {
        "epochs": 40,
        "learning_rate": 0.0007174057925944157,
        "batch_size": 8,
        "mae": 47.7172021971562
    }
FFV:
    TODO: Running 3090, top, reduced search range
Tc:
    {
        "epochs": 40,
        "learning_rate": 0.0002294345507838138,
        "batch_size": 8,
        "mae": 0.022057177832748225
    }
Density:
    {
        "epochs": 40,
        "learning_rate": 0.00013659089436050554,
        "batch_size": 16,
        "mae": 0.024114097675668907
    }
Rg:
    {
        "epochs": 40,
        "learning_rate": 0.00010684814134962663,
        "batch_size": 8,
        "mae": 1.362561852835438
    }
'''
def main() -> None:
    """Entry point."""
    args = parse_arguments()
    data_file: Path = Path(args.data.format(target=args.target)).expanduser().resolve()

    if not data_file.exists():
        raise FileNotFoundError(f"Dataset not found at {data_file}")

    study = optuna.create_study(
        direction="minimize",
        study_name=f"unimolv2_{args.target}",
        storage=args.storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )

    experiment_root = Path("gnn/optuna_runs")
    experiment_root.mkdir(exist_ok=True)

    study.optimize(
        lambda trial: safe_objective(
            trial=trial,
            target_name=args.target,
            data_path=data_file,
            experiment_root=experiment_root,
        ),
        n_trials=args.trials,
        show_progress_bar=True,
    )

    print(json.dumps(study.best_trial.params, indent=2))

if __name__ == "__main__":
    # main()

    # Rg (1.3590926417750255)
    # {'epochs': 53, 'learning_rate': 0.0002838578498771946, 'batch_size': 16, 'replacement_strategy': '_At'}

    # Tg (46.5?)
    # { 
    #     "epochs": 61,
    #     "learning_rate": 0.0004802722104942256,
    #     "batch_size": 16,
    #     'replacement_strategy': ''?
    # }

    #'''
    # TARGET_NAME = 'Tg' 
    # TARGET_NAME = 'FFV'
    # TARGET_NAME = 'Tc' 
    # TARGET_NAME = 'Density'
    # TARGET_NAME = 'Rg' # 
    # PLACEHOLDER_ELEMENT_SUFFIX = '_At'
    TARGET_NAME = 'Tg' # TODO: Running center
    PLACEHOLDER_ELEMENT_SUFFIX = '_C'
    # TARGET_NAME = 'Density'
    # PLACEHOLDER_ELEMENT_SUFFIX = ''
    # TARGET_NAME = 'Density' # TODO: Running top left
    # PLACEHOLDER_ELEMENT_SUFFIX = '_At'

    '''
    TARGET_NAMES_TO_CONFIGS = {
        "Tg": { # 47.398 *, 49.827 C, 51.253 At  
            "epochs": 40,
            "learning_rate": 0.0007174057925944157,
            "batch_size": 8,
        },
        "FFV": {
            "epochs": 40,
            "learning_rate": 0.0002,
            "batch_size": 8,
        },
        "Tc": { # 0.0221 *, 0220 C, 0.0207 At
            "epochs": 40,
            "learning_rate": 0.0002294345507838138,
            "batch_size": 8,
        },
        "Density": { # 0.0254 *
            "epochs": 40,
            "learning_rate": 0.00013659089436050554,
            "batch_size": 16,
        },
        "Rg": { # 1.362 * 
            "epochs": 40,
            "learning_rate": 0.00010684814134962663,
            "batch_size": 8,
        },
    }
    '''

    TARGET_NAMES_TO_CONFIGS = {
        'Tg': { # '' 50.588, _At ERR, C ERR
            'epochs': 61, ''
            'learning_rate': 0.0004802722104942256, 
            'batch_size': 16
        },
        'Density': { # '' 0.02225, _At 0.02297
            "epochs": 68,
            "learning_rate": 6.150604962297023e-05,
            "batch_size": 8,
        },
        'Rg': { # _At 1.381 
            'epochs': 53, ''
            'learning_rate': 0.0002838578498771946, 
            'batch_size': 16
        },
    }

    trainer = MolTrain(
        task="regression",
        data_type="molecule",
        epochs=TARGET_NAMES_TO_CONFIGS[TARGET_NAME]['epochs'],
        learning_rate=TARGET_NAMES_TO_CONFIGS[TARGET_NAME]['learning_rate'],
        batch_size=TARGET_NAMES_TO_CONFIGS[TARGET_NAME]['batch_size'],
        kfold=5,
        model_name="unimolv2",
        model_size="84m",
        early_stopping=1e9,
        metrics="mae",
        conf_cache_level=2,
        save_path=f'models/UniMol2_2025_07_17{PLACEHOLDER_ELEMENT_SUFFIX}/{TARGET_NAME}',
    )

    data_path = f"gnn/unimol_datasets/from_host/{TARGET_NAME}{PLACEHOLDER_ELEMENT_SUFFIX}.csv"
    trainer.fit(data=str(data_path))
    #'''