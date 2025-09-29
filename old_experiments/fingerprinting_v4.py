import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import math
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence, Tuple
import json
from datetime import datetime
import os

import numpy as np
import polars as pl
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, GraphDescriptors, MACCSkeys, rdFingerprintGenerator, AllChem, rdmolops
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from tqdm import tqdm
# from autogluon.tabular import TabularDataset, TabularPredictor
from rdkit.ML.Descriptors import MoleculeDescriptors
import networkx as nx

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import optuna

def canonicalize_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(molecule, canonical=True) if molecule is not None else None

@lru_cache(100_000)
def smiles_to_morgan_fp(smiles_string: str, radius: int = 2, n_bits: int = 2048):
    molecule = Chem.MolFromSmiles(smiles_string)
    if molecule is None:
        return None
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return morgan_generator.GetFingerprint(molecule)

def compute_max_tanimoto_per_train(
    train_smiles: Sequence[str],
    test_smiles: Sequence[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    test_fingerprints: List = [
        fp for fp in (smiles_to_morgan_fp(s, radius, n_bits) for s in test_smiles) if fp is not None
    ]
    if not test_fingerprints:
        raise ValueError("No valid test SMILES after RDKit parsing.")

    max_similarities: List[float] = []
    for train_string in train_smiles:
        train_fp = smiles_to_morgan_fp(train_string, radius, n_bits)
        if train_fp is None:
            max_similarities.append(np.nan)
            continue
        similarities = DataStructs.BulkTanimotoSimilarity(train_fp, test_fingerprints)
        max_similarities.append(max(similarities))

    return np.array(max_similarities, dtype=float)

# TODO: Cache lower-level results.
@lru_cache(maxsize=1_000_000)
def get_feature_vector(
        smiles: str,
        morgan_fingerprint_dim: int,
        atom_pair_fingerprint_dim: int,
        torsion_dim: int,
        use_maccs_keys: bool,
        use_graph_features: bool):
    # PARSE SMILES.
    mol = Chem.MolFromSmiles(smiles)
    
    # GET DESCRIPTORS.
    descriptor_names = [descriptor[0] for descriptor in Descriptors._descList]
    descriptor_generator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    descriptors = np.array(descriptor_generator.CalcDescriptors(mol))

    # GET MORGAN FINGERPRINT.
    morgan_fingerprint = np.array([])
    if morgan_fingerprint_dim > 0:
        morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=morgan_fingerprint_dim)
        morgan_fingerprint = list(morgan_generator.GetFingerprint(mol))

    # GET ATOM PAIR FINGERPRINT.
    atom_pair_fingerprint = np.array([])
    if atom_pair_fingerprint_dim > 0:
        atom_pair_generator = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=atom_pair_fingerprint_dim)
        atom_pair_fingerprint = list(atom_pair_generator.GetFingerprint(mol))

    # GET MACCS.
    maccs_keys = np.array([])
    if use_maccs_keys:
        maccs_keys = MACCSkeys.GenMACCSKeys(mol)
        maccs_keys = list(maccs_keys)

    # GET TORSION FINGERPRINT.
    torsion_fingerprint = np.array([])
    if torsion_dim > 0:
        torsion_generator = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=torsion_dim)
        torsion_fingerprint = list(torsion_generator.GetFingerprint(mol))

    # GET GRAPH FEATURES.
    graph_features = []
    if use_graph_features:
        adjacency_matrix = rdmolops.GetAdjacencyMatrix(mol)
        graph = nx.from_numpy_array(adjacency_matrix)
        graph_diameter = nx.diameter(graph) if nx.is_connected(graph) else 0
        avg_shortest_path = nx.average_shortest_path_length(graph) if nx.is_connected(graph) else 0
        cycle_count = len(list(nx.cycle_basis(graph)))
        graph_features = [graph_diameter, avg_shortest_path, cycle_count]

    # CONCATENATE FEATURES.
    features = np.concatenate([
        descriptors, 
        morgan_fingerprint, 
        atom_pair_fingerprint, 
        maccs_keys, 
        torsion_fingerprint,
        graph_features
    ])
    return features

def get_features_dataframe(
        smiles_df: pd.DataFrame, 
        morgan_fingerprint_dim: int,
        atom_pair_fingerprint_dim: int,
        torsion_dim: int,
        use_maccs_keys: bool,
        use_graph_features: bool) -> tuple[pl.DataFrame, pl.DataFrame]:
    # GET FEATURE NAMES.
    descriptor_names = [descriptor[0] for descriptor in Descriptors._descList]
    morgan_col_names = [f'mfp_{i}' for i in range(morgan_fingerprint_dim)]
    atom_pair_col_names = [f'ap_{i}' for i in range(atom_pair_fingerprint_dim)]
    maccs_col_names = [f'maccs_{i}' for i in range(167)] if use_maccs_keys else []
    torsion_col_names = [f'tt_{i}' for i in range(torsion_dim)]
    graph_col_names = ['graph_diameter', 'avg_shortest_path', 'num_cycles'] if use_graph_features else []
    feature_col_names = descriptor_names + morgan_col_names + atom_pair_col_names + maccs_col_names + torsion_col_names + graph_col_names

    # GET FEATURES.
    features_df = pd.DataFrame(
        np.vstack([
            get_feature_vector(
                smiles,
                morgan_fingerprint_dim,
                atom_pair_fingerprint_dim,
                torsion_dim,
                use_maccs_keys,
                use_graph_features
            ) 
            for smiles 
            in smiles_df['SMILES']]),
        columns=feature_col_names
    )

    # CLEAN FEATURES.
    f32_max = np.finfo(np.float32).max
    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    features_df[features_df > f32_max] = np.nan
    features_df[features_df < -f32_max] = np.nan

    return features_df

def train_test_single_target_models(
        train_test_df: pd.DataFrame,
        extra_data_config: list[dict],
        features_config: dict,
        target_name: str,
        model_class,
        model_kwargs: dict,
        fold_count: int,
        seed: int = 42):
    # PREPROCESS MAIN TRAIN/TEST DATASET DATA.
    filtered_train_test_df = train_test_df.dropna(subset=[target_name])
    train_test_labels = filtered_train_test_df[target_name].reset_index(drop=True)
    
    train_test_features = get_features_dataframe(
        smiles_df=filtered_train_test_df[['SMILES']], 
        **features_config
    )

    models = []
    oof_predictions = np.zeros_like(train_test_labels, dtype=float)
    kf = KFold(fold_count, shuffle=True, random_state=seed)
    for train_indices, test_indices in kf.split(train_test_features):
        # SPLIT DATA.
        train_features, train_labels = train_test_features.iloc[train_indices], train_test_labels.iloc[train_indices]
        test_features, test_labels = train_test_features.iloc[test_indices], train_test_labels.iloc[test_indices]
        train_smiles = filtered_train_test_df['SMILES'].iloc[train_indices]
        test_smiles = filtered_train_test_df['SMILES'].iloc[test_indices]
        sample_weights = [1 for _ in range(len(train_smiles))]

        # ADD EXTRA DATA.
        if len(extra_data_config) > 0:
            extra_train_features, extra_train_labels, extra_sample_weights = load_extra_data(
                extra_data_config,
                features_config,
                target_name,
                train_smiles.to_list(),
                test_smiles.to_list(),
                max_similarity = 0.99
            )

            train_features = pd.concat([train_features, extra_train_features])
            train_labels = pd.concat([train_labels, extra_train_labels])
            sample_weights.extend(extra_sample_weights)
            # print(f'Added {len(extra_train_labels)} extra samples')

        # TRAIN.
        if True: # model_class != TabularPredictor:
            model = model_class(**model_kwargs)
            model.fit(train_features, train_labels, sample_weight=sample_weights)
        else:
            dataset = train_features.copy(deep=True)
            dataset['label'] = train_labels.values
            dataset['sample_weight'] = sample_weights
            model = model_class(**model_kwargs)
            model.fit(dataset, presets='good_quality', time_limit=200)
            # model.fit(dataset, presets='best_quality', time_limit=2_000, refit_full=False)

        models.append(model)

        # TEST.
        predictions = model.predict(test_features)
        oof_predictions[test_indices] = predictions

    mae = mean_absolute_error(train_test_labels, oof_predictions)
    return models, mae

def get_target_weights(csv_path, target_names):
    df = pd.read_csv(csv_path)

    scale_normalization_factors = []
    sample_count_normalization_factors = []
    for target in target_names:
        target_values = df[target].values
        target_values = target_values[~np.isnan(target_values)]

        scale_normalization_factors.append(1 / (max(target_values) - min(target_values)))
        sample_count_normalization_factors.append((1/len(target_values))**0.5)

    scale_normalization_factors = np.array(scale_normalization_factors)
    sample_count_normalization_factors = np.array(sample_count_normalization_factors)

    class_weights = scale_normalization_factors * len(target_names) * sample_count_normalization_factors / sum(sample_count_normalization_factors)

    return class_weights

'''
Input configs like:
[
    {
        'filepath': 'data_filtering/standardized_datasets/Tc/RadonPy.csv',
        'raw_label_weight': 0,
        'dataset_weight': 0.5,
        'max_error_ratio': 2,
        'purge_extra_train_smiles_overlap': True
    },
    ...
]
'''
def load_extra_data(
        extra_data_configs: list[dict],
        features_config: dict,
        target_name: str,
        train_smiles: list[str],
        test_smiles: list[str],
        max_similarity: float) -> tuple[pd.DataFrame, pd.Series, list]:
    # Sorted to prioritize keeping highest-weight data in event of conflicting labels.
    sorted_extra_data_configs = sorted(
        extra_data_configs,
        key=lambda config: config['dataset_weight'],
        reverse=True
    )

    extra_features_dfs = []
    extra_labels = []
    extra_sample_weights = []
    added_smiles = set()
    for extra_data_config in sorted_extra_data_configs:
        # LOAD EXTRA DATA.
        raw_extra_data_df = pd.read_csv(extra_data_config['filepath'])
        
        # REMOVE DUPLICATES & NEAR-DUPLICATES.
        raw_extra_data_df['SMILES'] = raw_extra_data_df['SMILES'].map(canonicalize_smiles)

        # Avoid duplicates within training set(s).
        raw_extra_data_df = raw_extra_data_df[~raw_extra_data_df['SMILES'].isin(added_smiles)] # Avoid overlap between extra train datasets.       
        if extra_data_config['purge_extra_train_smiles_overlap']:
            raw_extra_data_df = raw_extra_data_df[~raw_extra_data_df['SMILES'].isin(train_smiles)] # Avoid overlap with host train dataset.

        # Avoid (near) duplicates vs. test set.
        raw_extra_data_df = raw_extra_data_df[~raw_extra_data_df['SMILES'].isin(test_smiles)]
        extra_train_similarity_scores = np.array(compute_max_tanimoto_per_train(
            train_smiles = raw_extra_data_df['SMILES'].to_list(),
            test_smiles = test_smiles,
        ))
        valid_train_mask = extra_train_similarity_scores < max_similarity
        raw_extra_data_df = raw_extra_data_df[valid_train_mask]

        if len(raw_extra_data_df) == 0:
            print(f'WARNING: Skipping {extra_data_config["filepath"]} because it contributes nothing unique')
            continue

        # MERGE LABEL COLS.
        raw_labels = raw_extra_data_df[f'{target_name}_label']
        scaled_labels = raw_extra_data_df[f'{target_name}_rescaled_label']
        raw_label_weight = extra_data_config['raw_label_weight']
        labels = (raw_labels*raw_label_weight) + (scaled_labels*(1-raw_label_weight))
        
        # DISCARD HIGH ERROR ROWS.
        mae = mean_absolute_error(labels, raw_extra_data_df[f'{target_name}_pred'])
        absolute_errors = (labels - raw_extra_data_df[f'{target_name}_pred']).abs()
        mae_ratios = absolute_errors / mae
        acceptable_error_row_mask = mae_ratios < extra_data_config['max_error_ratio']

        labels = labels[acceptable_error_row_mask]
        raw_extra_data_df = raw_extra_data_df[acceptable_error_row_mask]

        if len(raw_extra_data_df) == 0:
            print(f'WARNING: Skipping {extra_data_config["filepath"]} because all unique rows are above max error threshold')
            continue

        # COMPUTE FEATURES.
        features_df = get_features_dataframe(
            smiles_df=raw_extra_data_df,
            **features_config
        )

        # RECORD DATASET.
        added_smiles.update(raw_extra_data_df['SMILES'].tolist())
        extra_labels.append(labels)
        extra_features_dfs.append(features_df)
        
        dataset_weight = extra_data_config['dataset_weight']
        extra_sample_weights.extend([dataset_weight for _ in range(len(features_df))])

    extra_features_df = pd.concat(extra_features_dfs) if len(extra_features_dfs) > 0 else None
    extra_labels = pd.concat(extra_labels) if len(extra_labels) > 0 else None

    return extra_features_df, extra_labels, extra_sample_weights

def train_models(
        model_class: type, 
        target_names_to_model_config_paths: dict[str, str], 
        target_names_to_extra_data_config_paths: dict[str, str],
        fold_count: int = 5, 
        data_path: str = 'data/from_host_v2/train.csv'):
    # CREATE OUTPUT DIRECTORY.
    output_dir = f"models/{model_class.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    # LOAD DATA.
    train_test_df = pd.read_csv(data_path)
    train_test_df['SMILES'] = train_test_df['SMILES'].map(canonicalize_smiles)

    # TRAIN & TEST MODELS
    maes = []
    target_names = list(target_names_to_model_config_paths.keys())
    for target_index, target_name in enumerate(target_names):
        # LOAD MODEL CONFIG.
        with open(target_names_to_model_config_paths[target_name], 'r') as model_config_file:
            model_config = json.load(model_config_file)

        features_config = {
            'morgan_fingerprint_dim' : model_config.get("morgan_fingerprint_dim", 0),
            'atom_pair_fingerprint_dim' : model_config.get("atom_pair_fingerprint_dim", 0),
            'torsion_dim' : model_config.get("torsion_dim", 0),
            'use_maccs_keys' : model_config.get("use_maccs_keys", False),
            'use_graph_features' : model_config.get("use_graph_features", False)
        }
        with open(f'{output_dir}/{target_name}_features_config.json', 'w') as features_config_file:
            json.dump(features_config, features_config_file)

        model_kwargs = {k: v for k, v in model_config.items() if k not in {
            "morgan_fingerprint_dim", "atom_pair_fingerprint_dim", 
            "torsion_dim", "use_maccs_keys", "use_graph_features"}}
        
        # ADD DEFAULT SETTINGS IF MISSING.
        if model_class == XGBRegressor:
            model_kwargs.setdefault("device", "cuda")
            model_kwargs.setdefault("tree_method", "hist")
        elif model_class == CatBoostRegressor:
            model_kwargs.setdefault("task_type", "GPU")
            model_kwargs.setdefault("logging_level", "Silent")
        elif model_class == LGBMRegressor:
            model_kwargs.setdefault("verbose", -1)
        # elif model_class == TabularPredictor:
        #     model_kwargs.setdefault("label", "label")
        #     model_kwargs.setdefault("sample_weight", "sample_weight")
        #     model_kwargs.setdefault("weight_evaluation", True)
        #     model_kwargs.setdefault("eval_metric", "mean_absolute_error")

        # LOAD EXTRA DATA CONFIG.
        with open(target_names_to_extra_data_config_paths[target_name], 'r') as extra_data_config_file:
            raw_extra_data_config = json.load(extra_data_config_file)

        extra_dataset_configs = []
        extra_dataset_count = int(len(raw_extra_data_config.keys()) / 5)
        for extra_dataset_index in range(extra_dataset_count):
            extra_dataset_configs.append({
                'filepath': raw_extra_data_config[f'filepath_{extra_dataset_index}'],
                'raw_label_weight': raw_extra_data_config[f'raw_label_weight_{extra_dataset_index}'],
                'dataset_weight': raw_extra_data_config[f'dataset_weight_{extra_dataset_index}'],
                'max_error_ratio': raw_extra_data_config[f'max_error_ratio_{extra_dataset_index}'],
                'purge_extra_train_smiles_overlap': raw_extra_data_config[f'purge_extra_train_smiles_overlap_{extra_dataset_index}'],
            })

        # TRAIN & TEST MODEL.
        models, mae = train_test_single_target_models(
            train_test_df=train_test_df,
            extra_data_config=extra_dataset_configs,
            features_config=features_config,
            target_name=target_name,
            model_class=model_class,
            model_kwargs=model_kwargs,
            fold_count=fold_count,
            seed=69,
        )

        # SAVE MODELS & STATS.
        maes.append(mae)

        for fold_index, model in enumerate(models):
            model_filename = f"{target_name}_fold{fold_index}_mae{int(mae * 10000)}.pkl"
            model_filepath = os.path.join(output_dir, model_filename)
            if True: # model_class != TabularPredictor:
                with open(model_filepath, "wb") as model_file:
                    pickle.dump(model, model_file)
            else:
                model.clone(model_filepath)

        print(f"{target_name}: MAE = {mae:.5f}")

    # LOG wMAE.
    class_weights = get_target_weights(data_path, target_names)
    weighted_mae = np.average(maes, weights=class_weights)
    print(f"Weighted average MAE: {weighted_mae:.5f}")


def get_optimal_model_config(model_class, target_name, trial_count, data_path='data/from_host/train.csv'):
    def objective(trial) -> float:
        # LOAD DATA.
        labeled_smiles_df = pd.read_csv(data_path)
        labeled_smiles_df['SMILES'] = labeled_smiles_df['SMILES'].map(canonicalize_smiles)
        labeled_smiles_df = labeled_smiles_df.dropna(subset=target_name)

        # COMPUTE FEATURES.
        features_df = get_features_dataframe(
            labeled_smiles_df, 
            morgan_fingerprint_dim = trial.suggest_categorical('morgan_fingerprint_dim', [0, 512, 1024, 2048]),
            atom_pair_fingerprint_dim = trial.suggest_categorical('atom_pair_fingerprint_dim', [0, 512, 1024, 2048]),
            torsion_dim = trial.suggest_categorical('torsion_dim', [0, 512, 1024, 2048]),
            use_maccs_keys = trial.suggest_categorical('use_maccs_keys', [True, False]),
            use_graph_features = trial.suggest_categorical('use_graph_features', [True, False]),
        )

        # PICK MODEL CONFIG.
        if model_class == XGBRegressor:
            model_kwargs = {
                "device": "cuda",
                "tree_method": "hist",
                "n_estimators": trial.suggest_int("n_estimators", 50, 3000, log=True),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 20.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 20.0, log=True),
                "objective": trial.suggest_categorical("objective", ["reg:squarederror", "reg:absoluteerror", "reg:pseudohubererror"]),
            }
        elif model_class == CatBoostRegressor:
            loss_function = trial.suggest_categorical("loss_function", ["RMSE", "MAE"])
            model_kwargs = {
                "task_type": "GPU",
                # "iterations": trial.suggest_int("iterations", 100, 1000),
                "iterations": trial.suggest_int("iterations", 100, 1500),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.4, log=True),
                # "depth": trial.suggest_int("depth", 3, 10),
                "depth": trial.suggest_int("depth", 3, 12),
                # "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 15.0),
                "border_count": trial.suggest_int("border_count", 128, 254),
                "loss_function": loss_function,
                "logging_level": "Silent",
            }
        elif model_class == LGBMRegressor:
            model_kwargs = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 40_000, log=True),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 12),
                "num_leaves": trial.suggest_int("num_leaves", 8, 512, log=True),
                # "min_child_samples": trial.suggest_int("min_child_samples", 5, 100), # 5
                # "subsample": trial.suggest_float("subsample", 0.5, 1.0), # 0.54
                # "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0), # 0.52
                # "colsample_node": trial.suggest_float("colsample_node", 0.5, 1.0), # 0.53
                "min_child_samples": trial.suggest_int("min_child_samples", 1, 100),
                "subsample": trial.suggest_float("subsample", 0.4, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "colsample_node": trial.suggest_float("colsample_node", 0.4, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "objective": trial.suggest_categorical("objective", ["regression", "mae", "huber"]),
                "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
                "verbose": -1,
            }
        elif model_class == TabularPredictor:
            model_kwargs = {
                'label': 'label',
                'eval_metric': 'mean_absolute_error',
                # 'verbosity': 1,
            }

        _, mae = train_test_single_target_models(
            train_test_features=features_df,
            train_test_labels=labeled_smiles_df[target_name],
            model_class=model_class,
            model_kwargs=model_kwargs,
            fold_count=5
        )
        return mae
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=trial_count, show_progress_bar=True)

    print(f"\nBest sMAE: {study.best_value:.5f}")
    print("Params:")
    print(json.dumps(study.best_params, indent=4))

    output_filepath = f'configs/{model_class.__name__}_{target_name}_{int(study.best_value * 10000)}.json'
    with open(output_filepath, 'w') as output_file:
        json.dump(study.best_params, output_file, indent=4)

def get_optimal_data_config(target_name, model_class, model_config_path, trial_count):
    def objective(trial: optuna.Trial) -> float:
        # LOAD STANDARD DATASET.
        train_test_df = pd.read_csv('data/from_host_v2/train.csv')
        train_test_df['SMILES'] = train_test_df['SMILES'].map(canonicalize_smiles)

        # LOAD MODEL CONFIG.
        with open(model_config_path, 'r') as model_config_file:
            best_params = json.load(model_config_file)

        features_config = {
            'morgan_fingerprint_dim' : best_params.get("morgan_fingerprint_dim", 0),
            'atom_pair_fingerprint_dim' : best_params.get("atom_pair_fingerprint_dim", 0),
            'torsion_dim' : best_params.get("torsion_dim", 0),
            'use_maccs_keys' : best_params.get("use_maccs_keys", False),
            'use_graph_features' : best_params.get("use_graph_features", False)
        }

        model_kwargs = {k: v for k, v in best_params.items() if k not in {
            "morgan_fingerprint_dim", "atom_pair_fingerprint_dim", 
            "torsion_dim", "use_maccs_keys", "use_graph_features"}}

        if model_class == XGBRegressor:
            model_kwargs.setdefault("device", "cuda")
            model_kwargs.setdefault("tree_method", "hist")
        elif model_class == CatBoostRegressor:
            model_kwargs.setdefault("task_type", "GPU")
            model_kwargs.setdefault("logging_level", "Silent")
        elif model_class == LGBMRegressor:
            model_kwargs.setdefault("verbose", -1)
        elif model_class == TabularPredictor:
            model_kwargs.setdefault("label", "label")
            model_kwargs.setdefault("sample_weight", "sample_weight")
            model_kwargs.setdefault("weight_evaluation", True)
            model_kwargs.setdefault("eval_metric", "mean_absolute_error")

        # GET EXTRA DATA CONFIG.
        TARGETS_TO_DATA_FILENAMES = {
            'Tg': [
                'dmitry_2.csv',
                'dmitry_3.csv',
                'host_extra.csv',
                'LAMALAB.csv',
            ],
            'FFV': [
                'host_extra.csv'
            ],
            'Tc': [
                'host_extra.csv',
                'RadonPy.csv',
                'RadonPy_filtered.csv',
            ],
            'Density': [
                'dmitry.csv',
                'RadonPy.csv',
            ],
            'Rg': [
                'RadonPy.csv'
            ],
        }
        data_filenames = TARGETS_TO_DATA_FILENAMES[target_name]

        extra_data_configs = []
        for dataset_index, data_filename in enumerate(data_filenames):        
            extra_data_configs.append({
                'filepath': trial.suggest_categorical(f'filepath_{dataset_index}', [f'data_filtering/standardized_datasets/{target_name}/{data_filename}']),
                'raw_label_weight': trial.suggest_float(f'raw_label_weight_{dataset_index}', low=0, high=1),
                'dataset_weight': trial.suggest_float(f'dataset_weight_{dataset_index}', low=0, high=1),
                'max_error_ratio': trial.suggest_float(f'max_error_ratio_{dataset_index}', low=0.5, high=5),
                'purge_extra_train_smiles_overlap': trial.suggest_categorical(f'purge_extra_train_smiles_overlap_{dataset_index}', [True, False]),
            })

        # TRAIN & TEST.
        _, mae = train_test_single_target_models(
            train_test_df=train_test_df,
            extra_data_config=extra_data_configs,
            features_config=features_config,
            target_name=target_name,
            model_class=model_class,
            model_kwargs=model_kwargs,
            fold_count=5,
            seed=1337,
        )
        return mae
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=trial_count, show_progress_bar=True)

    print(f"\nBest wMAE: {study.best_value:.5f}")
    print("Params:")
    print(json.dumps(study.best_params, indent=4))

    output_filepath = f'configs/{model_class.__name__}_data_{target_name}_{int(study.best_value * 10000)}.json'
    with open(output_filepath, 'w') as output_file:
        json.dump(study.best_params, output_file, indent=4)

if __name__ == '__main__':
    # TARGET_NAMES= ["Tg", "FFV", "Tc", "Density", "Rg"]
    # for target_name in TARGET_NAMES:
    #     get_optimal_data_config(
    #         target_name=target_name, 
    #         # model_class=XGBRegressor, # TODO: Running bottom left
    #         # model_config_path={
    #         #     'Tg': 'configs/XGBRegressor_Tg_493667.json',
    #         #     'FFV': 'configs/XGBRegressor_FFV_53.json',
    #         #     'Tc': 'configs/XGBRegressor_Tc_254.json',
    #         #     'Density': 'configs/XGBRegressor_Density_293.json',
    #         #     'Rg': 'configs/XGBRegressor_Rg_14916.json',
    #         # }[target_name],
    #         model_class=LGBMRegressor, # TODO: Running right
    #         model_config_path={
    #             'Tg': 'configs/LGBMRegressor_Tg_486709.json',
    #             'FFV': 'configs/LGBMRegressor_FFV_49.json',
    #             'Tc': 'configs/LGBMRegressor_Tc_247.json',
    #             'Density': 'configs/LGBMRegressor_Density_296.json',
    #             'Rg': 'configs/LGBMRegressor_Rg_14593.json',
    #         }[target_name],
    #         # model_class=CatBoostRegressor, # TODO: Running top left
    #         # model_config_path={
    #         #     'Tg': 'configs/CatBoostRegressor_Tg_492811.json',
    #         #     'FFV': 'configs/CatBoostRegressor_FFV_58.json',
    #         #     'Tc': 'configs/CatBoostRegressor_Tc_253.json',
    #         #     'Density': 'configs/CatBoostRegressor_Density_310.json',
    #         #     'Rg': 'configs/CatBoostRegressor_Rg_15101.json',
    #         # }[target_name],
    #         # model_class=TabularPredictor, # TODO: Running center
    #         # model_config_path={
    #         #     'Tg': 'configs/TabularPredictor_Tg_487930.json',
    #         #     'FFV': 'configs/TabularPredictor_FFV_53.json',
    #         #     'Tc': 'configs/TabularPredictor_Tc_253.json',
    #         #     'Density': 'configs/TabularPredictor_Density_290.json',
    #         #     'Rg': 'configs/TabularPredictor_Rg_14917.json',
    #         # }[target_name],
    #         trial_count=50, # GBDT
    #         # trial_count=10, # AutoGluon
    #     )

    # Equal weights, LGBMRegressor_20250722_210924, 0.06756
    # V0 tuning, LGBMRegressor_20250722_212718, 0.06658
    # V0.1 tuning, LGBMRegressor_20250722_222206, 0.06460
    # V1 tuning (20 trials), LGBMRegressor_20250724_002831, 0.06199 <-- # TODO: Submit as LB rerun prep
    #       Tg: MAE = 45.89876
    #       FFV: MAE = 0.00489
    #       Tc: MAE = 0.02497 (used older config)
    #       Density: MAE = 0.02740 (used older config)
    #       Rg: MAE = 1.36233 (used older config)
    # V1.1 Tuning, LGBMRegressor_20250724_234051, 0.06222
    #     Tg: MAE = 45.89876
    #     FFV: MAE = 0.00489
    #     Tc: MAE = 0.02567
    #     Density: MAE = 0.02653
    #     Rg: MAE = 1.36864
    '''
    XGBoost (wMAE = 0.06220):
        Tg: MAE = 44.77303
        FFV: MAE = 0.00521
        Tc: MAE = 0.02533
        Density: MAE = 0.02608
        Rg: MAE = 1.43795
    CatBoost (wMAE = 0.06268):
        Tg: MAE = 44.62340
        FFV: MAE = 0.00577
        Tc: MAE = 0.02478
        Density: MAE = 0.02781
        Rg: MAE = 1.46541
    TabularPredictor (0.06101):
        Rg: MAE = 1.42686
    '''
    train_models(
        model_class=XGBRegressor,
        target_names_to_model_config_paths={
            'Tg': 'configs/XGBRegressor_Tg_493667.json',
            'FFV': 'configs/XGBRegressor_FFV_53.json',
            'Tc': 'configs/XGBRegressor_Tc_254.json',
            'Density': 'configs/XGBRegressor_Density_293.json',
            'Rg': 'configs/XGBRegressor_Rg_14916.json',
        },
        target_names_to_extra_data_config_paths = {
            'Tg': 'configs/XGBRegressor_data_Tg_447786.json',
            'FFV': 'configs/XGBRegressor_data_FFV_51.json',
            'Tc': 'configs/XGBRegressor_data_Tc_248.json',
            'Density': 'configs/XGBRegressor_data_Density_254.json',
            'Rg': 'configs/XGBRegressor_data_Rg_14271.json',
        }
        # model_class=LGBMRegressor,
        # target_names_to_model_config_paths={
        #     'Tg': 'configs/LGBMRegressor_Tg_486709.json',
        #     'FFV': 'configs/LGBMRegressor_FFV_49.json',
        #     'Tc': 'configs/LGBMRegressor_Tc_247.json',
        #     'Density': 'configs/LGBMRegressor_Density_296.json',
        #     'Rg': 'configs/LGBMRegressor_Rg_14593.json',
        # },
        # target_names_to_extra_data_config_paths = {
        #     'Tg': 'configs/LGBMRegressor_data_Tg_461983.json',
        #     'FFV': 'configs/LGBMRegressor_data_FFV_48.json',
        #     'Tc': 'configs/LGBMRegressor_data_Tc_252.json',
        #     'Density': 'configs/LGBMRegressor_data_Density_262.json',
        #     'Rg': 'configs/LGBMRegressor_data_Rg_13684.json',
        # }
        # model_class=CatBoostRegressor,
        # target_names_to_model_config_paths={
        #     'Tg': 'configs/CatBoostRegressor_Tg_492811.json',
        #     'FFV': 'configs/CatBoostRegressor_FFV_58.json',
        #     'Tc': 'configs/CatBoostRegressor_Tc_253.json',
        #     'Density': 'configs/CatBoostRegressor_Density_310.json',
        #     'Rg': 'configs/CatBoostRegressor_Rg_15101.json',
        # },
        # target_names_to_extra_data_config_paths = {
        #     'Tg': 'configs/CatBoostRegressor_data_Tg_443744.json',
        #     'FFV': 'configs/CatBoostRegressor_data_FFV_56.json',
        #     'Tc': 'configs/CatBoostRegressor_data_Tc_244.json',
        #     'Density': 'configs/CatBoostRegressor_data_Density_276.json',
        #     'Rg': 'configs/CatBoostRegressor_data_Rg_14720.json',
        # }
        # model_class=TabularPredictor,
        # target_names_to_model_config_paths={
        #     'Tg': 'configs/TabularPredictor_Tg_487930.json',
        #     'FFV': 'configs/TabularPredictor_FFV_53.json',
        #     'Tc': 'configs/TabularPredictor_Tc_253.json',
        #     'Density': 'configs/TabularPredictor_Density_290.json',
        #     'Rg': 'configs/TabularPredictor_Rg_14917.json',
        # },
        # target_names_to_extra_data_config_paths = {
        #     'Tg': 'configs/TabularPredictor_data_Tg_440748.json',
        #     'FFV': 'configs/TabularPredictor_data_FFV_51.json',
        #     'Tc': 'configs/TabularPredictor_data_Tc_249.json',
        #     'Density': 'configs/TabularPredictor_data_Density_255.json',
        #     'Rg': 'configs/TabularPredictor_data_Rg_14477.json',
        # }
    )