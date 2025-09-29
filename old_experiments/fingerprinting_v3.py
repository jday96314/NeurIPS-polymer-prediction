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
from rdkit import Chem
from rdkit.Chem import Descriptors, GraphDescriptors, MACCSkeys, rdFingerprintGenerator, AllChem, rdmolops
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from tqdm import tqdm
from autogluon.tabular import TabularDataset, TabularPredictor
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

MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
ATOMPAIR_GEN = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)
TORSION_GEN = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)

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
        all_features: pd.DataFrame, 
        all_labels: pd.Series, 
        model_class,
        model_kwargs: dict,
        fold_count: int,
        seed: int = 42):
    models = []
    oof_predictions = np.zeros_like(all_labels, dtype=float)
    kf = KFold(fold_count, shuffle=True, random_state=seed)
    for train_indices, test_indices in kf.split(all_features):
        # SPLIT DATA.
        train_features, train_labels = all_features.iloc[train_indices], all_labels.iloc[train_indices]
        test_features, test_labels = all_features.iloc[test_indices], all_labels.iloc[test_indices]

        # TRAIN.
        if model_class != TabularPredictor:
            model = model_class(**model_kwargs)
            model.fit(train_features, train_labels)
        else:
            dataset = train_features.copy(deep=True)
            dataset['label'] = train_labels.values
            model = model_class(**model_kwargs)
            model.fit(dataset, presets='good_quality', time_limit=200)
            # model.fit(dataset, presets='best_quality', time_limit=2_000, refit_full=False)

        models.append(model)

        # TEST.
        predictions = model.predict(test_features)
        oof_predictions[test_indices] = predictions

    mae = mean_absolute_error(all_labels, oof_predictions)
    return models, mae

def get_optimal_config(model_class, target_name, trial_count, data_path='data/from_host/train.csv'):
    def objective(trial) -> float:
        # LOAD DATA.
        labeled_smiles_df = pd.read_csv(data_path)
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
            all_features=features_df,
            all_labels=labeled_smiles_df[target_name],
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

def train_models(
        model_class: type, 
        target_names_to_config_paths: dict[str, str], 
        fold_count: int = 5, 
        data_path: str = 'data/from_host/train.csv'):
    # CREATE OUTPUT DIRECTORY.
    output_dir = f"models/{model_class.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    # LOAD DATA.
    df = pd.read_csv(data_path)

    # TRAIN & TEST MODELS
    maes = []
    target_names = list(target_names_to_config_paths.keys())
    for target_index, target_name in enumerate(target_names):
        # LOAD CONFIG.
        with open(target_names_to_config_paths[target_name], 'r') as f:
            best_params = json.load(f)

        features_config = {
            'morgan_fingerprint_dim' : best_params.get("morgan_fingerprint_dim", 0),
            'atom_pair_fingerprint_dim' : best_params.get("atom_pair_fingerprint_dim", 0),
            'torsion_dim' : best_params.get("torsion_dim", 0),
            'use_maccs_keys' : best_params.get("use_maccs_keys", False),
            'use_graph_features' : best_params.get("use_graph_features", False)
        }
        with open(f'{output_dir}/{target_name}_features_config.json', 'w') as features_config_file:
            json.dump(features_config, features_config_file)

        model_kwargs = {k: v for k, v in best_params.items() if k not in {
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
        elif model_class == TabularPredictor:
            model_kwargs.setdefault("label", "label")
            model_kwargs.setdefault("eval_metric", "mean_absolute_error")
        
        # PREPROCESS DATA.
        target_df = df.dropna(subset=[target_name])
        smiles_df = target_df[['SMILES']]
        labels = target_df[target_name].reset_index(drop=True)
        
        features_df = get_features_dataframe(
            smiles_df=smiles_df, 
            **features_config
        )

        # TRAIN & TEST MODEL.
        models, mae = train_test_single_target_models(
            all_features=features_df,
            all_labels=labels,
            model_class=model_class,
            model_kwargs=model_kwargs,
            fold_count=fold_count,
            seed=69
            # seed=42
        )

        # SAVE MODELS & STATS.
        maes.append(mae)

        for fold_index, model in enumerate(models):
            model_filename = f"{target_name}_fold{fold_index}_mae{int(mae * 10000)}.pkl"
            model_filepath = os.path.join(output_dir, model_filename)
            if model_class != TabularPredictor:
                with open(model_filepath, "wb") as model_file:
                    pickle.dump(model, model_file)
            else:
                model.clone(model_filepath)

        print(f"{target_name}: MAE = {mae:.5f}")

    # LOG wMAE.
    class_weights = get_target_weights(data_path, target_names)
    weighted_mae = np.average(maes, weights=class_weights)
    print(f"Weighted average MAE: {weighted_mae:.5f}")

'''
All except for CatBoost -> Run with seed 42 before 69
CatBoost -> Run with seed 69, then 42

Seed 42:
    XGBoost (wMAE = 0.06616):
        Tg: MAE = 49.36676
        FFV: MAE = 0.00534
        Tc: MAE = 0.02544
        Density: MAE = 0.02933
        Rg: MAE = 1.49169
    LightGBM (wMAE = 0.06506):
        Tg: MAE = 48.67097
        FFV: MAE = 0.00500
        Tc: MAE = 0.02471
        Density: MAE = 0.02969
        Rg: MAE = 1.45933
    CatBoost (wMAE = 0.06685):
        Tg: MAE = 49.28110
        FFV: MAE = 0.00590
        Tc: MAE = 0.02538
        Density: MAE = 0.03104
        Rg: MAE = 1.51019
    AutoGluon (wMAE = 0.06605):
        Rg: MAE = 1.51149
Seed 69:
    XGBoost (wMAE = 0.06695):
        Tg: MAE = 50.98381
        FFV: MAE = 0.00536
        Tc: MAE = 0.02607
        Density: MAE = 0.02783
        Rg: MAE = 1.49142
    LightGBM (wMAE = 0.06603):
        Tg: MAE = 50.39704
        FFV: MAE = 0.00499
        Tc: MAE = 0.02557
        Density: MAE = 0.02827
        Rg: MAE = 1.45782
    CatBoost (wMAE = 0.06723):
        Tg: MAE = 50.00325
        FFV: MAE = 0.00583
        Tc: MAE = 0.02557
        Density: MAE = 0.02964
        Rg: MAE = 1.53415
    AutoGluon (wMAE = 0.06591):
        Rg: MAE = 1.50385
    AutoGluon 10x compute (wMAE = 0.06428):
        Rg: MAE = 1.47592
    AutoGluon, rerun with normal compute & 5800x3d (TabularPredictor_20250715_190532)
        Weighted average MAE: 0.06546
        Rg: MAE = 1.47908
'''
if __name__ == '__main__':
    # TARGET_NAMES= ["Tg", "FFV", "Tc", "Density", "Rg"]
    # for target_name in TARGET_NAMES:
    #     get_optimal_config(
    #         target_name=target_name, 
    #         # model_class=XGBRegressor,
    #         # trial_count=120,
    #         model_class=CatBoostRegressor,
    #         trial_count=75, # 90,
    #         # model_class=LGBMRegressor,
    #         # trial_count=120,
    #         # model_class=TabularPredictor,
    #         # trial_count=12,
    #     )

    train_models(
        # model_class=XGBRegressor,
        # target_names_to_config_paths={
        #     'Tg': 'configs/XGBRegressor_Tg_493667.json',
        #     'FFV': 'configs/XGBRegressor_FFV_53.json',
        #     'Tc': 'configs/XGBRegressor_Tc_254.json',
        #     'Density': 'configs/XGBRegressor_Density_293.json',
        #     'Rg': 'configs/XGBRegressor_Rg_14916.json',
        # },
        # model_class=LGBMRegressor,
        # target_names_to_config_paths={
        #     'Tg': 'configs/LGBMRegressor_Tg_486709.json',
        #     'FFV': 'configs/LGBMRegressor_FFV_49.json',
        #     'Tc': 'configs/LGBMRegressor_Tc_247.json',
        #     'Density': 'configs/LGBMRegressor_Density_296.json',
        #     'Rg': 'configs/LGBMRegressor_Rg_14593.json',
        # },
        # model_class=CatBoostRegressor,
        # target_names_to_config_paths={
        #     'Tg': 'configs/CatBoostRegressor_Tg_492811.json',
        #     'FFV': 'configs/CatBoostRegressor_FFV_58.json',
        #     'Tc': 'configs/CatBoostRegressor_Tc_253.json',
        #     'Density': 'configs/CatBoostRegressor_Density_310.json',
        #     'Rg': 'configs/CatBoostRegressor_Rg_15101.json',
        # },
        model_class=TabularPredictor,
        target_names_to_config_paths={
            'Tg': 'configs/TabularPredictor_Tg_487930.json',
            'FFV': 'configs/TabularPredictor_FFV_53.json',
            'Tc': 'configs/TabularPredictor_Tc_253.json',
            'Density': 'configs/TabularPredictor_Density_290.json',
            'Rg': 'configs/TabularPredictor_Rg_14917.json',
        }
    )