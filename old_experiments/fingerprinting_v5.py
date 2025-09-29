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
import traceback
try:
    import optuna
    from catboost import CatBoostRegressor
    from xgboost import XGBRegressor
    from autogluon.tabular import TabularDataset, TabularPredictor
    from pytabkit import RealMLP_TD_Regressor, TabM_D_Regressor
except:
    RealMLP_TD_Regressor, TabM_D_Regressor = None, None
    print('WARNING: Missing dependencies. Continuing anyway.')
    traceback.print_exc()
from rdkit.ML.Descriptors import MoleculeDescriptors
import networkx as nx

from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.feature_selection import VarianceThreshold

from side_chain_features import ALL_SIDECHAIN_BACKBONE_FEATURE_NAMES, IMPORTANT_SIDECHAIN_BACKBONE_FEATURE_NAMES, extract_sidechain_and_backbone_features

def augment_dataset(
        X: pd.DataFrame | np.ndarray, 
        y: pd.Series | np.ndarray, 
        synthetic_sample_count: int,
        gmm_component_count: int,
        random_state: int = None):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    elif not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame or a NumPy array")

    X.columns = X.columns.astype(str)

    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    elif not isinstance(y, pd.Series):
        raise ValueError("y must be a pandas Series or a NumPy array")

    # Combine X and y for modeling
    df = X.copy()
    df['Target'] = y.values

    # Separate features and target before scaling
    feature_columns = X.columns
    target_column = 'Target'

    # Pipeline: impute → scale → GMM
    imputer = SimpleImputer()
    scaler = StandardScaler()
    gmm = GaussianMixture(n_components=gmm_component_count, random_state=random_state)

    # Impute and scale features
    X_imputed = imputer.fit_transform(df[feature_columns])
    X_scaled = scaler.fit_transform(X_imputed)

    # Fit GMM on scaled features + target (target unscaled to preserve real distribution)
    df_scaled = pd.DataFrame(X_scaled, columns=feature_columns)
    df_scaled[target_column] = df[target_column].values
    gmm.fit(df_scaled)

    # Generate synthetic data
    synthetic_data, _ = gmm.sample(synthetic_sample_count)
    synthetic_df_scaled = pd.DataFrame(synthetic_data, columns=df_scaled.columns)

    # Inverse scaling on feature columns
    synthetic_features_unscaled = scaler.inverse_transform(synthetic_df_scaled[feature_columns])
    synthetic_df_unscaled = pd.DataFrame(synthetic_features_unscaled, columns=feature_columns)
    synthetic_df_unscaled[target_column] = synthetic_df_scaled[target_column].values

    # Combine original + synthetic
    augmented_df = pd.concat(
        [df, synthetic_df_unscaled],
        ignore_index=True
    )

    # Return as separate X, y
    return augmented_df[feature_columns], augmented_df[target_column]

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
        use_graph_features: bool,
        backbone_sidechain_detail_level: int):
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

    # GET SIDECHAIN & BACKBONE FEATURES.
    if backbone_sidechain_detail_level == 0:
        sidechain_backbone_features = []
    elif backbone_sidechain_detail_level == 1:
        sidechain_backbone_features = extract_sidechain_and_backbone_features(smiles)
        sidechain_backbone_features = [sidechain_backbone_features[name] for name in IMPORTANT_SIDECHAIN_BACKBONE_FEATURE_NAMES]
    elif backbone_sidechain_detail_level == 2:
        sidechain_backbone_features = extract_sidechain_and_backbone_features(smiles)
        sidechain_backbone_features = [sidechain_backbone_features[name] for name in ALL_SIDECHAIN_BACKBONE_FEATURE_NAMES]
    else:
        assert False, f'Invalid backbone vs. sidechain detail level: {backbone_sidechain_detail_level}'

    # CONCATENATE FEATURES.
    features = np.concatenate([
        descriptors, 
        morgan_fingerprint, 
        atom_pair_fingerprint, 
        maccs_keys, 
        torsion_fingerprint,
        graph_features,
        sidechain_backbone_features
    ])
    return features

def get_features_dataframe(
        smiles_df: pd.DataFrame, 
        morgan_fingerprint_dim: int,
        atom_pair_fingerprint_dim: int,
        torsion_dim: int,
        use_maccs_keys: bool,
        use_graph_features: bool,
        backbone_sidechain_detail_level: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    # GET FEATURE NAMES.
    descriptor_names = [descriptor[0] for descriptor in Descriptors._descList]
    morgan_col_names = [f'mfp_{i}' for i in range(morgan_fingerprint_dim)]
    atom_pair_col_names = [f'ap_{i}' for i in range(atom_pair_fingerprint_dim)]
    maccs_col_names = [f'maccs_{i}' for i in range(167)] if use_maccs_keys else []
    torsion_col_names = [f'tt_{i}' for i in range(torsion_dim)]
    graph_col_names = ['graph_diameter', 'avg_shortest_path', 'num_cycles'] if use_graph_features else []
    sidechain_col_names = [[], IMPORTANT_SIDECHAIN_BACKBONE_FEATURE_NAMES, ALL_SIDECHAIN_BACKBONE_FEATURE_NAMES][backbone_sidechain_detail_level]
    feature_col_names = descriptor_names + morgan_col_names + atom_pair_col_names + maccs_col_names + torsion_col_names + graph_col_names + sidechain_col_names

    # GET FEATURES.
    features_df = pd.DataFrame(
        np.vstack([
            get_feature_vector(
                smiles,
                morgan_fingerprint_dim,
                atom_pair_fingerprint_dim,
                torsion_dim,
                use_maccs_keys,
                use_graph_features,
                backbone_sidechain_detail_level
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
        seed: int = 42,
        augmentation_config: dict = None,):
    # PREPROCESS MAIN TRAIN/TEST DATASET DATA.
    filtered_train_test_df = train_test_df.dropna(subset=[target_name])
    train_test_labels = filtered_train_test_df[target_name].reset_index(drop=True)
    
    train_test_features = get_features_dataframe(
        smiles_df=filtered_train_test_df[['SMILES']], 
        **features_config
    )

    print('WARNING: Dropping features!')
    selected_column_names = None
    use_augmentation = (augmentation_config is not None) and (augmentation_config['synthetic_sample_count'] > 0)
    if use_augmentation:
        selector = VarianceThreshold(threshold=0.01)
        selector.fit(train_test_features)
        selected_column_names = train_test_features.columns[selector.get_support()].to_list()
        train_test_features = train_test_features[selected_column_names]
    
    models = []
    imputers = []
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

            print('WARNING: Dropping features!')
            extra_train_features = extra_train_features[train_features.columns]

            train_features = pd.concat([train_features, extra_train_features])
            train_labels = pd.concat([train_labels, extra_train_labels])
            sample_weights.extend(extra_sample_weights)
            # print(f'Added {len(extra_train_labels)} extra samples')

        # MAYBE AUGMENT TRAINING DATA.
        if use_augmentation:
            synthetic_sample_weight = augmentation_config['synthetic_sample_weight']
            synthetic_sample_count = augmentation_config['synthetic_sample_count']
            gmm_component_count = augmentation_config['gmm_component_count']
            train_features, train_labels = augment_dataset(
                train_features, 
                train_labels, 
                synthetic_sample_count=synthetic_sample_count,
                gmm_component_count=gmm_component_count,
            )
            sample_weights.extend([synthetic_sample_weight] * synthetic_sample_count)

        # MAYBE IMPUTE MISSING VALUES.
        imputer = None
        if model_class in [RealMLP_TD_Regressor, TabM_D_Regressor]:
            selection_mask = np.array(sample_weights) > 0.5
            train_features = train_features[selection_mask]
            train_labels = train_labels[selection_mask]

            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            train_features = imputer.fit_transform(train_features)
            test_features = imputer.transform(test_features)

        # TRAIN.
        if model_class in [CatBoostRegressor, LGBMRegressor, XGBRegressor]:
            model = model_class(**model_kwargs)
            model.fit(train_features, train_labels, sample_weight=sample_weights)
        elif model_class in [RealMLP_TD_Regressor, TabM_D_Regressor]:
            model = model_class(**model_kwargs)
            model.fit(train_features, train_labels)
        elif model_class == TabularPredictor:
            dataset = train_features.copy(deep=True)
            dataset['label'] = train_labels.values
            dataset['sample_weight'] = sample_weights
            model = model_class(**model_kwargs)
            # model.fit(dataset, presets='good_quality', time_limit=300, dynamic_stacking=False)
            model.fit(dataset, presets='good_quality', time_limit=300)
            # model.fit(dataset, presets='best_quality', time_limit=2_000, refit_full=False)
        else:
            assert False, f"Unsupported model class: {model_class}"

        models.append(model)
        imputers.append(imputer)

        # TEST.
        predictions = model.predict(test_features)
        oof_predictions[test_indices] = predictions

    mae = mean_absolute_error(train_test_labels, oof_predictions)
    return imputers, models, mae, selected_column_names

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

def add_defaults(model_class, model_kwargs):
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
    elif model_class == RealMLP_TD_Regressor:
        # Remove 'width' and 'layer_count' if present
        model_kwargs.pop("width", None)
        model_kwargs.pop("layer_count", None)
        model_kwargs.setdefault("embedding_size", 8)
        model_kwargs.setdefault("hidden_sizes", [256, 256])
        model_kwargs.setdefault("p_drop", 0.1)
        model_kwargs.setdefault("lr", 0.05)
        model_kwargs.setdefault("n_epochs", 256)
        model_kwargs.setdefault("num_emb_type", "pbld")
        model_kwargs.setdefault("max_one_hot_cat_size", 9)
        model_kwargs.setdefault("weight_param", "ntk")
        model_kwargs.setdefault("weight_init_mode", "std")
        model_kwargs.setdefault("bias_init_mode", "he+5")
        model_kwargs.setdefault("bias_lr_factor", 0.1)
        model_kwargs.setdefault("act", "mish")
        model_kwargs.setdefault("use_parametric_act", True)
        model_kwargs.setdefault("act_lr_factor", 0.1)
        model_kwargs.setdefault("wd", 2e-2)
        model_kwargs.setdefault("wd_sched", "flat_cos")
        model_kwargs.setdefault("bias_wd_factor", 0.0)
        model_kwargs.setdefault("block_str", "w-b-a-d")
        model_kwargs.setdefault("p_drop_sched", "flat_cos")
        model_kwargs.setdefault("add_front_scale", True)
        model_kwargs.setdefault("scale_lr_factor", 6.0)
        model_kwargs.setdefault("tfms", ["one_hot", "median_center", "robust_scale", "smooth_clip", "embedding"])
        model_kwargs.setdefault("plr_sigma", 0.1)
        model_kwargs.setdefault("plr_hidden_1", 16)
        model_kwargs.setdefault("plr_hidden_2", 4)
        model_kwargs.setdefault("plr_lr_factor", 0.1)
        model_kwargs.setdefault("clamp_output", True)
        model_kwargs.setdefault("normalize_output", True)
        model_kwargs.setdefault("lr_sched", "coslog4")
        model_kwargs.setdefault("opt", "adam")
        model_kwargs.setdefault("sq_mom", 0.95)
    elif model_class == TabM_D_Regressor:
        model_kwargs.pop("use_weight_decay", None)
        model_kwargs.setdefault("num_emb_n_bins", 64)
        model_kwargs.setdefault("batch_size", 256)
        model_kwargs.setdefault("lr", 1e-3)
        model_kwargs.setdefault("d_embedding", 16)
        model_kwargs.setdefault("d_block", 512)
        model_kwargs.setdefault("dropout", 0.1)
        model_kwargs.setdefault("gradient_clipping_norm", None)
        model_kwargs.setdefault("num_emb_type", "pbld")
        model_kwargs.setdefault("n_blocks", 3)
        model_kwargs.setdefault("arch_type", "tabm")
        model_kwargs.setdefault("tabm_k", 32)
        model_kwargs.setdefault("weight_decay", 0.0)
        model_kwargs.setdefault("n_epochs", 1_000_000_000)
        model_kwargs.setdefault("patience", 16)
        model_kwargs.setdefault("compile_model", False)
        model_kwargs.setdefault("allow_amp", False)
        model_kwargs.setdefault("tfms", ["quantile_tabr"])

    return model_kwargs

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
            'use_graph_features' : model_config.get("use_graph_features", False),
            'backbone_sidechain_detail_level' : model_config.get("backbone_sidechain_detail_level", 0)
        }
        with open(f'{output_dir}/{target_name}_features_config.json', 'w') as features_config_file:
            json.dump(features_config, features_config_file)

        model_kwargs = {k: v for k, v in model_config.items() if k not in {
            "morgan_fingerprint_dim", "atom_pair_fingerprint_dim", 
            "torsion_dim", "use_maccs_keys", "use_graph_features",
            "backbone_sidechain_detail_level"}}
        
        
        # ADD DEFAULT SETTINGS IF MISSING.
        model_kwargs = add_defaults(model_class, model_kwargs)

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

        use_augmentation = raw_extra_data_config.get('use_augmentation', False)
        if use_augmentation:
            augmentation_config = {
                'synthetic_sample_weight': raw_extra_data_config['synthetic_sample_weight'],
                'synthetic_sample_count': raw_extra_data_config['synthetic_sample_count'],
                'gmm_component_count': raw_extra_data_config['gmm_component_count'],
            }
        else:
            augmentation_config = None
        
        # TRAIN & TEST MODEL.
        imputers, models, mae, selected_column_names = train_test_single_target_models(
            train_test_df=train_test_df,
            extra_data_config=extra_dataset_configs,
            features_config=features_config,
            target_name=target_name,
            model_class=model_class,
            model_kwargs=model_kwargs,
            fold_count=fold_count,
            seed=69,
            augmentation_config=augmentation_config
        )

        # SAVE MODELS & STATS.
        maes.append(mae)

        for fold_index, model in enumerate(models):
            model_filename = f"{target_name}_fold{fold_index}_mae{int(mae * 10000)}.pkl"
            model_filepath = os.path.join(output_dir, model_filename)

            if selected_column_names is not None:
                col_names_filename = f"{target_name}_fold{fold_index}_mae{int(mae * 10000)}_features.json"
                col_names_filepath = os.path.join(output_dir, col_names_filename)
                with open(col_names_filepath, 'w') as col_names_file:
                    json.dump(selected_column_names, col_names_file)

            if model_class != TabularPredictor:
                with open(model_filepath, "wb") as model_file:
                    pickle.dump(model, model_file)

                if len(imputers) > 0 and imputers[0] is not None:
                    imputers_filename = f"{target_name}_fold{fold_index}_mae{int(mae * 10000)}_imputer.pkl"
                    imputers_filepath = os.path.join(output_dir, imputers_filename)
                    with open(imputers_filepath, "wb") as imputers_file:
                        pickle.dump(imputers, imputers_file)
            else:
                model.clone(model_filepath)

        print(f"{target_name}: MAE = {mae:.5f}")

    # LOG wMAE.
    class_weights = get_target_weights(data_path, target_names)
    weighted_mae = np.average(maes, weights=class_weights)
    print(f"Weighted average MAE: {weighted_mae:.5f}")


def get_optimal_model_config(model_class, target_name, trial_count, extra_data_config_path, data_path='data/from_host_v2/train.csv'):
    def objective(trial) -> float:
        # LOAD DATA.
        labeled_smiles_df = pd.read_csv(data_path)
        labeled_smiles_df['SMILES'] = labeled_smiles_df['SMILES'].map(canonicalize_smiles)
        # labeled_smiles_df = labeled_smiles_df.dropna(subset=target_name)

        # LOAD EXTRA DATA CONFIG.
        with open(extra_data_config_path, 'r') as extra_data_config_file:
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

        # COMPUTE FEATURES.
        # features_df = get_features_dataframe(
        #     labeled_smiles_df, 
        #     morgan_fingerprint_dim = trial.suggest_categorical('morgan_fingerprint_dim', [0, 512, 1024, 2048]),
        #     atom_pair_fingerprint_dim = trial.suggest_categorical('atom_pair_fingerprint_dim', [0, 512, 1024, 2048]),
        #     torsion_dim = trial.suggest_categorical('torsion_dim', [0, 512, 1024, 2048]),
        #     use_maccs_keys = trial.suggest_categorical('use_maccs_keys', [True, False]),
        #     use_graph_features = trial.suggest_categorical('use_graph_features', [True, False]),
        #     backbone_sidechain_detail_level = trial.suggest_categorical('backbone_sidechain_detail_level', [0, 1, 2]),
        # )

        features_config = {
            'morgan_fingerprint_dim': trial.suggest_categorical('morgan_fingerprint_dim', [0, 512, 1024, 2048]),
            'atom_pair_fingerprint_dim': trial.suggest_categorical('atom_pair_fingerprint_dim', [0, 512, 1024, 2048]),
            'torsion_dim': trial.suggest_categorical('torsion_dim', [0, 512, 1024, 2048]),
            'use_maccs_keys': trial.suggest_categorical('use_maccs_keys', [True, False]),
            'use_graph_features': trial.suggest_categorical('use_graph_features', [True, False]),
            'backbone_sidechain_detail_level': trial.suggest_categorical('backbone_sidechain_detail_level', [0, 1, 2]),
        }

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
                "iterations": trial.suggest_int("iterations", 100, 1500),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.4, log=True),
                # "iterations": trial.suggest_int("iterations", 100, 3000),
                # "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.4, log=True),
                "depth": trial.suggest_int("depth", 3, 12),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 15.0),
                # "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 25.0),
                "border_count": trial.suggest_int("border_count", 128, 254),
                "loss_function": loss_function,
                "logging_level": "Silent",
                "gpu_ram_part": 0.8
            }
        elif model_class == LGBMRegressor:
            model_kwargs = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 40_000, log=True),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 12),
                "num_leaves": trial.suggest_int("num_leaves", 8, 512, log=True),
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
                'sample_weight': 'sample_weight',
                'weight_evaluation': True,
                'eval_metric': 'mean_absolute_error',
                # 'verbosity': 1,
            }
        elif model_class == RealMLP_TD_Regressor:
            width = trial.suggest_int("width", 64, 1024, log=True)
            layer_count = trial.suggest_int("layer_count", 1, 5)
            model_kwargs = {
                "hidden_sizes": [width]*layer_count,
                "embedding_size": trial.suggest_int("embedding_size", 4, 16, log=True), # TODO: Increase upper bound to 24
                "p_drop": trial.suggest_float("p_drop", 0.1, 0.2),
                "lr": trial.suggest_float("lr", 0.05, 0.8, log=True), # TODO: Reduce lower bound to 0.02
                "n_epochs": trial.suggest_int("n_epochs", 128, 512, log=True), # TODO: Increase upper bound
                "num_emb_type": "pbld", # TODO: Tune this.
                "max_one_hot_cat_size": 9,
                "weight_param": "ntk",
                "weight_init_mode": "std",
                "bias_init_mode": "he+5",
                "bias_lr_factor": 0.1,
                "act": "mish",
                "use_parametric_act": True,
                "act_lr_factor": 0.1,
                "wd": 2e-2,
                "wd_sched": "flat_cos",
                "bias_wd_factor": 0.0,
                "block_str": "w-b-a-d",
                "p_drop_sched": "flat_cos",
                "add_front_scale": True,
                "scale_lr_factor": 6.0,
                "tfms": ["one_hot", "median_center", "robust_scale", "smooth_clip", "embedding"],
                "plr_sigma": 0.1,
                "plr_hidden_1": 16,
                "plr_hidden_2": 4,
                "plr_lr_factor": 0.1,
                "clamp_output": True,
                "normalize_output": True,
                "lr_sched": "coslog4",
                "opt": "adam",
                "sq_mom": 0.95,
            }
            # model_kwargs = {
            #     "hidden_sizes": [width]*layer_count,
            #     "embedding_size": trial.suggest_int("embedding_size", 4, 24, log=True),
            #     "p_drop": trial.suggest_float("p_drop", 0.1, 0.2),
            #     "lr": trial.suggest_float("lr", 0.02, 0.8, log=True),
            #     "n_epochs": trial.suggest_int("n_epochs", 128, 768, log=True),
            #     'num_emb_type': trial.suggest_categorical('num_emb_type', ['none', 'pbld', 'pl', 'plr']),
            #     "max_one_hot_cat_size": 9,
            #     "weight_param": "ntk",
            #     "weight_init_mode": "std",
            #     "bias_init_mode": "he+5",
            #     "bias_lr_factor": 0.1,
            #     "act": "mish",
            #     "use_parametric_act": True,
            #     "act_lr_factor": 0.1,
            #     "wd": 2e-2,
            #     "wd_sched": "flat_cos",
            #     "bias_wd_factor": 0.0,
            #     "block_str": "w-b-a-d",
            #     "p_drop_sched": "flat_cos",
            #     "add_front_scale": True,
            #     "scale_lr_factor": 6.0,
            #     "tfms": ["one_hot", "median_center", "robust_scale", "smooth_clip", "embedding"],
            #     "plr_sigma": 0.1,
            #     "plr_hidden_1": 16,
            #     "plr_hidden_2": 4,
            #     "plr_lr_factor": 0.1,
            #     "clamp_output": True,
            #     "normalize_output": True,
            #     "lr_sched": "coslog4",
            #     "opt": "adam",
            #     "sq_mom": 0.95,
            # }
        elif model_class == TabM_D_Regressor:
            model_kwargs = {
                'num_emb_n_bins': trial.suggest_int('num_emb_n_bins', 32, 64),
                'batch_size': trial.suggest_int('batch_size', 64, 512, log=True),
                'lr': trial.suggest_float('lr', 5e-4, 8e-3, log=True),
                'd_embedding': trial.suggest_int('d_embedding', 8, 32, log=True),
                'd_block': trial.suggest_int("d_block", 128, 1536, log=True),
                'dropout': trial.suggest_float("dropout", 0, 0.2),
                'gradient_clipping_norm': trial.suggest_categorical('gradient_clipping_norm', [None, 1.0]),
                'num_emb_type': trial.suggest_categorical('num_emb_type', ['none', 'pbld', 'pl', 'plr']),
                'n_blocks': trial.suggest_int('n_blocks', 1, 6),
                'arch_type': trial.suggest_categorical('arch_type', ['tabm', 'tabm-mini']),
                'tabm_k': 32,
                'weight_decay': 0.0,
                'n_epochs': 1_000_000_000,
                'patience': 16,
                'compile_model': False,
                'allow_amp': False,
                'tfms': ['quantile_tabr'],
            }
            # use_weight_decay = trial.suggest_categorical('use_weight_decay', [True, False])
            # model_kwargs = {
            #     'num_emb_n_bins': trial.suggest_int('num_emb_n_bins', 24, 80),
            #     'batch_size': trial.suggest_int('batch_size', 64, 512, log=True),
            #     'lr': trial.suggest_float('lr', 2e-4, 1e-2, log=True),
            #     'd_embedding': trial.suggest_int('d_embedding', 6, 32, log=True),
            #     'd_block': trial.suggest_int("d_block", 96, 2048, log=True),
            #     'dropout': trial.suggest_float("dropout", 0, 0.25),
            #     'gradient_clipping_norm': trial.suggest_categorical('gradient_clipping_norm', [None, 1.0]),
            #     'num_emb_type': trial.suggest_categorical('num_emb_type', ['none', 'pbld', 'pl', 'plr']),
            #     'n_blocks': trial.suggest_int('n_blocks', 1, 6),
            #     'arch_type': trial.suggest_categorical('arch_type', ['tabm', 'tabm-mini']),
            #     'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True) if use_weight_decay else 0,
            #     'tabm_k': 32,
            #     'n_epochs': 1_000_000_000,
            #     'patience': 16,
            #     'compile_model': False,
            #     'allow_amp': False,
            #     'tfms': ['quantile_tabr'],
            # }

        # try:
        _, _, mae, _ = train_test_single_target_models(
            train_test_df=labeled_smiles_df,
            extra_data_config=extra_dataset_configs,
            features_config=features_config,
            target_name=target_name,
            model_class=model_class,
            model_kwargs=model_kwargs,
            fold_count=5,
            seed=42,
        )
        # except:
        #     traceback.print_exc()
        #     return 100

        return mae
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=trial_count, show_progress_bar=True)

    print(f"\nBest wMAE: {study.best_value:.5f}")
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
            'use_graph_features' : best_params.get("use_graph_features", False),
            'backbone_sidechain_detail_level' : best_params.get("backbone_sidechain_detail_level", 0)
        }

        use_augmentation = trial.suggest_categorical('use_augmentation', [True, False])
        if use_augmentation:
            augmentation_config = {
                'synthetic_sample_weight': trial.suggest_float('synthetic_sample_weight', 0.1, 1.0),
                'synthetic_sample_count': trial.suggest_int('synthetic_sample_count', 200, 5_000, log=True),
                'gmm_component_count': trial.suggest_int('gmm_component_count', 2, 10),
            }
        else:
            augmentation_config = None

        model_kwargs = {k: v for k, v in best_params.items() if k not in {
            "morgan_fingerprint_dim", "atom_pair_fingerprint_dim", 
            "torsion_dim", "use_maccs_keys", "use_graph_features",
            "backbone_sidechain_detail_level"}}

        model_kwargs = add_defaults(model_class, model_kwargs)

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
        _, _, mae, _ = train_test_single_target_models(
            train_test_df=train_test_df,
            extra_data_config=extra_data_configs,
            features_config=features_config,
            target_name=target_name,
            model_class=model_class,
            model_kwargs=model_kwargs,
            fold_count=5,
            seed=1337,
            augmentation_config=augmentation_config
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
    #         # model_class=XGBRegressor,
    #         # model_config_path={
    #         #     'Tg': 'configs/XGBRegressor_Tg_493667.json',
    #         #     'FFV': 'configs/XGBRegressor_FFV_53.json',
    #         #     'Tc': 'configs/XGBRegressor_Tc_254.json',
    #         #     'Density': 'configs/XGBRegressor_Density_293.json',
    #         #     'Rg': 'configs/XGBRegressor_Rg_14916.json',
    #         # }[target_name],
    #         model_class=LGBMRegressor,
    #         model_config_path={
    #             'Tg': 'configs/LGBMRegressor_Tg_486709.json',
    #             'FFV': 'configs/LGBMRegressor_FFV_49.json',
    #             'Tc': 'configs/LGBMRegressor_Tc_247.json',
    #             'Density': 'configs/LGBMRegressor_Density_296.json',
    #             'Rg': 'configs/LGBMRegressor_Rg_14593.json',
    #         }[target_name],
    #         # model_class=CatBoostRegressor,
    #         # model_config_path={
    #         #     'Tg': 'configs/CatBoostRegressor_Tg_492811.json',
    #         #     'FFV': 'configs/CatBoostRegressor_FFV_58.json',
    #         #     'Tc': 'configs/CatBoostRegressor_Tc_253.json',
    #         #     'Density': 'configs/CatBoostRegressor_Density_310.json',
    #         #     'Rg': 'configs/CatBoostRegressor_Rg_15101.json',
    #         # }[target_name],
    #         # model_class=TabularPredictor,
    #         # model_config_path={
    #         #     'Tg': 'configs/TabularPredictor_Tg_487930.json',
    #         #     'FFV': 'configs/TabularPredictor_FFV_53.json',
    #         #     'Tc': 'configs/TabularPredictor_Tc_253.json',
    #         #     'Density': 'configs/TabularPredictor_Density_290.json',
    #         #     'Rg': 'configs/TabularPredictor_Rg_14917.json',
    #         # }[target_name],
    #         # model_class=RealMLP_TD_Regressor,
    #         # model_config_path={
    #         #     'Tg': 'configs/RealMLP_TD_Regressor_Tg_432262.json',
    #         #     'FFV': 'configs/RealMLP_TD_Regressor_FFV_52.json',
    #         #     'Tc': 'configs/RealMLP_TD_Regressor_Tc_258.json',
    #         #     'Density': 'configs/RealMLP_TD_Regressor_Density_270.json',
    #         #     'Rg': 'configs/RealMLP_TD_Regressor_Rg_13811.json',
    #         # }[target_name],
    #         # model_class=TabM_D_Regressor,
    #         # model_config_path={
    #         #     'Tg': 'configs/TabM_D_Regressor_Tg_433197.json',
    #         #     'FFV': 'configs/TabM_D_Regressor_FFV_50.json',
    #         #     'Tc': 'configs/TabM_D_Regressor_Tc_245.json',
    #         #     'Density': 'configs/TabM_D_Regressor_Density_241.json',
    #         #     'Rg': 'configs/TabM_D_Regressor_Rg_13313.json',
    #         # }[target_name],
    #         trial_count=50,
    #         # trial_count=10,
    #     )

    # TARGET_NAMES= ["Tg", "FFV", "Tc", "Density", "Rg"]
    # # TARGET_NAMES= ["Tc", "Density", "Rg"]
    # # print('WARNING: Only tuning for', TARGET_NAMES)
    # for target_name in TARGET_NAMES:
    #     get_optimal_model_config( 
    #         target_name=target_name, 
    #         # model_class=XGBRegressor,
    #         # extra_data_config_path = {
    #         #     'Tg': 'configs/XGBRegressor_data_Tg_447786.json',
    #         #     'FFV': 'configs/XGBRegressor_data_FFV_51.json',
    #         #     'Tc': 'configs/XGBRegressor_data_Tc_248.json',
    #         #     'Density': 'configs/XGBRegressor_data_Density_254.json',
    #         #     'Rg': 'configs/XGBRegressor_data_Rg_14271.json',
    #         # }[target_name],
    #         # model_class=LGBMRegressor,
    #         # extra_data_config_path={
    #         #     'Tg': 'configs/LGBMRegressor_data_Tg_461983.json',
    #         #     'FFV': 'configs/LGBMRegressor_data_FFV_48.json',
    #         #     'Tc': 'configs/LGBMRegressor_data_Tc_252.json',
    #         #     'Density': 'configs/LGBMRegressor_data_Density_262.json',
    #         #     'Rg': 'configs/LGBMRegressor_data_Rg_13684.json',
    #         # }[target_name],
    #         # model_class=CatBoostRegressor,
    #         # extra_data_config_path = {
    #         #     'Tg': 'configs/CatBoostRegressor_data_Tg_443744.json',
    #         #     'FFV': 'configs/CatBoostRegressor_data_FFV_56.json',
    #         #     'Tc': 'configs/CatBoostRegressor_data_Tc_244.json',
    #         #     'Density': 'configs/CatBoostRegressor_data_Density_276.json',
    #         #     'Rg': 'configs/CatBoostRegressor_data_Rg_14720.json',
    #         # }[target_name],
    #         # model_class=TabularPredictor,
    #         # extra_data_config_path = {
    #         #     'Tg': 'configs/TabularPredictor_data_Tg_440748.json',
    #         #     'FFV': 'configs/TabularPredictor_data_FFV_51.json',
    #         #     'Tc': 'configs/TabularPredictor_data_Tc_249.json',
    #         #     'Density': 'configs/TabularPredictor_data_Density_255.json',
    #         #     'Rg': 'configs/TabularPredictor_data_Rg_14477.json',
    #         # }[target_name],
    #         model_class=RealMLP_TD_Regressor,
    #         extra_data_config_path = {
    #             'Tg': 'configs/avg_data_Tg.json',
    #             'FFV': 'configs/avg_data_FFV.json',
    #             'Tc': 'configs/avg_data_Tc.json',
    #             'Density': 'configs/avg_data_Density.json',
    #             'Rg': 'configs/avg_data_Rg.json',
    #         }[target_name],
    #         # model_class=TabM_D_Regressor,
    #         # extra_data_config_path = {
    #         #     'Tg': 'configs/avg_data_Tg.json',
    #         #     'FFV': 'configs/avg_data_FFV.json',
    #         #     'Tc': 'configs/avg_data_Tc.json',
    #         #     'Density': 'configs/avg_data_Density.json',
    #         #     'Rg': 'configs/avg_data_Rg.json',
    #         # }[target_name],
    #         trial_count=50,
    #         # trial_count=10, # AutoGluon
    #     )

    '''
    No extra features:
        LightGBM (wMAE = 0.06222):
            Tg: MAE = 45.89876
            FFV: MAE = 0.00489
            Tc: MAE = 0.02567
            Density: MAE = 0.02653
            Rg: MAE = 1.36864
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
    Initial side vs backbone features:
        LightGBM (wMAE = 0.06115):
            Tg: MAE = 45.48372
            FFV: MAE = 0.00472
            Tc: MAE = 0.02537
            Density: MAE = 0.02693
            Rg: MAE = 1.30282
        XGBoost (wMAE = 0.06005):
            Tg: MAE = 44.51492
            FFV: MAE = 0.00512
            Tc: MAE = 0.02449
            Density: MAE = 0.02564
            Rg: MAE = 1.31804
        CatBoost: (wMAE = 0.06059)
            Tg: MAE = 44.59141
            FFV: MAE = 0.00570
            Tc: MAE = 0.02451
            Density: MAE = 0.02720
            Rg: MAE = 1.31688
        TabularPredictor: (wMAE = 0.06113)
            Rg: MAE = 1.43479
    Retuned with side vs. backbone features:
        LightGBM (wMAE = 0.05886):
            Tg: MAE = 42.97017
            FFV: MAE = 0.00472
            Tc: MAE = 0.02430
            Density: MAE = 0.02557
            Rg: MAE = 1.30148
        XGBoost (wMAE = 0.05923):
            Tg: MAE = 43.78010
            FFV: MAE = 0.00512
            Tc: MAE = 0.02384
            Density: MAE = 0.02564
            Rg: MAE = 1.31212
        CatBoost: (wMAE = 0.06001)
            Tg: MAE = 44.21214
            FFV: MAE = 0.00570
            Tc: MAE = 0.02423
            Density: MAE = 0.02767
            Rg: MAE = 1.28625
        TabM: (wMAE = 0.06005)
        RealMLP: (wMAE = 0.06194)
    Added data aug:
        LightGBM (wMAE = 0.05872)   <-- Beats baseline
            Tg: MAE = 42.54945      <-- Beats baseline
            FFV: MAE = 0.00479
            Tc: MAE = 0.02508
            Density: MAE = 0.02415  <-- Beats baseline
            Rg: MAE = 1.30224
        XGBoost (wMAE = 0.05944)
            Tg: MAE = 44.41698
            FFV: MAE = 0.00514
            Tc: MAE = 0.02391
            Density: MAE = 0.02534  <-- Beats baseline
            Rg: MAE = 1.30520
        CatBoost (wMAE = 0.06164)
            Tg: MAE = 45.71007
            FFV: MAE = 0.00567      <-- Beats baseline
            Tc: MAE = 0.02465
            Density: MAE = 0.02815
            Rg: MAE = 1.32846
    Tuned data aug:
        XGBoost: (wMAE = 0.05962)
            Tg: MAE = 44.26355
            FFV: MAE = 0.00510
            Tc: MAE = 0.02445
            Density: MAE = 0.02496
            Rg: MAE = 1.31068
        CatBoost (wMAE = 0.06046): 
            Tg: MAE = 43.90599
            FFV: MAE = 0.00554
            Tc: MAE = 0.02414
            Density: MAE = 0.02845
            Rg: MAE = 1.32656

    TabM, partial tune (wMAE = 0.06079):
        Tg: 44.73
        FFV: 0.0050
        Tc: 0.0248
        Density: 0.0347
        Rg: MAE = 1.37482
    RealMLP, partial tune (wMAE = 0.06275):
        Tg: MAE = 44.02621
        FFV: MAE = 0.00541
        Tc: MAE = 0.02632
        Density: MAE = 0.02700
        Rg: MAE = 1.44657
    '''
    #'''
    train_models(
        # model_class=XGBRegressor,
        # target_names_to_model_config_paths={
        #     # 'Tg': 'configs/XGBRegressor_Tg_493667.json',
        #     # 'FFV': 'configs/XGBRegressor_FFV_53.json',
        #     # 'Tc': 'configs/XGBRegressor_Tc_254.json',
        #     # 'Density': 'configs/XGBRegressor_Density_293.json',
        #     # 'Rg': 'configs/XGBRegressor_Rg_14916.json',
        #     'Tg': 'configs/XGBRegressor_Tg_437882.json',
        #     'FFV': 'configs/XGBRegressor_FFV_53_manual.json',
        #     'Tc': 'configs/XGBRegressor_Tc_243.json',
        #     'Density': 'configs/XGBRegressor_Density_293_manual.json',
        #     'Rg': 'configs/XGBRegressor_Rg_12990.json',
        # },
        # target_names_to_extra_data_config_paths = {
        #     # 'Tg': 'configs/XGBRegressor_data_Tg_447786.json',       # Pre data aug - maybe load from 1st backup.
        #     # 'FFV': 'configs/XGBRegressor_data_FFV_51.json',
        #     # 'Tc': 'configs/XGBRegressor_data_Tc_248.json',
        #     # 'Density': 'configs/XGBRegressor_data_Density_254.json',
        #     # 'Rg': 'configs/XGBRegressor_data_Rg_14271.json',
        #     'Tg': 'configs/XGBRegressor_data_Tg_448797.json',         # Augmented
        #     'FFV': 'configs/XGBRegressor_data_FFV_51.json',
        #     'Tc': 'configs/XGBRegressor_data_Tc_248.json',
        #     'Density': 'configs/XGBRegressor_data_Density_256.json',
        #     'Rg': 'configs/XGBRegressor_data_Rg_14350.json',
        # }
        model_class=LGBMRegressor,
        target_names_to_model_config_paths={
            # 'Tg': 'configs/LGBMRegressor_Tg_486709.json',
            # 'FFV': 'configs/LGBMRegressor_FFV_49_no_sidechain.json',
            # 'Tc': 'configs/LGBMRegressor_Tc_247.json',
            # 'Density': 'configs/LGBMRegressor_Density_296.json',
            # 'Rg': 'configs/LGBMRegressor_Rg_14593.json',
            'Tg': 'configs/LGBMRegressor_Tg_429591.json',
            'FFV': 'configs/LGBMRegressor_FFV_47_manual.json',
            'Tc': 'configs/LGBMRegressor_Tc_240.json',
            'Density': 'configs/LGBMRegressor_Density_258.json',
            'Rg': 'configs/LGBMRegressor_Rg_13172.json',
        },
        target_names_to_extra_data_config_paths = {
            # 'Tg': 'configs/LGBMRegressor_data_Tg_461983.json',      # Pre data aug - maybe load from 1st backup.
            # 'FFV': 'configs/LGBMRegressor_data_FFV_48.json',
            # 'Tc': 'configs/LGBMRegressor_data_Tc_252.json',
            # 'Density': 'configs/LGBMRegressor_data_Density_262.json',
            # 'Rg': 'configs/LGBMRegressor_data_Rg_13684.json',
            'Tg': 'configs/LGBMRegressor_data_Tg_466529.json',      # Augmented.
            'FFV': 'configs/LGBMRegressor_data_FFV_48.json',
            'Tc': 'configs/LGBMRegressor_data_Tc_244.json',
            'Density': 'configs/LGBMRegressor_data_Density_263.json',
            'Rg': 'configs/LGBMRegressor_data_Rg_13757.json',
        }
        # model_class=CatBoostRegressor,
        # target_names_to_model_config_paths={
        #     # 'Tg': 'configs/CatBoostRegressor_Tg_492811.json',                 # Baseline
        #     # 'FFV': 'configs/CatBoostRegressor_FFV_58.json',
        #     # 'Tc': 'configs/CatBoostRegressor_Tc_253.json',
        #     # 'Density': 'configs/CatBoostRegressor_Density_310.json',
        #     # 'Rg': 'configs/CatBoostRegressor_Rg_15101.json',
        #     # 'Tg': 'configs/CatBoostRegressor_Tg_492811_manual.json',          # Baseline with manual featureset updates
        #     # 'FFV': 'configs/CatBoostRegressor_FFV_58_manual.json',
        #     # 'Tc': 'configs/CatBoostRegressor_Tc_253_manual.json',
        #     # 'Density': 'configs/CatBoostRegressor_Density_310_manual.json',
        #     # 'Rg': 'configs/CatBoostRegressor_Rg_15101_manual.json',
        #     # 'Tg': 'configs/CatBoostRegressor_Tg_434829.json',                 # Re-tuned
        #     # 'FFV': 'configs/CatBoostRegressor_FFV_59.json',
        #     # 'Tc': 'configs/CatBoostRegressor_Tc_240.json',
        #     # 'Density': 'configs/CatBoostRegressor_Density_286.json',
        #     # 'Rg': 'configs/CatBoostRegressor_Rg_13052.json',
        #     'Tg': 'configs/CatBoostRegressor_Density_310_manual.json',          # Hybrid 
        #     'FFV': 'configs/CatBoostRegressor_FFV_58_manual.json',
        #     'Tc': 'configs/CatBoostRegressor_Tc_240.json',
        #     'Density': 'configs/CatBoostRegressor_Density_286.json',
        #     'Rg': 'configs/CatBoostRegressor_Rg_13052.json',
        # },
        # target_names_to_extra_data_config_paths = {
        #     # 'Tg': 'configs/CatBoostRegressor_data_Tg_443744.json',          # Pre data aug - maybe load from 1st backup.
        #     # 'FFV': 'configs/CatBoostRegressor_data_FFV_56.json',
        #     # 'Tc': 'configs/CatBoostRegressor_data_Tc_244.json',
        #     # 'Density': 'configs/CatBoostRegressor_data_Density_276.json',
        #     # 'Rg': 'configs/CatBoostRegressor_data_Rg_14720.json',
        #     'Tg': 'configs/CatBoostRegressor_data_Tg_435141.json',          # Augmented.
        #     'FFV': 'configs/CatBoostRegressor_data_FFV_56.json',
        #     'Tc': 'configs/CatBoostRegressor_data_Tc_243.json',
        #     'Density': 'configs/CatBoostRegressor_data_Density_286.json',
        #     'Rg': 'configs/CatBoostRegressor_data_Rg_14609.json',
        # }
        # model_class=TabularPredictor,
        # target_names_to_model_config_paths={
        #     # 'Tg': 'configs/TabularPredictor_Tg_487930.json',
        #     # 'FFV': 'configs/TabularPredictor_FFV_53.json',
        #     # 'Tc': 'configs/TabularPredictor_Tc_253.json',
        #     # 'Density': 'configs/TabularPredictor_Density_290.json',
        #     # 'Rg': 'configs/TabularPredictor_Rg_14917.json',
        #     'Tg': 'configs/TabularPredictor_Tg_487930_manual.json',
        #     'FFV': 'configs/TabularPredictor_FFV_53_manual.json',
        #     'Tc': 'configs/TabularPredictor_Tc_253_manual.json',
        #     'Density': 'configs/TabularPredictor_Density_290_manual.json',
        #     'Rg': 'configs/TabularPredictor_Rg_14917_manual.json',
        # },
        # target_names_to_extra_data_config_paths = {
        #     'Tg': 'configs/TabularPredictor_data_Tg_440748.json',
        #     'FFV': 'configs/TabularPredictor_data_FFV_51.json',
        #     'Tc': 'configs/TabularPredictor_data_Tc_249.json',
        #     'Density': 'configs/TabularPredictor_data_Density_255.json',
        #     'Rg': 'configs/TabularPredictor_data_Rg_14477.json',
        # }
        # model_class=RealMLP_TD_Regressor,
        # target_names_to_model_config_paths={
        #     'Tg': 'configs/RealMLP_TD_Regressor_Tg_432262.json',
        #     'FFV': 'configs/RealMLP_TD_Regressor_FFV_52.json',
        #     'Tc': 'configs/RealMLP_TD_Regressor_Tc_258.json',
        #     'Density': 'configs/RealMLP_TD_Regressor_Density_270.json',
        #     'Rg': 'configs/RealMLP_TD_Regressor_Rg_13811.json',
        # },
        # target_names_to_extra_data_config_paths = {
        #     'Tg': 'configs/RealMLP_TD_Regressor_data_Tg_428058.json',
        #     'FFV': 'configs/RealMLP_TD_Regressor_data_FFV_52.json',
        #     'Tc': 'configs/RealMLP_TD_Regressor_data_Tc_252.json',
        #     'Density': 'configs/RealMLP_TD_Regressor_data_Density_245.json',
        #     'Rg': 'configs/RealMLP_TD_Regressor_data_Rg_13772.json',
        # },
        # model_class=TabM_D_Regressor,
        # target_names_to_model_config_paths={
        #     'Tg': 'configs/TabM_D_Regressor_Tg_433197.json',
        #     'FFV': 'configs/TabM_D_Regressor_FFV_50.json',
        #     'Tc': 'configs/TabM_D_Regressor_Tc_245.json',
        #     'Density': 'configs/TabM_D_Regressor_Density_241.json',
        #     'Rg': 'configs/TabM_D_Regressor_Rg_13313.json',
        # },
        # target_names_to_extra_data_config_paths = {
        #     'Tg': 'configs/TabM_D_Regressor_data_Tg_434216.json',
        #     'FFV': 'configs/TabM_D_Regressor_data_FFV_49.json',
        #     'Tc': 'configs/TabM_D_Regressor_data_Tc_237.json',
        #     'Density': 'configs/TabM_D_Regressor_data_Density_230.json',
        #     'Rg': 'configs/TabM_D_Regressor_data_Rg_13361.json',
        # },
    )
    #'''