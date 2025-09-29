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
except:
    CatBoostRegressor, XGBRegressor = None, None
    print('WARNING: Missing dependencies. Continuing anyway.')
    traceback.print_exc()
try:
    from pytabkit import RealMLP_TD_Regressor, TabM_D_Regressor
except:
    RealMLP_TD_Regressor, TabM_D_Regressor = None, None
    print('WARNING: Missing dependencies. Continuing anyway.')
    traceback.print_exc()
try:
    from autogluon.tabular import TabularDataset, TabularPredictor
except:
    TabularDataset, TabularPredictor = None, None
    print('WARNING: Missing dependencies. Continuing anyway.')
    traceback.print_exc()

from rdkit.ML.Descriptors import MoleculeDescriptors
import networkx as nx
from sentence_transformers import SentenceTransformer
import torch

from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.feature_selection import VarianceThreshold
import glob
import joblib
from joblib import Memory

from side_chain_features import ALL_SIDECHAIN_BACKBONE_FEATURE_NAMES, IMPORTANT_SIDECHAIN_BACKBONE_FEATURE_NAMES, EXTRA_SIDECHAIN_BACKBONE_FEATURE_NAMES, extract_sidechain_and_backbone_features
from gemini_features import compute_inexpensive_features, ALL_GEMINI_FEATURE_NAMES, IMPORTANT_GEMINI_FEATURE_NAMES

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

_FEATURE_VECTOR_CACHE_DIR = os.path.expanduser("~/.cache/feature_vectors")
@Memory(location=_FEATURE_VECTOR_CACHE_DIR, verbose=0).cache
def get_feature_vector(
        smiles: str,
        morgan_fingerprint_dim: int,
        atom_pair_fingerprint_dim: int,
        torsion_dim: int,
        use_maccs_keys: bool,
        use_graph_features: bool,
        backbone_sidechain_detail_level: int,
        use_extra_backbone_sidechain_features: bool,
        gemini_features_detail_level: int):
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
    extra_sidechain_backbone_feature_names = EXTRA_SIDECHAIN_BACKBONE_FEATURE_NAMES if use_extra_backbone_sidechain_features else []
    if (backbone_sidechain_detail_level == 0) and (not use_extra_backbone_sidechain_features):
        sidechain_backbone_features = []
    elif (backbone_sidechain_detail_level == 0) and use_extra_backbone_sidechain_features:
        sidechain_backbone_features = extract_sidechain_and_backbone_features(smiles)
        sidechain_backbone_features = [sidechain_backbone_features[name] for name in extra_sidechain_backbone_feature_names]
    elif backbone_sidechain_detail_level == 1:
        sidechain_backbone_features = extract_sidechain_and_backbone_features(smiles)
        sidechain_backbone_features = [sidechain_backbone_features[name] for name in IMPORTANT_SIDECHAIN_BACKBONE_FEATURE_NAMES + extra_sidechain_backbone_feature_names]
    elif backbone_sidechain_detail_level == 2:
        sidechain_backbone_features = extract_sidechain_and_backbone_features(smiles)
        sidechain_backbone_features = [sidechain_backbone_features[name] for name in ALL_SIDECHAIN_BACKBONE_FEATURE_NAMES + extra_sidechain_backbone_feature_names]
    else:
        assert False, f'Invalid backbone vs. sidechain detail level: {backbone_sidechain_detail_level}'

    # GET GEMINI FEATURES.
    if gemini_features_detail_level == 0:
        gemini_features = []
    elif gemini_features_detail_level == 1:
        gemini_features = compute_inexpensive_features(mol)
        gemini_features = [gemini_features[name] for name in IMPORTANT_GEMINI_FEATURE_NAMES]
    elif gemini_features_detail_level == 2:
        gemini_features = compute_inexpensive_features(mol)
        gemini_features = [gemini_features[name] for name in ALL_GEMINI_FEATURE_NAMES]
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
        sidechain_backbone_features,
        gemini_features
    ])
    return features

def _get_standard_features_dataframe(
        smiles_df: pd.DataFrame, 
        morgan_fingerprint_dim: int,
        atom_pair_fingerprint_dim: int,
        torsion_dim: int,
        use_maccs_keys: bool,
        use_graph_features: bool,
        backbone_sidechain_detail_level: int,
        use_extra_backbone_sidechain_features: bool,
        gemini_features_detail_level: int) -> pd.DataFrame:
    # GET FEATURE NAMES.
    descriptor_names = [descriptor[0] for descriptor in Descriptors._descList]
    morgan_col_names = [f'mfp_{i}' for i in range(morgan_fingerprint_dim)]
    atom_pair_col_names = [f'ap_{i}' for i in range(atom_pair_fingerprint_dim)]
    maccs_col_names = [f'maccs_{i}' for i in range(167)] if use_maccs_keys else []
    torsion_col_names = [f'tt_{i}' for i in range(torsion_dim)]
    graph_col_names = ['graph_diameter', 'avg_shortest_path', 'num_cycles'] if use_graph_features else []
    extra_sidechain_col_names = EXTRA_SIDECHAIN_BACKBONE_FEATURE_NAMES if use_extra_backbone_sidechain_features else []
    sidechain_col_names = [[], IMPORTANT_SIDECHAIN_BACKBONE_FEATURE_NAMES, ALL_SIDECHAIN_BACKBONE_FEATURE_NAMES][backbone_sidechain_detail_level] + extra_sidechain_col_names
    gemini_col_names = [[], IMPORTANT_GEMINI_FEATURE_NAMES, ALL_GEMINI_FEATURE_NAMES][gemini_features_detail_level]
    feature_col_names = descriptor_names + morgan_col_names + atom_pair_col_names + maccs_col_names + torsion_col_names + graph_col_names + sidechain_col_names + gemini_col_names

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
                backbone_sidechain_detail_level,
                use_extra_backbone_sidechain_features,
                gemini_features_detail_level
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

_PRED_CACHE_DIR = os.path.expanduser("~/.cache/predicted_features")
@Memory(location=_PRED_CACHE_DIR, verbose=0).cache
def _get_predicted_features_dataframe(smiles_df: pd.DataFrame, models_directory_path: str) -> pd.DataFrame:
    # LOAD MODELS.
    model_paths = glob.glob(os.path.join(models_directory_path, '*.joblib'))
    model_paths = sorted(model_paths)
    models = [joblib.load(model_path) for model_path in model_paths]

    # LOAD FEATURES CONFIG.
    features_config_path = os.path.join(models_directory_path, 'features_config.json')
    with open(features_config_path, 'r') as features_file:
        features_config = json.load(features_file)

    # COMPUTE INPUT FEATURES.
    input_features_df = _get_standard_features_dataframe(
        smiles_df, 
        **features_config, 
        gemini_features_detail_level=0, 
        use_extra_backbone_sidechain_features=False
    )

    # COMPUTE PREDICTED FEATURES.
    predicted_features_df = pd.DataFrame()
    for model_path, model in zip(model_paths, models):
        predictions = model.predict(input_features_df)
        col_name = os.path.splitext(os.path.basename(model_path))[0]
        predicted_features_df[col_name] = predictions

    return predicted_features_df

def load_backbone_into_sentence_transformer(
        base_model_name_or_path: str,
        finetuned_state_dict_path: str,
        device: str | torch.device = "cpu"
    ) -> SentenceTransformer:
    sentence_transformer_model = SentenceTransformer(base_model_name_or_path, device=str(device))

    finetuned_state_dict = torch.load(finetuned_state_dict_path, map_location="cpu")

    remapped_state_dict = {}
    for parameter_name, tensor in finetuned_state_dict.items():
        # Keep only backbone params; drop custom heads like pooler/output
        if parameter_name.startswith("backbone."):
            stripped_name = parameter_name[len("backbone."):]
            stripped_name = stripped_name.replace('encoder', '0.auto_model.encoder')
            stripped_name = stripped_name.replace('embeddings', '0.auto_model.embeddings')
            stripped_name = stripped_name.replace('word_0.auto_model.embeddings', 'word_embeddings')
            stripped_name = stripped_name.replace('position_0.auto_model.embeddings', 'position_embeddings')
            remapped_state_dict[stripped_name] = tensor
        else:
            # If you happened to save only the HF backbone (no prefix), pass them through
            # and let strict=False ignore anything that doesn't match.
            remapped_state_dict[parameter_name] = tensor

    # Load into the underlying HF model used by SentenceTransformer
    missing_keys, unexpected_keys = sentence_transformer_model.load_state_dict(
        remapped_state_dict,
        strict=False
    )
    if missing_keys:
        print(f"[Info] Missing keys when loading backbone (expected for dropped heads): {len(missing_keys)}")
    if unexpected_keys:
        print(f"[Info] Unexpected keys ignored: {len(unexpected_keys)}")

    sentence_transformer_model.eval()
    return sentence_transformer_model

_POLY_BERT_CACHE_DIR = os.path.expanduser("~/.cache/poly_bert_embeddings")
@Memory(location=_POLY_BERT_CACHE_DIR, verbose=0).cache
def _get_poly_bert_embeddings(smiles_df: pd.DataFrame, polybert_embedding_dim_count: int):
    # polybert = SentenceTransformer('kuelumbus/polyBERT')

    polybert: SentenceTransformer = load_backbone_into_sentence_transformer(
        base_model_name_or_path="kuelumbus/polyBERT",
        finetuned_state_dict_path="models/poly_6epochs_2thresh_v2.1_rankup/single_split/polymer_bert_rankup.pth",
        device="cuda"
    )

    embeddings = polybert.encode(smiles_df['SMILES'].to_list())
    
    embedding_col_names = [f'polyBERT_{index}' for index in range(len(embeddings[0]))]
    features_df = pd.DataFrame(embeddings, columns=embedding_col_names)
    
    # ranked_features = ['polyBERT_360', 'polyBERT_68', 'polyBERT_32', 'polyBERT_207', 'polyBERT_127', 'polyBERT_348', 'polyBERT_189', 'polyBERT_285', 'polyBERT_424', 'polyBERT_415', 'polyBERT_51', 'polyBERT_384', 'polyBERT_210', 'polyBERT_492', 'polyBERT_204', 'polyBERT_410', 'polyBERT_203', 'polyBERT_402', 'polyBERT_50', 'polyBERT_88', 'polyBERT_457', 'polyBERT_584', 'polyBERT_112', 'polyBERT_295', 'polyBERT_544', 'polyBERT_190', 'polyBERT_408', 'polyBERT_338', 'polyBERT_54', 'polyBERT_179', 'polyBERT_246', 'polyBERT_471', 'polyBERT_540', 'polyBERT_280', 'polyBERT_428', 'polyBERT_512', 'polyBERT_82', 'polyBERT_275', 'polyBERT_417', 'polyBERT_154', 'polyBERT_449', 'polyBERT_554', 'polyBERT_74', 'polyBERT_396', 'polyBERT_91', 'polyBERT_208', 'polyBERT_375', 'polyBERT_288', 'polyBERT_201', 'polyBERT_400', 'polyBERT_289', 'polyBERT_134', 'polyBERT_468', 'polyBERT_183', 'polyBERT_228', 'polyBERT_0', 'polyBERT_171', 'polyBERT_436', 'polyBERT_95', 'polyBERT_151', 'polyBERT_297', 'polyBERT_36', 'polyBERT_186', 'polyBERT_316', 'polyBERT_81', 'polyBERT_463', 'polyBERT_302', 'polyBERT_227', 'polyBERT_78', 'polyBERT_168', 'polyBERT_145', 'polyBERT_3', 'polyBERT_170', 'polyBERT_423', 'polyBERT_571', 'polyBERT_301', 'polyBERT_176', 'polyBERT_432', 'polyBERT_60', 'polyBERT_552', 'polyBERT_262', 'polyBERT_300', 'polyBERT_93', 'polyBERT_169', 'polyBERT_191', 'polyBERT_160', 'polyBERT_503', 'polyBERT_380', 'polyBERT_451', 'polyBERT_548', 'polyBERT_57', 'polyBERT_570', 'polyBERT_332', 'polyBERT_441', 'polyBERT_99', 'polyBERT_478', 'polyBERT_477', 'polyBERT_394', 'polyBERT_58', 'polyBERT_308', 'polyBERT_118', 'polyBERT_85', 'polyBERT_101', 'polyBERT_84', 'polyBERT_475', 'polyBERT_440', 'polyBERT_299', 'polyBERT_397', 'polyBERT_325', 'polyBERT_327', 'polyBERT_244', 'polyBERT_1', 'polyBERT_100', 'polyBERT_209', 'polyBERT_343', 'polyBERT_109', 'polyBERT_226', 'polyBERT_21', 'polyBERT_370', 'polyBERT_367', 'polyBERT_23', 'polyBERT_193', 'polyBERT_476', 'polyBERT_369', 'polyBERT_556', 'polyBERT_357', 'polyBERT_335', 'polyBERT_511', 'polyBERT_597', 'polyBERT_494', 'polyBERT_309', 'polyBERT_517', 'polyBERT_562', 'polyBERT_376', 'polyBERT_165', 'polyBERT_336', 'polyBERT_293', 'polyBERT_546', 'polyBERT_158', 'polyBERT_42', 'polyBERT_591', 'polyBERT_218', 'polyBERT_215', 'polyBERT_458', 'polyBERT_383', 'polyBERT_216', 'polyBERT_337', 'polyBERT_253', 'polyBERT_153', 'polyBERT_496', 'polyBERT_328', 'polyBERT_22', 'polyBERT_578', 'polyBERT_319', 'polyBERT_56', 'polyBERT_196', 'polyBERT_433', 'polyBERT_257', 'polyBERT_86', 'polyBERT_486', 'polyBERT_281', 'polyBERT_594', 'polyBERT_497', 'polyBERT_7', 'polyBERT_66', 'polyBERT_15', 'polyBERT_286', 'polyBERT_290', 'polyBERT_37', 'polyBERT_557', 'polyBERT_312', 'polyBERT_141', 'polyBERT_506', 'polyBERT_304', 'polyBERT_598', 'polyBERT_40', 'polyBERT_466', 'polyBERT_10', 'polyBERT_320', 'polyBERT_264', 'polyBERT_500', 'polyBERT_72', 'polyBERT_501', 'polyBERT_76', 'polyBERT_270', 'polyBERT_434', 'polyBERT_374', 'polyBERT_16', 'polyBERT_558', 'polyBERT_550', 'polyBERT_454', 'polyBERT_260', 'polyBERT_177', 'polyBERT_138', 'polyBERT_28', 'polyBERT_2', 'polyBERT_461', 'polyBERT_223', 'polyBERT_499', 'polyBERT_229', 'polyBERT_513', 'polyBERT_233', 'polyBERT_89', 'polyBERT_490', 'polyBERT_19', 'polyBERT_219', 'polyBERT_514', 'polyBERT_135', 'polyBERT_132', 'polyBERT_474', 'polyBERT_483', 'polyBERT_70', 'polyBERT_339', 'polyBERT_491', 'polyBERT_41', 'polyBERT_46', 'polyBERT_493', 'polyBERT_504', 'polyBERT_157', 'polyBERT_391', 'polyBERT_373', 'polyBERT_566', 'polyBERT_102', 'polyBERT_5', 'polyBERT_205', 'polyBERT_47', 'polyBERT_314', 'polyBERT_479', 'polyBERT_65', 'polyBERT_156', 'polyBERT_63', 'polyBERT_581', 'polyBERT_545', 'polyBERT_276', 'polyBERT_291', 'polyBERT_379', 'polyBERT_429', 'polyBERT_108', 'polyBERT_582', 'polyBERT_350', 'polyBERT_352', 'polyBERT_361', 'polyBERT_251', 'polyBERT_49', 'polyBERT_324', 'polyBERT_27', 'polyBERT_247', 'polyBERT_509', 'polyBERT_577', 'polyBERT_284', 'polyBERT_182', 'polyBERT_372', 'polyBERT_206', 'polyBERT_508', 'polyBERT_87', 'polyBERT_242', 'polyBERT_147', 'polyBERT_534', 'polyBERT_188', 'polyBERT_237', 'polyBERT_502', 'polyBERT_273', 'polyBERT_403', 'polyBERT_298', 'polyBERT_307', 'polyBERT_368', 'polyBERT_61', 'polyBERT_267', 'polyBERT_541', 'polyBERT_322', 'polyBERT_580', 'polyBERT_527', 'polyBERT_238', 'polyBERT_358', 'polyBERT_192', 'polyBERT_199', 'polyBERT_555', 'polyBERT_498', 'polyBERT_495', 'polyBERT_94', 'polyBERT_202', 'polyBERT_453', 'polyBERT_442', 'polyBERT_75', 'polyBERT_38', 'polyBERT_55', 'polyBERT_488', 'polyBERT_29', 'polyBERT_425', 'polyBERT_124', 'polyBERT_447', 'polyBERT_366', 'polyBERT_259', 'polyBERT_404', 'polyBERT_387', 'polyBERT_146', 'polyBERT_537', 'polyBERT_221', 'polyBERT_536', 'polyBERT_178', 'polyBERT_437', 'polyBERT_572', 'polyBERT_163', 'polyBERT_287', 'polyBERT_392', 'polyBERT_45', 'polyBERT_531', 'polyBERT_507', 'polyBERT_243', 'polyBERT_465', 'polyBERT_567', 'polyBERT_510', 'polyBERT_266', 'polyBERT_444', 'polyBERT_194', 'polyBERT_388', 'polyBERT_330', 'polyBERT_586', 'polyBERT_9', 'polyBERT_6', 'polyBERT_122', 'polyBERT_313', 'polyBERT_589', 'polyBERT_239', 'polyBERT_344', 'polyBERT_549', 'polyBERT_200', 'polyBERT_418', 'polyBERT_574', 'polyBERT_560', 'polyBERT_155', 'polyBERT_416', 'polyBERT_214', 'polyBERT_363', 'polyBERT_575', 'polyBERT_482', 'polyBERT_174', 'polyBERT_356', 'polyBERT_305', 'polyBERT_181', 'polyBERT_347', 'polyBERT_217', 'polyBERT_371', 'polyBERT_184', 'polyBERT_364', 'polyBERT_282', 'polyBERT_354', 'polyBERT_590', 'polyBERT_44', 'polyBERT_161', 'polyBERT_59', 'polyBERT_144', 'polyBERT_345', 'polyBERT_180', 'polyBERT_470', 'polyBERT_129', 'polyBERT_469', 'polyBERT_526', 'polyBERT_148', 'polyBERT_125', 'polyBERT_113', 'polyBERT_248', 'polyBERT_351', 'polyBERT_92', 'polyBERT_24', 'polyBERT_43', 'polyBERT_473', 'polyBERT_52', 'polyBERT_409', 'polyBERT_349', 'polyBERT_220', 'polyBERT_274', 'polyBERT_98', 'polyBERT_103', 'polyBERT_53', 'polyBERT_587', 'polyBERT_487', 'polyBERT_543', 'polyBERT_412', 'polyBERT_67', 'polyBERT_230', 'polyBERT_321', 'polyBERT_283', 'polyBERT_30', 'polyBERT_136', 'polyBERT_104', 'polyBERT_445', 'polyBERT_419', 'polyBERT_73', 'polyBERT_456', 'polyBERT_123', 'polyBERT_64', 'polyBERT_333', 'polyBERT_386', 'polyBERT_265', 'polyBERT_472', 'polyBERT_164', 'polyBERT_481', 'polyBERT_128', 'polyBERT_489', 'polyBERT_198', 'polyBERT_568', 'polyBERT_406', 'polyBERT_547', 'polyBERT_20', 'polyBERT_389', 'polyBERT_271', 'polyBERT_225', 'polyBERT_167', 'polyBERT_35', 'polyBERT_579', 'polyBERT_551', 'polyBERT_278', 'polyBERT_139', 'polyBERT_272', 'polyBERT_166', 'polyBERT_398', 'polyBERT_133', 'polyBERT_515', 'polyBERT_8', 'polyBERT_83', 'polyBERT_80', 'polyBERT_462', 'polyBERT_378', 'polyBERT_317', 'polyBERT_310', 'polyBERT_268', 'polyBERT_599', 'polyBERT_14', 'polyBERT_279', 'polyBERT_79', 'polyBERT_254', 'polyBERT_411', 'polyBERT_39', 'polyBERT_329', 'polyBERT_185', 'polyBERT_519', 'polyBERT_25', 'polyBERT_107', 'polyBERT_521', 'polyBERT_120', 'polyBERT_435', 'polyBERT_152', 'polyBERT_385', 'polyBERT_197', 'polyBERT_71', 'polyBERT_443', 'polyBERT_353', 'polyBERT_480', 'polyBERT_126', 'polyBERT_342', 'polyBERT_121', 'polyBERT_77', 'polyBERT_459', 'polyBERT_18', 'polyBERT_583', 'polyBERT_359', 'polyBERT_110', 'polyBERT_13', 'polyBERT_323', 'polyBERT_69', 'polyBERT_296', 'polyBERT_306', 'polyBERT_595', 'polyBERT_341', 'polyBERT_255', 'polyBERT_563', 'polyBERT_175', 'polyBERT_505', 'polyBERT_533', 'polyBERT_539', 'polyBERT_245', 'polyBERT_522', 'polyBERT_250', 'polyBERT_431', 'polyBERT_414', 'polyBERT_173', 'polyBERT_224', 'polyBERT_381', 'polyBERT_149', 'polyBERT_455', 'polyBERT_114', 'polyBERT_249', 'polyBERT_421', 'polyBERT_426', 'polyBERT_240', 'polyBERT_365', 'polyBERT_172', 'polyBERT_117', 'polyBERT_318', 'polyBERT_485', 'polyBERT_211', 'polyBERT_467', 'polyBERT_355', 'polyBERT_382', 'polyBERT_464', 'polyBERT_256', 'polyBERT_4', 'polyBERT_460', 'polyBERT_520', 'polyBERT_346', 'polyBERT_390', 'polyBERT_393', 'polyBERT_62', 'polyBERT_576', 'polyBERT_232', 'polyBERT_524', 'polyBERT_115', 'polyBERT_131', 'polyBERT_484', 'polyBERT_334', 'polyBERT_377', 'polyBERT_303', 'polyBERT_535', 'polyBERT_31', 'polyBERT_142', 'polyBERT_222', 'polyBERT_407', 'polyBERT_263', 'polyBERT_315', 'polyBERT_159', 'polyBERT_326', 'polyBERT_592', 'polyBERT_111', 'polyBERT_565', 'polyBERT_162', 'polyBERT_529', 'polyBERT_569', 'polyBERT_340', 'polyBERT_518', 'polyBERT_235', 'polyBERT_97', 'polyBERT_446', 'polyBERT_150', 'polyBERT_399', 'polyBERT_422', 'polyBERT_241', 'polyBERT_516', 'polyBERT_187', 'polyBERT_137', 'polyBERT_195', 'polyBERT_561', 'polyBERT_12', 'polyBERT_538', 'polyBERT_116', 'polyBERT_236', 'polyBERT_252', 'polyBERT_525', 'polyBERT_401', 'polyBERT_564', 'polyBERT_34', 'polyBERT_105', 'polyBERT_413', 'polyBERT_11', 'polyBERT_106', 'polyBERT_234', 'polyBERT_530', 'polyBERT_277', 'polyBERT_405', 'polyBERT_450', 'polyBERT_438', 'polyBERT_213', 'polyBERT_588', 'polyBERT_420', 'polyBERT_212', 'polyBERT_90', 'polyBERT_439', 'polyBERT_528', 'polyBERT_596', 'polyBERT_292', 'polyBERT_258', 'polyBERT_261', 'polyBERT_430', 'polyBERT_311', 'polyBERT_17', 'polyBERT_26', 'polyBERT_96', 'polyBERT_143', 'polyBERT_119', 'polyBERT_130', 'polyBERT_532', 'polyBERT_395', 'polyBERT_542', 'polyBERT_559', 'polyBERT_33', 'polyBERT_593', 'polyBERT_427', 'polyBERT_294', 'polyBERT_48', 'polyBERT_452', 'polyBERT_573', 'polyBERT_585', 'polyBERT_523', 'polyBERT_362', 'polyBERT_553', 'polyBERT_140', 'polyBERT_269', 'polyBERT_448', 'polyBERT_331', 'polyBERT_231']
    ranked_features = ['polyBERT_169', 'polyBERT_308', 'polyBERT_361', 'polyBERT_246', 'polyBERT_56', 'polyBERT_177', 'polyBERT_10', 'polyBERT_434', 'polyBERT_534', 'polyBERT_294', 'polyBERT_93', 'polyBERT_295', 'polyBERT_14', 'polyBERT_52', 'polyBERT_389', 'polyBERT_597', 'polyBERT_406', 'polyBERT_38', 'polyBERT_166', 'polyBERT_544', 'polyBERT_342', 'polyBERT_375', 'polyBERT_32', 'polyBERT_594', 'polyBERT_459', 'polyBERT_218', 'polyBERT_510', 'polyBERT_134', 'polyBERT_242', 'polyBERT_62', 'polyBERT_396', 'polyBERT_102', 'polyBERT_57', 'polyBERT_379', 'polyBERT_157', 'polyBERT_297', 'polyBERT_37', 'polyBERT_27', 'polyBERT_349', 'polyBERT_572', 'polyBERT_141', 'polyBERT_207', 'polyBERT_193', 'polyBERT_309', 'polyBERT_74', 'polyBERT_384', 'polyBERT_554', 'polyBERT_408', 'polyBERT_1', 'polyBERT_525', 'polyBERT_435', 'polyBERT_39', 'polyBERT_370', 'polyBERT_479', 'polyBERT_289', 'polyBERT_383', 'polyBERT_138', 'polyBERT_108', 'polyBERT_142', 'polyBERT_335', 'polyBERT_280', 'polyBERT_198', 'polyBERT_489', 'polyBERT_36', 'polyBERT_549', 'polyBERT_583', 'polyBERT_70', 'polyBERT_352', 'polyBERT_300', 'polyBERT_186', 'polyBERT_423', 'polyBERT_550', 'polyBERT_576', 'polyBERT_372', 'polyBERT_54', 'polyBERT_500', 'polyBERT_590', 'polyBERT_490', 'polyBERT_299', 'polyBERT_587', 'polyBERT_279', 'polyBERT_400', 'polyBERT_471', 'polyBERT_89', 'polyBERT_144', 'polyBERT_348', 'polyBERT_112', 'polyBERT_191', 'polyBERT_578', 'polyBERT_211', 'polyBERT_360', 'polyBERT_514', 'polyBERT_77', 'polyBERT_557', 'polyBERT_382', 'polyBERT_85', 'polyBERT_537', 'polyBERT_437', 'polyBERT_386', 'polyBERT_481', 'polyBERT_555', 'polyBERT_306', 'polyBERT_367', 'polyBERT_577', 'polyBERT_82', 'polyBERT_333', 'polyBERT_478', 'polyBERT_410', 'polyBERT_5', 'polyBERT_446', 'polyBERT_458', 'polyBERT_363', 'polyBERT_124', 'polyBERT_283', 'polyBERT_265', 'polyBERT_165', 'polyBERT_357', 'polyBERT_8', 'polyBERT_447', 'polyBERT_436', 'polyBERT_99', 'polyBERT_499', 'polyBERT_339', 'polyBERT_507', 'polyBERT_12', 'polyBERT_369', 'polyBERT_531', 'polyBERT_78', 'polyBERT_545', 'polyBERT_491', 'polyBERT_201', 'polyBERT_535', 'polyBERT_338', 'polyBERT_120', 'polyBERT_511', 'polyBERT_92', 'polyBERT_237', 'polyBERT_401', 'polyBERT_494', 'polyBERT_546', 'polyBERT_508', 'polyBERT_50', 'polyBERT_84', 'polyBERT_270', 'polyBERT_456', 'polyBERT_189', 'polyBERT_548', 'polyBERT_209', 'polyBERT_581', 'polyBERT_76', 'polyBERT_187', 'polyBERT_170', 'polyBERT_168', 'polyBERT_536', 'polyBERT_203', 'polyBERT_290', 'polyBERT_51', 'polyBERT_276', 'polyBERT_417', 'polyBERT_293', 'polyBERT_517', 'polyBERT_519', 'polyBERT_515', 'polyBERT_183', 'polyBERT_438', 'polyBERT_527', 'polyBERT_298', 'polyBERT_591', 'polyBERT_588', 'polyBERT_398', 'polyBERT_154', 'polyBERT_573', 'polyBERT_516', 'polyBERT_513', 'polyBERT_161', 'polyBERT_278', 'polyBERT_378', 'polyBERT_58', 'polyBERT_466', 'polyBERT_196', 'polyBERT_118', 'polyBERT_560', 'polyBERT_135', 'polyBERT_87', 'polyBERT_133', 'polyBERT_132', 'polyBERT_445', 'polyBERT_197', 'polyBERT_30', 'polyBERT_9', 'polyBERT_172', 'polyBERT_556', 'polyBERT_136', 'polyBERT_521', 'polyBERT_208', 'polyBERT_286', 'polyBERT_488', 'polyBERT_66', 'polyBERT_453', 'polyBERT_558', 'polyBERT_319', 'polyBERT_113', 'polyBERT_397', 'polyBERT_592', 'polyBERT_216', 'polyBERT_244', 'polyBERT_199', 'polyBERT_256', 'polyBERT_288', 'polyBERT_81', 'polyBERT_540', 'polyBERT_41', 'polyBERT_359', 'polyBERT_388', 'polyBERT_310', 'polyBERT_412', 'polyBERT_495', 'polyBERT_409', 'polyBERT_337', 'polyBERT_415', 'polyBERT_473', 'polyBERT_547', 'polyBERT_180', 'polyBERT_114', 'polyBERT_307', 'polyBERT_347', 'polyBERT_247', 'polyBERT_429', 'polyBERT_267', 'polyBERT_392', 'polyBERT_156', 'polyBERT_97', 'polyBERT_407', 'polyBERT_59', 'polyBERT_190', 'polyBERT_264', 'polyBERT_482', 'polyBERT_7', 'polyBERT_579', 'polyBERT_151', 'polyBERT_0', 'polyBERT_230', 'polyBERT_121', 'polyBERT_150', 'polyBERT_3', 'polyBERT_185', 'polyBERT_304', 'polyBERT_158', 'polyBERT_380', 'polyBERT_559', 'polyBERT_416', 'polyBERT_301', 'polyBERT_314', 'polyBERT_275', 'polyBERT_137', 'polyBERT_11', 'polyBERT_483', 'polyBERT_526', 'polyBERT_475', 'polyBERT_440', 'polyBERT_493', 'polyBERT_552', 'polyBERT_351', 'polyBERT_520', 'polyBERT_29', 'polyBERT_60', 'polyBERT_457', 'polyBERT_316', 'polyBERT_75', 'polyBERT_123', 'polyBERT_455', 'polyBERT_470', 'polyBERT_140', 'polyBERT_472', 'polyBERT_599', 'polyBERT_15', 'polyBERT_110', 'polyBERT_582', 'polyBERT_551', 'polyBERT_243', 'polyBERT_228', 'polyBERT_322', 'polyBERT_506', 'polyBERT_404', 'polyBERT_354', 'polyBERT_311', 'polyBERT_331', 'polyBERT_233', 'polyBERT_336', 'polyBERT_217', 'polyBERT_441', 'polyBERT_171', 'polyBERT_19', 'polyBERT_24', 'polyBERT_425', 'polyBERT_376', 'polyBERT_393', 'polyBERT_164', 'polyBERT_566', 'polyBERT_444', 'polyBERT_68', 'polyBERT_250', 'polyBERT_390', 'polyBERT_353', 'polyBERT_542', 'polyBERT_240', 'polyBERT_104', 'polyBERT_223', 'polyBERT_562', 'polyBERT_152', 'polyBERT_426', 'polyBERT_381', 'polyBERT_175', 'polyBERT_45', 'polyBERT_512', 'polyBERT_528', 'polyBERT_127', 'polyBERT_561', 'polyBERT_318', 'polyBERT_23', 'polyBERT_541', 'polyBERT_259', 'polyBERT_63', 'polyBERT_595', 'polyBERT_443', 'polyBERT_277', 'polyBERT_522', 'polyBERT_464', 'polyBERT_115', 'polyBERT_188', 'polyBERT_148', 'polyBERT_538', 'polyBERT_480', 'polyBERT_153', 'polyBERT_461', 'polyBERT_503', 'polyBERT_179', 'polyBERT_586', 'polyBERT_553', 'polyBERT_269', 'polyBERT_257', 'polyBERT_21', 'polyBERT_262', 'polyBERT_368', 'polyBERT_34', 'polyBERT_44', 'polyBERT_106', 'polyBERT_391', 'polyBERT_43', 'polyBERT_424', 'polyBERT_206', 'polyBERT_61', 'polyBERT_529', 'polyBERT_271', 'polyBERT_65', 'polyBERT_28', 'polyBERT_448', 'polyBERT_449', 'polyBERT_111', 'polyBERT_394', 'polyBERT_451', 'polyBERT_46', 'polyBERT_329', 'polyBERT_79', 'polyBERT_80', 'polyBERT_296', 'polyBERT_96', 'polyBERT_73', 'polyBERT_303', 'polyBERT_162', 'polyBERT_450', 'polyBERT_433', 'polyBERT_6', 'polyBERT_284', 'polyBERT_224', 'polyBERT_427', 'polyBERT_419', 'polyBERT_69', 'polyBERT_98', 'polyBERT_465', 'polyBERT_496', 'polyBERT_492', 'polyBERT_248', 'polyBERT_395', 'polyBERT_287', 'polyBERT_219', 'polyBERT_454', 'polyBERT_502', 'polyBERT_413', 'polyBERT_235', 'polyBERT_580', 'polyBERT_202', 'polyBERT_505', 'polyBERT_484', 'polyBERT_252', 'polyBERT_231', 'polyBERT_160', 'polyBERT_477', 'polyBERT_147', 'polyBERT_315', 'polyBERT_260', 'polyBERT_373', 'polyBERT_485', 'polyBERT_374', 'polyBERT_26', 'polyBERT_131', 'polyBERT_371', 'polyBERT_268', 'polyBERT_249', 'polyBERT_255', 'polyBERT_574', 'polyBERT_305', 'polyBERT_101', 'polyBERT_272', 'polyBERT_163', 'polyBERT_320', 'polyBERT_420', 'polyBERT_428', 'polyBERT_463', 'polyBERT_181', 'polyBERT_364', 'polyBERT_332', 'polyBERT_469', 'polyBERT_4', 'polyBERT_501', 'polyBERT_130', 'polyBERT_421', 'polyBERT_377', 'polyBERT_241', 'polyBERT_476', 'polyBERT_596', 'polyBERT_215', 'polyBERT_90', 'polyBERT_254', 'polyBERT_100', 'polyBERT_328', 'polyBERT_414', 'polyBERT_17', 'polyBERT_468', 'polyBERT_355', 'polyBERT_126', 'polyBERT_530', 'polyBERT_86', 'polyBERT_146', 'polyBERT_385', 'polyBERT_589', 'polyBERT_40', 'polyBERT_200', 'polyBERT_486', 'polyBERT_487', 'polyBERT_266', 'polyBERT_107', 'polyBERT_568', 'polyBERT_344', 'polyBERT_285', 'polyBERT_105', 'polyBERT_35', 'polyBERT_226', 'polyBERT_575', 'polyBERT_504', 'polyBERT_462', 'polyBERT_204', 'polyBERT_251', 'polyBERT_323', 'polyBERT_340', 'polyBERT_176', 'polyBERT_598', 'polyBERT_274', 'polyBERT_64', 'polyBERT_291', 'polyBERT_53', 'polyBERT_143', 'polyBERT_302', 'polyBERT_42', 'polyBERT_346', 'polyBERT_593', 'polyBERT_563', 'polyBERT_149', 'polyBERT_225', 'polyBERT_543', 'polyBERT_16', 'polyBERT_345', 'polyBERT_33', 'polyBERT_103', 'polyBERT_192', 'polyBERT_565', 'polyBERT_178', 'polyBERT_245', 'polyBERT_72', 'polyBERT_432', 'polyBERT_365', 'polyBERT_20', 'polyBERT_326', 'polyBERT_411', 'polyBERT_184', 'polyBERT_67', 'polyBERT_174', 'polyBERT_460', 'polyBERT_239', 'polyBERT_116', 'polyBERT_48', 'polyBERT_167', 'polyBERT_564', 'polyBERT_584', 'polyBERT_210', 'polyBERT_117', 'polyBERT_431', 'polyBERT_263', 'polyBERT_418', 'polyBERT_467', 'polyBERT_47', 'polyBERT_273', 'polyBERT_281', 'polyBERT_570', 'polyBERT_524', 'polyBERT_194', 'polyBERT_122', 'polyBERT_220', 'polyBERT_313', 'polyBERT_571', 'polyBERT_139', 'polyBERT_533', 'polyBERT_253', 'polyBERT_405', 'polyBERT_330', 'polyBERT_321', 'polyBERT_94', 'polyBERT_430', 'polyBERT_49', 'polyBERT_327', 'polyBERT_509', 'polyBERT_261', 'polyBERT_523', 'polyBERT_212', 'polyBERT_532', 'polyBERT_155', 'polyBERT_282', 'polyBERT_119', 'polyBERT_442', 'polyBERT_205', 'polyBERT_343', 'polyBERT_238', 'polyBERT_366', 'polyBERT_88', 'polyBERT_221', 'polyBERT_312', 'polyBERT_229', 'polyBERT_439', 'polyBERT_91', 'polyBERT_317', 'polyBERT_129', 'polyBERT_399', 'polyBERT_71', 'polyBERT_325', 'polyBERT_83', 'polyBERT_324', 'polyBERT_227', 'polyBERT_350', 'polyBERT_18', 'polyBERT_95', 'polyBERT_213', 'polyBERT_125', 'polyBERT_128', 'polyBERT_145', 'polyBERT_234', 'polyBERT_195', 'polyBERT_258', 'polyBERT_402', 'polyBERT_539', 'polyBERT_474', 'polyBERT_292', 'polyBERT_109', 'polyBERT_569', 'polyBERT_358', 'polyBERT_341', 'polyBERT_13', 'polyBERT_567', 'polyBERT_22', 'polyBERT_55', 'polyBERT_214', 'polyBERT_497', 'polyBERT_403', 'polyBERT_31', 'polyBERT_362', 'polyBERT_422', 'polyBERT_182', 'polyBERT_334', 'polyBERT_498', 'polyBERT_2', 'polyBERT_232', 'polyBERT_25', 'polyBERT_159', 'polyBERT_585', 'polyBERT_173', 'polyBERT_356', 'polyBERT_452', 'polyBERT_236', 'polyBERT_387', 'polyBERT_518', 'polyBERT_222']
    features_df = features_df[ranked_features[:polybert_embedding_dim_count]]

    return features_df

'''
XGB:
    Baseline:
        Tg: MAE = 43.79204
        FFV: MAE = 0.00512 <-- Best
        Tc: MAE = 0.02383
        Density: MAE = 0.02563
        Rg: MAE = 1.31212
        Weighted average MAE: 0.05923
    + All predicted features:
        Tg: MAE = 43.96195
        FFV: MAE = 0.00525
        Tc: MAE = 0.02382
        Density: MAE = 0.02543
        Rg: MAE = 1.28458
        Weighted average MAE: 0.05896
    + Drop less-important predicted features:
        Tg: MAE = 43.77086
        FFV: MAE = 0.00525
        Tc: MAE = 0.02359 <-- Best
        Density: MAE = 0.02541
        Rg: MAE = 1.28150
        Weighted average MAE: 0.05868
    + Gemini features (all non side-chain/backbone)
        Tg: MAE = 43.52136
        FFV: MAE = 0.00518
        Tc: MAE = 0.02388
        Density: MAE = 0.02403
        Rg: MAE = 1.28961
        Weighted average MAE: 0.05843
    + Drop less important Gemini features
        Tg: MAE = 43.48086
        FFV: MAE = 0.00521
        Tc: MAE = 0.02372
        Density: MAE = 0.02392 <-- Best
        Rg: MAE = 1.27835
        Weighted average MAE: 0.05816
    + Top-10 polyBERT embedding dims (rejected)
        Tg: MAE = 43.46521
        FFV: MAE = 0.00525
        Tc: MAE = 0.02403
        Density: MAE = 0.02470
        Rg: MAE = 1.27492       <--Best
        Weighted average MAE: 0.05850
    + Extra backbone/sidechain features (rejected)
        Tg: MAE = 43.46423
        FFV: MAE = 0.00516
        Tc: MAE = 0.02394
        Density: MAE = 0.02413
        Rg: MAE = 1.28300
        Weighted average MAE: 0.05838
    100 optuna trials:
        Tg: MAE = 43.36325  <-- Best
        FFV: MAE = 0.00541
        Tc: MAE = 0.02472
        Density: MAE = 0.02433
        Rg: MAE = 1.28981
        Weighted average MAE: 0.05893
    Manual tweaks/selection:
        Tg: MAE = 43.36325
        FFV: MAE = 0.00512
        Tc: MAE = 0.02359
        Density: MAE = 0.02392
        Rg: MAE = 1.27492
        Weighted average MAE: 0.05797
LGBM:
    Baseline:
        Tg: MAE = 42.97017
        FFV: MAE = 0.00472  <-- Best
        Tc: MAE = 0.02430
        Density: MAE = 0.02557
        Rg: MAE = 1.30148
        Weighted average MAE: 0.05886
    + All predicted features:
        Tg: MAE = 42.77903
        FFV: MAE = 0.00478
        Tc: MAE = 0.02493
        Density: MAE = 0.02410
        Rg: MAE = 1.29640
        Weighted average MAE: 0.05867
    + Drop less-important predicted features:
        Tg: MAE = 42.78191
        FFV: MAE = 0.00481
        Tc: MAE = 0.02469
        Density: MAE = 0.02417
        Rg: MAE = 1.27833
        Weighted average MAE: 0.05834
    + Gemini features (all non side-chain/backbone)
        Tg: MAE = 42.90845
        FFV: MAE = 0.00473
        Tc: MAE = 0.02462
        Density: MAE = 0.02325
        Rg: MAE = 1.28292
        Weighted average MAE: 0.05817
    + Drop less important Gemini features
        Tg: MAE = 42.83928
        FFV: MAE = 0.00474
        Tc: MAE = 0.02474
        Density: MAE = 0.02312
        Rg: MAE = 1.27987
        Weighted average MAE: 0.05813
    + Top-10 polyBERT embedding dims
        Tg: MAE = 42.99671
        FFV: MAE = 0.00477
        Tc: MAE = 0.02474
        Density: MAE = 0.02319
        Rg: MAE = 1.25998
        Weighted average MAE: 0.05800
    + Extra backbone/sidechain features
        Tg: MAE = 42.99287
        FFV: MAE = 0.00476
        Tc: MAE = 0.02493
        Density: MAE = 0.02310  <-- Best
        Rg: MAE = 1.25270       <-- Best
        Weighted average MAE: 0.05799
    100 optuna trials:
        Tg: MAE = 42.51370  <-- Best
        FFV: MAE = 0.00504
        Tc: MAE = 0.02408   <-- Best
        Density: MAE = 0.02433
        Rg: MAE = 1.26934
        Weighted average MAE: 0.05784
    Manual tweaks/selection:
        Tg: MAE = 42.51370
        FFV: MAE = 0.00473
        Tc: MAE = 0.02408
        Density: MAE = 0.02310
        Rg: MAE = 1.25270
        Weighted average MAE: 0.05726
'''
def get_features_dataframe(
        smiles_df: pd.DataFrame, 
        morgan_fingerprint_dim: int,
        atom_pair_fingerprint_dim: int,
        torsion_dim: int,
        use_maccs_keys: bool,
        use_graph_features: bool,
        backbone_sidechain_detail_level: int,
        use_extra_backbone_sidechain_features: bool = False,
        models_directory_path: str | None = None,
        predicted_features_detail_level: int = 0,
        gemini_features_detail_level: int = 0,
        polybert_embedding_dim_count: int = 0,
        ) -> tuple[pl.DataFrame, pl.DataFrame]:
    # COMPUTE "STANDARD" FEATURES.
    features_df = _get_standard_features_dataframe(
        smiles_df, 
        morgan_fingerprint_dim,
        atom_pair_fingerprint_dim,
        torsion_dim,
        use_maccs_keys,
        use_graph_features,
        backbone_sidechain_detail_level,
        use_extra_backbone_sidechain_features,
        gemini_features_detail_level
    )

    # MAYBE COMPUTE PREDICTED SIMULATION RESULT FEATURES.
    if (models_directory_path is not None) and (predicted_features_detail_level > 0):
        predicted_features_df = _get_predicted_features_dataframe(smiles_df, models_directory_path)

        if predicted_features_detail_level == 1:
            IMPORTANT_FEATURE_NAMES = [
                'xgb_ffv',
                'xgb_density_g_cm3',
                'xgb_homopoly_NPR1',
                'xgb_homopoly_NPR2',
                'xgb_homopoly_Eccentricity',
                'xgb_homopoly_Asphericity',
                'xgb_homopoly_SpherocityIndex',
                'xgb_monomer_NPR2',
                'xgb_monomer_NPR1',
                'xgb_monomer_Asphericity',
                'xgb_lambda1_A2',
                'xgb_monomer_SpherocityIndex',
                'xgb_diffusivity_A2_per_ps',
                'xgb_p10_persistence_length',
                'xgb_homopoly_PBF',
                'xgb_lambda2_A2',
                'xgb_voxel_count_occupied',
                'xgb_homopoly_LabuteASA',
                'xgb_occupied_volume_A3',
                'xgb_box_volume_A3'
            ]
            predicted_features_df = predicted_features_df[IMPORTANT_FEATURE_NAMES]

        features_df = pd.concat([features_df, predicted_features_df], axis=1)

    # MAYBE COMPUTE polyBERT FINGERPRINTS.
    if polybert_embedding_dim_count > 0:
        polybert_embeddings_df = _get_poly_bert_embeddings(smiles_df, polybert_embedding_dim_count)
        features_df = pd.concat([features_df, polybert_embeddings_df], axis=1)

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

    selected_column_names = None
    use_augmentation = (augmentation_config is not None) and (augmentation_config['synthetic_sample_count'] > 0)
    if use_augmentation:
        print('WARNING: Dropping features!')
        selector = VarianceThreshold(threshold=0.01)
        selector.fit(train_test_features)
        selected_column_names = train_test_features.columns[selector.get_support()].to_list()
        train_test_features = train_test_features[selected_column_names]
    
    models = []
    imputers = []
    oof_predictions = np.zeros_like(train_test_labels, dtype=float)
    if fold_count > 1:
        kf = KFold(fold_count, shuffle=True, random_state=seed)
        splits = kf.split(train_test_features)
    else:
        train_indices = list(range(len(train_test_features)))
        splits = [(train_indices, train_indices)]
    for train_indices, test_indices in splits:
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

            if len(extra_train_features.columns) > len(train_features.columns):
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
            # model.fit(dataset, presets='good_quality', time_limit=200, ag_args_fit={'num_gpus': 1})
            # model.fit(dataset, presets='best_quality', time_limit=2_000)
            # model.fit(dataset, presets=['best_quality', 'optimize_for_deployment'], time_limit=7_200, ag_args_fit={'num_gpus': 1})
            model.fit(dataset, presets=['best_quality', 'optimize_for_deployment'], time_limit=6_480, ag_args_fit={'num_gpus': 1}) # Used for final ensemble
            # model.fit(dataset, presets=['best_quality', 'optimize_for_deployment'], time_limit=9_000, ag_args_fit={'num_gpus': 1})
        else:
            assert False, f"Unsupported model class: {model_class}"

        models.append(model)
        imputers.append(imputer)

        # TEST.
        predictions = model.predict(test_features)
        oof_predictions[test_indices] = predictions

    mae = mean_absolute_error(train_test_labels, oof_predictions)
    return imputers, models, mae, selected_column_names, train_test_labels, oof_predictions

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
        raw_extra_data_df = pd.read_csv(extra_data_config['filepath'].replace('data_filtering/standardized_datasets', 'data_preprocessing/results'))
        
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
        data_path: str = 'data/from_host/train.csv'):
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
            'backbone_sidechain_detail_level' : model_config.get("backbone_sidechain_detail_level", 0),
            'use_extra_backbone_sidechain_features' : model_config.get("use_extra_backbone_sidechain_features", False),
            'models_directory_path' : model_config.get("models_directory_path", None),
            'predicted_features_detail_level' : model_config.get("predicted_features_detail_level", 0),
            'gemini_features_detail_level' : model_config.get("gemini_features_detail_level", 0),
            'polybert_embedding_dim_count' : model_config.get("polybert_embedding_dim_count", 0),
        }
        with open(f'{output_dir}/{target_name}_features_config.json', 'w') as features_config_file:
            json.dump(features_config, features_config_file)

        model_kwargs = {k: v for k, v in model_config.items() if k not in {
            "morgan_fingerprint_dim", "atom_pair_fingerprint_dim", 
            "torsion_dim", "use_maccs_keys", "use_graph_features",
            "backbone_sidechain_detail_level", "use_polybert_features",
            "use_extra_backbone_sidechain_features", "models_directory_path",
            "predicted_features_detail_level", "gemini_features_detail_level",
            "polybert_embedding_dim_count",
        }}
        
        
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
        imputers, models, mae, selected_column_names, test_labels, oof_predictions = train_test_single_target_models(
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

        labels_filename = f'{target_name}_labels.pkl'
        labels_filepath = os.path.join(output_dir, labels_filename)
        with open(labels_filepath, 'wb') as test_labels_file:
            pickle.dump(test_labels, test_labels_file)
        
        predictions_filename = f'{target_name}_oof_predictions.pkl'
        predictions_filepath = os.path.join(output_dir, predictions_filename)
        with open(predictions_filepath, 'wb') as oof_predictions_file:
            pickle.dump(oof_predictions, oof_predictions_file)

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
        use_polybert_features = trial.suggest_categorical('use_polybert_features', [True, False])
        features_config = {
            'morgan_fingerprint_dim': trial.suggest_categorical('morgan_fingerprint_dim', [0, 512, 1024, 2048]),
            'atom_pair_fingerprint_dim': trial.suggest_categorical('atom_pair_fingerprint_dim', [0, 512, 1024, 2048]),
            'torsion_dim': trial.suggest_categorical('torsion_dim', [0, 512, 1024, 2048]),
            'use_maccs_keys': trial.suggest_categorical('use_maccs_keys', [True, False]),
            'use_graph_features': trial.suggest_categorical('use_graph_features', [True, False]),
            'backbone_sidechain_detail_level': trial.suggest_categorical('backbone_sidechain_detail_level', [0, 1, 2]),
            'use_extra_backbone_sidechain_features': trial.suggest_categorical('use_extra_backbone_sidechain_features', [True, False]),
            'models_directory_path': trial.suggest_categorical('models_directory_path', ['simulations/models']),
            'predicted_features_detail_level': trial.suggest_categorical('predicted_features_detail_level', [0, 1, 2]),
            'gemini_features_detail_level': trial.suggest_categorical('gemini_features_detail_level', [0, 1, 2]),
            'polybert_embedding_dim_count': trial.suggest_int('polybert_embedding_dim_count', 5, 600, log=True) if use_polybert_features else 0,
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
        _, _, mae, _, _, _ = train_test_single_target_models(
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

    timestamp_text = datetime.now().strftime('%Y%m%d%H%M%S')
    output_filepath = f'configs/{model_class.__name__}_{target_name}_{int(study.best_value * 10000)}_{timestamp_text}.json'
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
                'filepath': trial.suggest_categorical(f'filepath_{dataset_index}', [f'data_preprocessing/results/{target_name}/{data_filename}']),
                'raw_label_weight': trial.suggest_float(f'raw_label_weight_{dataset_index}', low=0, high=1),
                'dataset_weight': trial.suggest_float(f'dataset_weight_{dataset_index}', low=0, high=1),
                'max_error_ratio': trial.suggest_float(f'max_error_ratio_{dataset_index}', low=0.5, high=5),
                'purge_extra_train_smiles_overlap': trial.suggest_categorical(f'purge_extra_train_smiles_overlap_{dataset_index}', [True, False]),
            })

        # TRAIN & TEST.
        _, _, mae, _, _, _ = train_test_single_target_models(
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
    train_models(
        fold_count=1,
        model_class=TabularPredictor,
        target_names_to_model_config_paths={
            'Tg': 'configs/TabularPredictor_Tg_431374_20250909060116_tweaked.json',
            'FFV': 'configs/TabularPredictor_FFV_51_20250909100732_tweaked.json',
            'Tc': 'configs/TabularPredictor_Tc_237_20250909144909_tweaked.json',
            'Density': 'configs/TabularPredictor_Density_229_20250909194135_tweaked.json',
            'Rg': 'configs/TabularPredictor_Rg_13036_20250910001950_tweaked.json',
        },
        target_names_to_extra_data_config_paths = {
            'Tg': 'configs/TabularPredictor_data_Tg_440748.json',
            'FFV': 'configs/TabularPredictor_data_FFV_51.json',
            'Tc': 'configs/TabularPredictor_data_Tc_249.json',
            'Density': 'configs/TabularPredictor_data_Density_255.json',
            'Rg': 'configs/TabularPredictor_data_Rg_14477.json',
        }
    )