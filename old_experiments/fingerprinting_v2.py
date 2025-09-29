import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import math
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence, Tuple
import json

import numpy as np
import polars as pl
from rdkit import Chem
from rdkit.Chem import Descriptors, GraphDescriptors, MACCSkeys, rdFingerprintGenerator
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from tqdm import tqdm
from autogluon.tabular import TabularDataset, TabularPredictor

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import optuna

################################################################################
# ★  CONFIGURATION ★
################################################################################

TARGET_NAMES: List[str] = ["Tg", "FFV", "Tc", "Density", "Rg"]
# TARGET_NAMES: List[str] = ["Tc", "Density", "Rg"]
FINGERPRINT_GROUPS: List[str] = [
    "ecfp4_bits_2048",
    "atom_pair_bits_2048",
    # "torsion_bits_2048",
    "maccs_bits_166",
    "3d_descriptors"
]

MODEL_REGISTRY = {
    "xgboost": XGBRegressor,
    "lightgbm": LGBMRegressor,
    "catboost": CatBoostRegressor,
    "randomforest": RandomForestRegressor,
    "ridge": Ridge,
    "extratrees": ExtraTreesRegressor,
    "autogluon": TabularPredictor
}

################################################################################
# ★  RDKit utilities (fingerprints + descriptors) – cached for speed
################################################################################

MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
ATOMPAIR_GEN = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)
TORSION_GEN = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)

DESC_NAMES: List[str] = [name for name, _ in Descriptors.descList]
DESC_FUNCS = [fn for _n, fn in Descriptors.descList]
DESC_NAME_TO_FUNC = dict(Descriptors.descList)

@lru_cache(maxsize=20_000)
def _smiles_to_mol(smiles: str) -> Chem.Mol | None:
    return Chem.MolFromSmiles(smiles, sanitize=True)

@lru_cache(maxsize=20_000)
def load_3d_descriptors_lookup():
    # descriptors_3d_df = pl.read_csv('data/from_natsume/train_merged_with_3m_120t_descriptors.csv')
    descriptors_3d_df = pl.read_csv('data/from_natsume/train_merged_with_v2_3d_descriptors.csv')

    smiles_to_descriptors = {}
    for row in descriptors_3d_df.iter_rows(named=True):
        smiles = row['SMILES']
        # DESC_3D_NAMES = ['PMI1', 'PMI2', 'PMI3', 'NPR1', 'NPR2', 'RadiusOfGyration', 'InertialShapeFactor', 'Eccentricity', 'Asphericity', 'SpherocityIndex', 'PBF', 'used_random_fallback']
        # descriptors = [row[name] for name in DESC_3D_NAMES]
        
        DESC_3D_NAMES = ['PMI1', 'PMI2', 'PMI3', 'NPR1', 'NPR2', 'RadiusOfGyration', 'InertialShapeFactor', 'Eccentricity', 'Asphericity', 'SpherocityIndex', 'PBF']
        if not row['used_random_fallback']:
            descriptors = [row[name] for name in DESC_3D_NAMES]
        else:
            descriptors = [None] * len(DESC_3D_NAMES)
        smiles_to_descriptors[smiles] = descriptors

    return smiles_to_descriptors

@lru_cache(maxsize=20_000)
def fp_groups(smiles: str) -> Mapping[str, Sequence[int]]:
    """Return three fingerprint bit‑vectors (uint8 lists)."""
    mol = _smiles_to_mol(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return {
        "ecfp4_bits_2048": list(MORGAN_GEN.GetFingerprint(mol)),
        "atom_pair_bits_2048": list(ATOMPAIR_GEN.GetFingerprint(mol)),
        "torsion_bits_2048": list(TORSION_GEN.GetFingerprint(mol)),
        "maccs_bits_166": list(MACCSkeys.GenMACCSKeys(mol)),
        "3d_descriptors": load_3d_descriptors_lookup()[smiles]
    }

@lru_cache(maxsize=20_000)
def all_rdkit_descriptors(smiles: str) -> List[float]:
    mol = _smiles_to_mol(smiles)
    if mol is None:
        return [math.nan] * len(DESC_FUNCS)
    return [fn(mol) for fn in DESC_FUNCS]

def analyse_descriptor_mask() -> Tuple[np.ndarray, List[str]]:
    """Return (boolean mask, valid_names) for the *entire* dataset.

    The mask keeps descriptors that are:
      • finite (no NaN/inf across **all** rows), and
      • non‑constant.
    """
    train_df = pl.read_csv('data/from_host/train.csv', columns=["SMILES"], infer_schema_length=10000)
    smiles_series = train_df.get_column("SMILES")

    rows = [all_rdkit_descriptors(smi) for smi in tqdm(smiles_series, desc="scan‑descriptors")]
    mat = np.asarray(rows, dtype=float)

    # mask out ±inf → treat as NaN for the uniqueness test
    mat[~np.isfinite(mat)] = np.nan

    finite_mask = np.isfinite(mat).all(axis=0)
    range_mask = (mat < 1e12).all(axis=0)
    nonconst_mask = np.nanstd(mat, axis=0) > 0.0
    mask = finite_mask & nonconst_mask & range_mask
    valid = [n for n, keep in zip(DESC_NAMES, mask) if keep]
    print(f"RDKit descriptors kept after global scan: {mask.sum()} / {len(mask)}")
    return mask, valid

def load_features_and_target(
        csv_path: str | Path,
        target_name: str,
        desc_mask: np.ndarray) -> Tuple[np.ndarray, Mapping[str, np.ndarray], np.ndarray]:
    df = (
        pl.read_csv(csv_path, infer_schema_length=10000)
        .drop_nulls(subset=[target_name])
        .select(["SMILES", target_name])
    )
    smiles: list[str] = df["SMILES"].to_list()
    labels: np.ndarray = df[target_name].to_numpy()

    # ------------------------------ fingerprints & descriptors
    fp_ecfp, fp_ap, fp_maccs, fp_torsion, desc_3d_rows, desc_rows = [], [], [], [], [], []
    for smi in smiles:
        fp = fp_groups(smi)
        fp_ecfp.append(fp["ecfp4_bits_2048"])
        fp_ap.append(fp["atom_pair_bits_2048"])
        fp_torsion.append(fp["torsion_bits_2048"])
        fp_maccs.append(fp["maccs_bits_166"])
        desc_3d_rows.append(fp["3d_descriptors"])
        desc_rows.append(all_rdkit_descriptors(smi))

    descriptor_matrix_full = np.array(desc_rows, dtype=float)
    descriptor_matrix_full[~np.isfinite(descriptor_matrix_full)] = np.nan

    # keep only the previously validated columns
    descriptor_matrix = descriptor_matrix_full[:, desc_mask]

    # ---------- simple median imputation (column‑wise) ---------- #
    col_medians = np.nanmedian(descriptor_matrix, axis=0)
    # if an entire column is NaN (rare), fallback to zero
    col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
    # broadcast and fill
    inds = np.where(np.isnan(descriptor_matrix))
    descriptor_matrix[inds] = np.take(col_medians, inds[1])

    # final cast to float32 (reduces RAM & avoids >1e308 overflow)
    descriptor_matrix = descriptor_matrix.astype(np.float32)

    featuresets = {
        "ecfp4_bits_2048": np.asarray(fp_ecfp, dtype=np.uint8),
        "atom_pair_bits_2048": np.asarray(fp_ap, dtype=np.uint8),
        "torsion_bits_2048": np.asarray(fp_torsion, dtype=np.uint8),
        "maccs_bits_166": np.asarray(fp_maccs, dtype=np.uint8),
        "3d_descriptors": np.asarray(desc_3d_rows, dtype=np.float32)
    }
    return labels, featuresets, descriptor_matrix


def create_feature_matrix(
        featureset_names_to_values: Mapping[str, np.ndarray],
        descriptor_matrix: np.ndarray,
        desc_name_to_idx: Mapping[str, int],
        units: Sequence[str]) -> np.ndarray:
    """Stack fingerprint groups and/or individual descriptor columns."""

    cols = []
    for u in units:
        if u in featureset_names_to_values:
            cols.append(featureset_names_to_values[u])
        else:  # descriptor name
            idx = desc_name_to_idx[u]
            cols.append(descriptor_matrix[:, idx : idx + 1])  # keep 2‑D
    return np.hstack(cols)

################################################################################
# ★  K-Fold CV for all targets
################################################################################

def train_test_single_target_models(
        X: np.ndarray, 
        y: np.ndarray, 
        model_name: str | list[str], 
        model_kwargs: dict | list[dict], 
        splits: int = 5) -> float:
    models = []
    oof_predictions = np.zeros_like(y, dtype=float)
    kf = KFold(splits, shuffle=True, random_state=42)
    for tr, va in kf.split(X):
        if type(model_name) is str:
            model_cls = MODEL_REGISTRY[model_name]
            model = model_cls(**model_kwargs)
        else:
            models_list = []
            for model_name, model_kwargs in zip(model_name, model_kwargs):
                model_cls = MODEL_REGISTRY[model_name]
                models_list.append((
                    model_name,
                    model_cls(**model_kwargs)
                ))
            model = VotingRegressor(models_list)

        # model.fit(X[tr], y[tr])
        import pandas as pd
        df = pd.DataFrame(X[tr])
        df['label'] = y[tr]
        train_dataset = TabularDataset(df)
        model.fit(train_dataset)

        models.append(model)

        # oof_predictions[va] = model.predict(X[va])
        test_dataset = TabularDataset(pd.DataFrame(X[va]))
        oof_predictions[va] = model.predict(test_dataset)

    mae = mean_absolute_error(y, oof_predictions)
    return models, mae

def train_test_models(
    csv_path: str | Path,
    selected_featureset_names: Sequence[str],
    model_name: str,
    model_kwargs: dict,
    desc_mask: np.ndarray,
    desc_valid_names: List[str],
    splits: int = 5,
    silent: bool = False) -> float:
    desc_name_to_idx = {n: i for i, n in enumerate(desc_valid_names)}

    targets_to_models, maes, ranges, inv_sqrt_ns = {}, [], [], []
    for target_name in TARGET_NAMES:
        y, featuresets, descriptor_matrix = load_features_and_target(csv_path, target_name, desc_mask)
        X = create_feature_matrix(featuresets, descriptor_matrix, desc_name_to_idx, selected_featureset_names)
        
        models, mae = train_test_single_target_models(X, y, model_name, model_kwargs, splits)
        if not silent:
            print(f" {target_name:8s}: MAE {mae:.5f}")

        maes.append(mae)
        ranges.append(y.max() - y.min())
        inv_sqrt_ns.append(len(y) ** -0.5)

        targets_to_models[target_name] = models

    maes, ranges, inv_sqrt_ns = map(np.asarray, (maes, ranges, inv_sqrt_ns))
    weights = (inv_sqrt_ns * len(maes)) / inv_sqrt_ns.sum() / ranges
    
    targets_to_models['features'] = selected_featureset_names

    return targets_to_models, float(np.average(maes, weights=weights))

################################################################################
# ★  Feature selection
################################################################################

def backward_elimination(
    csv_path: str | Path,
    start_units: List[str],
    model_name: str,
    model_kwargs: dict,
    desc_mask: np.ndarray,
    desc_valid_names: List[str],
    splits: int = 5,
) -> List[str]:
    best_units = start_units.copy()
    _, best_score = train_test_models(
        csv_path,
        best_units,
        model_name,
        model_kwargs,
        desc_mask,
        desc_valid_names,
        splits,
        silent=True,
    )
    print(f"Start wMAE: {best_score:.5f}\n----------")

    while len(best_units) > 1:
        scores = []
        for u in best_units:
            trial = [x for x in best_units if x != u]
            _, score = train_test_models(
                csv_path,
                trial,
                model_name,
                model_kwargs,
                desc_mask,
                desc_valid_names,
                splits,
                silent=True,
            )
            scores.append((u, score))

        u_best, s_best = min(scores, key=lambda t: t[1])
        if s_best < best_score:
            print(f" ✂ drop {u_best}  → wMAE {s_best:.5f}")
            best_units.remove(u_best)
            best_score = s_best
        else:
            break
    print(f"Selected units ({len(best_units)}): {best_units}\nFinal wMAE {best_score:.5f}\n")
    return best_units


################################################################################
# ★  Main ★
################################################################################

def feature_selection_main():
    initial_units = FINGERPRINT_GROUPS + desc_valid_names
    best_units = backward_elimination(
        DATA_FILEPATH,
        initial_units,
        # model_name="xgboost",
        # model_name="lightgbm",
        # model_name="catboost",
        model_name="randomforest",
        model_kwargs={
            # RandomForest:
            "n_estimators": 25,
            "n_jobs": -1

            # # XGBoost
            # "n_estimators": 400,
            # "learning_rate": 0.05,
            # "max_depth": 8,
            # "n_jobs": -1,
            # "random_state": 42,

            # # LightGBM:
            # "verbose": -1

            # # CatBoost:
            # "task_type": "GPU",
            # "iterations": 500,
            # "border_count": 32,
            # "max_depth": 5,
            # "logging_level": "Silent"
        },
        desc_mask=desc_mask,
        desc_valid_names=desc_valid_names,
        splits=5,
    )

def train_all_models():
    selected_featureset_names = FINGERPRINT_GROUPS + desc_valid_names
    targets_to_models, wmae = train_test_models(
        DATA_FILEPATH,
        selected_featureset_names,
        model_name="autogluon",
        model_kwargs={
            'label': 'label',
            'eval_metric': 'mean_absolute_error'
        },
        desc_mask=desc_mask,
        desc_valid_names=desc_valid_names,
        splits=5,
    )
    print(f"Overall wMAE: {wmae:.5f}")
    with open(f'models/fingerprinting_lgbm_tuned_{int(wmae * 10_000)}.pkl', 'wb') as models_file:
        pickle.dump(targets_to_models, models_file)

    # # selected_featureset_names = ['ecfp4_bits_2048', 'atom_pair_bits_2048', 'maccs_bits_166', '3d_descriptors', 'MaxAbsEStateIndex', 'MaxEStateIndex', 'MinAbsEStateIndex', 'MinEStateIndex', 'qed', 'SPS', 'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons', 'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'AvgIpc', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA7', 'SlogP_VSA8', 'TPSA', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'FractionCSP3', 'HeavyAtomCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAmideBonds', 'NumAromaticCarbocycles', 'NumAromaticRings', 'NumBridgeheadAtoms', 'NumHAcceptors', 'NumHeteroatoms', 'NumHeterocycles', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'NumSpiroAtoms', 'NumUnspecifiedAtomStereoCenters', 'Phi', 'RingCount', 'MolLogP', 'MolMR', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_azide', 'fr_azo', 'fr_benzene', 'fr_bicyclic', 'fr_diazo', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea']
    # # selected_featureset_names = ['ecfp4_bits_2048', 'atom_pair_bits_2048', 'maccs_bits_166', 'MaxAbsEStateIndex', 'MaxEStateIndex', 'MinAbsEStateIndex', 'MinEStateIndex', 'qed', 'SPS', 'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons', 'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'AvgIpc', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA7', 'SlogP_VSA8', 'TPSA', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'FractionCSP3', 'HeavyAtomCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAmideBonds', 'NumAromaticCarbocycles', 'NumAromaticRings', 'NumBridgeheadAtoms', 'NumHAcceptors', 'NumHeteroatoms', 'NumHeterocycles', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'NumSpiroAtoms', 'NumUnspecifiedAtomStereoCenters', 'Phi', 'RingCount', 'MolLogP', 'MolMR', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_azide', 'fr_azo', 'fr_benzene', 'fr_bicyclic', 'fr_diazo', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea']
    # selected_featureset_names = FINGERPRINT_GROUPS + desc_valid_names
    # targets_to_models, wmae = train_test_models(
    #     DATA_FILEPATH,
    #     selected_featureset_names,
    #     model_name="lightgbm",
    #     model_kwargs={
    #         # "use_all_features": False,
    #         "n_estimators": 3913,
    #         "learning_rate": 0.01146676390577271,
    #         "max_depth": 8,
    #         "num_leaves": 51,
    #         "min_child_samples": 9,
    #         "subsample": 0.716123709975119,
    #         "colsample_bytree": 0.6971459387092316,
    #         "colsample_node": 0.895414938474575,
    #         "reg_alpha": 0.000254142363826773,
    #         "reg_lambda": 0.0012253814331723285,
    #         "objective": "mae",
    #         "verbose": -1,
    #     },
    #     desc_mask=desc_mask,
    #     desc_valid_names=desc_valid_names,
    #     splits=5,
    # )
    # print(f"Overall wMAE: {wmae:.5f}")
    # with open(f'models/fingerprinting_lgbm_tuned_{int(wmae * 10_000)}.pkl', 'wb') as models_file:
    #     pickle.dump(targets_to_models, models_file)
    
    # # XGB_DROPPED_FEATURES = ["SMR_VSA4", "fr_halogen", "Chi2v", "PEOE_VSA7", "SMR_VSA3", "SlogP_VSA6", "MolWt", "NumAliphaticCarbocycles", "fr_Al_OH", "LabuteASA", "ExactMolWt", "fr_para_hydroxylation", "fr_ketone"]
    # # selected_featureset_names = [name for name in (FINGERPRINT_GROUPS + desc_valid_names) if name not in XGB_DROPPED_FEATURES]
    # # selected_featureset_names = FINGERPRINT_GROUPS + ['3d_descriptors'] + desc_valid_names
    # selected_featureset_names = FINGERPRINT_GROUPS + desc_valid_names
    # targets_to_models, wmae = train_test_models(
    #     DATA_FILEPATH,
    #     selected_featureset_names,
    #     model_name="xgboost",
    #     model_kwargs={
    #         # "use_all_features": True,
    #         "n_estimators": 2565,
    #         "learning_rate": 0.007707324646799016,
    #         "max_depth": 5,
    #         "subsample": 0.5283801713131718,
    #         "colsample_bytree": 0.6558876296734392,
    #         "colsample_bynode": 0.8659796683087206,
    #         "reg_alpha": 0.01597169051838797,
    #         "reg_lambda": 0.16439169131661657,
    #         "objective": "reg:squarederror",
    #         "tree_method": "gpu_hist",
    #         "predictor": "gpu_predictor",
    #     },
    #     desc_mask=desc_mask,
    #     desc_valid_names=desc_valid_names,
    #     splits=5,
    # )
    # print(f"Overall wMAE: {wmae:.5f}")
    # with open(f'models/fingerprinting_xgb_tuned_{int(wmae * 10_000)}.pkl', 'wb') as models_file:
    #     pickle.dump(targets_to_models, models_file)
        
    # # selected_featureset_names = ['ecfp4_bits_2048', 'atom_pair_bits_2048', 'maccs_bits_166', 'MaxAbsEStateIndex', 'MaxEStateIndex', 'MinAbsEStateIndex', 'MinEStateIndex', 'qed', 'SPS', 'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons', 'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'AvgIpc', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'TPSA', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'FractionCSP3', 'HeavyAtomCount', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAmideBonds', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumAtomStereoCenters', 'NumBridgeheadAtoms', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumHeterocycles', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'NumSpiroAtoms', 'NumUnspecifiedAtomStereoCenters', 'Phi', 'RingCount', 'MolLogP', 'MolMR', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_benzene', 'fr_bicyclic', 'fr_diazo', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea']
    # # targets_to_models, wmae = train_test_models(
    # #     DATA_FILEPATH,
    # #     selected_featureset_names,
    # #     model_name="randomforest",
    # #     model_kwargs={
    # #         # "criterion": 'absolute_error',
    # #         "n_jobs": -1
    # #     },
    # #     desc_mask=desc_mask,
    # #     desc_valid_names=desc_valid_names,
    # #     splits=5,
    # # )
    # # print(f"Overall wMAE: {wmae:.5f}")
    # # with open(f'models/fingerprinting_rf_af_{int(wmae * 10_000)}.pkl', 'wb') as models_file:
    # #     pickle.dump(targets_to_models, models_file)
        
    # # selected_featureset_names = ['ecfp4_bits_2048', 'atom_pair_bits_2048', 'maccs_bits_166', 'MaxAbsEStateIndex', 'MaxEStateIndex', 'MinAbsEStateIndex', 'MinEStateIndex', 'qed', 'SPS', 'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons', 'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'AvgIpc', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'TPSA', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'FractionCSP3', 'HeavyAtomCount', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAmideBonds', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumAtomStereoCenters', 'NumBridgeheadAtoms', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumHeterocycles', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'NumSpiroAtoms', 'NumUnspecifiedAtomStereoCenters', 'Phi', 'RingCount', 'MolLogP', 'MolMR', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_benzene', 'fr_bicyclic', 'fr_diazo', 'fr_ester', 'fr_ether', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea']
    # # selected_featureset_names = FINGERPRINT_GROUPS + ['3d_descriptors'] + desc_valid_names
    # selected_featureset_names = FINGERPRINT_GROUPS + desc_valid_names
    # targets_to_models, wmae = train_test_models(
    #     DATA_FILEPATH,
    #     selected_featureset_names,
    #     model_name="catboost",
    #     model_kwargs={
    #         # "use_all_features": True,
    #         "loss_function": "RMSE",
    #         "iterations": 323,
    #         "learning_rate": 0.07919626362326303,
    #         "depth": 6,
    #         "l2_leaf_reg": 2.131833637816589,
    #         "border_count": 181,
    #         "logging_level": "Silent",
    #         "task_type": "GPU",
    #     },
    #     desc_mask=desc_mask,
    #     desc_valid_names=desc_valid_names,
    #     splits=5,
    # )
    # print(f"Overall wMAE: {wmae:.5f}")
    # with open(f'models/fingerprinting_cb_tuned_{int(wmae * 10_000)}.pkl', 'wb') as models_file:
    #     pickle.dump(targets_to_models, models_file)
        
    # # targets_to_models, wmae = train_test_models(
    # #     DATA_FILEPATH,
    # #     selected_featureset_names,
    # #     model_name=[
    # #         'xgboost',
    # #         'lightgbm',
    # #         'catboost',
    # #         'randomforest'
    # #     ],
    # #     model_kwargs=[
    # #         {},
    # #         {
    # #             "verbose": -1,
    # #         },
    # #         {
    # #             "logging_level": "Silent"
    # #         },
    # #         {
    # #             "n_jobs": -1
    # #         }
    # #     ],
    # #     desc_mask=desc_mask,
    # #     desc_valid_names=desc_valid_names,
    # #     splits=5,
    # # )
    # # print(f"Overall wMAE: {wmae:.5f}")
    # # with open(f'models/fingerprinting_ensemble_af_{int(wmae * 10_000)}.pkl', 'wb') as models_file:
    # #     pickle.dump(targets_to_models, models_file)

def hyperparameter_tuning_main(model_type_name: str, trail_count: int):
    assert model_type_name in MODEL_REGISTRY, f"Unsupported model type: {model_type_name}"

    def objective(trial: optuna.Trial) -> float:
        use_all_features = trial.suggest_categorical("use_all_features", [True, False])
        if model_type_name == "xgboost":
            model_kwargs = {
                "tree_method": "gpu_hist",
                "predictor": "gpu_predictor",
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
            if use_all_features:
                selected_featureset_names = FINGERPRINT_GROUPS + desc_valid_names
            else:
                XGB_DROPPED_FEATURES = ["SMR_VSA4", "fr_halogen", "Chi2v", "PEOE_VSA7", "SMR_VSA3", "SlogP_VSA6", "MolWt", "NumAliphaticCarbocycles", "fr_Al_OH", "LabuteASA", "ExactMolWt", "fr_para_hydroxylation", "fr_ketone"]
                selected_featureset_names = [name for name in (FINGERPRINT_GROUPS + desc_valid_names) if name not in XGB_DROPPED_FEATURES]
        
        elif model_type_name == "catboost":
            loss_function = trial.suggest_categorical("loss_function", ["RMSE", "MAE", "Huber"])
            model_kwargs = {
                "task_type": "GPU",
                "iterations": trial.suggest_int("iterations", 100, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.4, log=True),
                "depth": trial.suggest_int("depth", 3, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "border_count": trial.suggest_int("border_count", 128, 254),
                "loss_function": loss_function,
                "logging_level": "Silent",
            }
            if loss_function == "Huber":
                model_kwargs["loss_function"] = f"Huber:delta={trial.suggest_float('huber_delta', 0.5, 5.0)}"

            if use_all_features:
                selected_featureset_names = FINGERPRINT_GROUPS + desc_valid_names
            else:
                selected_featureset_names = ['ecfp4_bits_2048', 'atom_pair_bits_2048', 'maccs_bits_166', 'MaxAbsEStateIndex', 'MaxEStateIndex', 'MinAbsEStateIndex', 'MinEStateIndex', 'qed', 'SPS', 'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons', 'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'AvgIpc', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'TPSA', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'FractionCSP3', 'HeavyAtomCount', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAmideBonds', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumAtomStereoCenters', 'NumBridgeheadAtoms', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumHeterocycles', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'NumSpiroAtoms', 'NumUnspecifiedAtomStereoCenters', 'Phi', 'RingCount', 'MolLogP', 'MolMR', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_benzene', 'fr_bicyclic', 'fr_diazo', 'fr_ester', 'fr_ether', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea']
                
        elif model_type_name == "lightgbm":
            model_kwargs = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 40_000, log=True),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
                # "n_estimators": trial.suggest_int("n_estimators", 50, 20_000, log=True),
                # "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.5, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 12),
                "num_leaves": trial.suggest_int("num_leaves", 8, 512, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "colsample_node": trial.suggest_float("colsample_node", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "objective": trial.suggest_categorical("objective", ["regression", "mae", "huber"]),
                "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
                # "boosting": "dart",
                # "drop_rate": trial.suggest_float("drop_rate", 0, 0.2),
                # "max_drop": trial.suggest_int("max_drop", 25, 75),
                # "skip_drop": trial.suggest_float("skip_drop", 0.25, 0.75),
                "verbose": -1,
            }
            if use_all_features:
                selected_featureset_names = FINGERPRINT_GROUPS + desc_valid_names
            else:
                selected_featureset_names = ['ecfp4_bits_2048', 'atom_pair_bits_2048', 'maccs_bits_166', 'MaxAbsEStateIndex', 'MaxEStateIndex', 'MinAbsEStateIndex', 'MinEStateIndex', 'qed', 'SPS', 'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons', 'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'AvgIpc', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA7', 'SlogP_VSA8', 'TPSA', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'FractionCSP3', 'HeavyAtomCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAmideBonds', 'NumAromaticCarbocycles', 'NumAromaticRings', 'NumBridgeheadAtoms', 'NumHAcceptors', 'NumHeteroatoms', 'NumHeterocycles', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'NumSpiroAtoms', 'NumUnspecifiedAtomStereoCenters', 'Phi', 'RingCount', 'MolLogP', 'MolMR', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_azide', 'fr_azo', 'fr_benzene', 'fr_bicyclic', 'fr_diazo', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea']
                
        elif model_type_name == "randomforest":
            model_kwargs = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "criterion": trial.suggest_categorical("criterion", ["squared_error", "absolute_error"]),
                "n_jobs": -1,
            }
            if use_all_features:
                selected_featureset_names = FINGERPRINT_GROUPS + desc_valid_names
            else:
                selected_featureset_names = ['ecfp4_bits_2048', 'atom_pair_bits_2048', 'maccs_bits_166', 'MaxAbsEStateIndex', 'MaxEStateIndex', 'MinAbsEStateIndex', 'MinEStateIndex', 'qed', 'SPS', 'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons', 'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'AvgIpc', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'TPSA', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'FractionCSP3', 'HeavyAtomCount', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAmideBonds', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumAtomStereoCenters', 'NumBridgeheadAtoms', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumHeterocycles', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'NumSpiroAtoms', 'NumUnspecifiedAtomStereoCenters', 'Phi', 'RingCount', 'MolLogP', 'MolMR', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_benzene', 'fr_bicyclic', 'fr_diazo', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea']
                
        else:
            raise ValueError(f"Unsupported model type: {model_type_name}")

        _, wmae = train_test_models(
            DATA_FILEPATH,
            selected_featureset_names=selected_featureset_names,
            model_name=model_type_name,
            model_kwargs=model_kwargs,
            desc_mask=desc_mask,
            desc_valid_names=desc_valid_names,
            splits=3,
            silent=True,
        )
        return wmae

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=trail_count, show_progress_bar=True)

    print(f"\nBest sMAE: {study.best_value:.5f}")
    print("Params:")
    print(json.dumps(study.best_params, indent=4))

if __name__ == "__main__":
    DATA_FILEPATH = "data/from_host/train.csv"
    # DATA_FILEPATH = "data/from_natsume/train_merged.csv"
    desc_mask, desc_valid_names = analyse_descriptor_mask()

    # feature_selection_main()
    train_all_models()

    # hyperparameter_tuning_main('lightgbm', 100)
    # hyperparameter_tuning_main('catboost', 50)
    # hyperparameter_tuning_main('xgboost', 50)
    # hyperparameter_tuning_main('randomforest', 50)