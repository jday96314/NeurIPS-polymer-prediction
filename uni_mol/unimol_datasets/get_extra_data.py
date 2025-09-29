import polars as pl
import pandas as pd
from sklearn.metrics import mean_absolute_error
from rdkit.Chem import Descriptors, GraphDescriptors, MACCSkeys, rdFingerprintGenerator, AllChem, rdmolops
from rdkit import Chem, DataStructs
from typing import List, Sequence, Callable
import numpy as np
import json
from functools import lru_cache
from tqdm import tqdm
import traceback
import os

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

def load_extra_data(
        extra_data_configs: list[dict],
        host_train_df: pd.DataFrame,
        target_name: str,
        max_similarity: float) -> pd.DataFrame:
    # Sorted to prioritize keeping highest-weight data in event of conflicting labels.
    sorted_extra_data_configs = sorted(
        extra_data_configs,
        key=lambda config: config['dataset_weight'],
        reverse=True
    )

    train_smiles = host_train_df['SMILES'].to_list()

    extra_dataset_dfs = []
    added_smiles = set()
    for extra_data_config in sorted_extra_data_configs:
        # SKIP LOW WEIGHT DATASETS.
        dataset_weight = extra_data_config['dataset_weight']
        if dataset_weight < 0.5:
            continue

        # LOAD EXTRA DATA.
        raw_extra_data_df = pd.read_csv(extra_data_config['filepath'].replace('data_filtering/standardized_datasets', 'data_preprocessing/results'))
        
        # REMOVE DUPLICATES & NEAR-DUPLICATES.
        raw_extra_data_df['SMILES'] = raw_extra_data_df['SMILES'].map(canonicalize_smiles)

        # Avoid duplicates within training set(s).
        raw_extra_data_df = raw_extra_data_df[~raw_extra_data_df['SMILES'].isin(added_smiles)] # Avoid overlap between extra train datasets.       
        if extra_data_config['purge_extra_train_smiles_overlap']:
            raw_extra_data_df = raw_extra_data_df[~raw_extra_data_df['SMILES'].isin(train_smiles)] # Avoid overlap with host train dataset.

        # Avoid (near) duplicates vs. test set.
        extra_train_similarity_scores = np.array(compute_max_tanimoto_per_train(
            train_smiles = raw_extra_data_df['SMILES'].to_list(),
            test_smiles = train_smiles,
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

        # RECORD DATASET.
        # extra_dataset_dfs.append(raw_extra_data_df)
        extra_data_df = pd.DataFrame({
            'SMILES': raw_extra_data_df['SMILES'],
            target_name: labels,
        })
        extra_dataset_dfs.append(extra_data_df)

    all_train_df = pd.concat(extra_dataset_dfs + [host_train_df])

    return all_train_df

def build_smiles_processor(target_name: str) -> Callable[[str], str | None]:
    """
    Return a function that:
    * parses a SMILES,
    * replaces '*' (atomic number 0) with carbon,
    * canonicalises and embeds it,
    * applies optional size filter for FFV,
    * and returns the canonical SMILES *or* None on failure.
    """

    def _process(original_smiles: str) -> str | None:
        try:
            # — 1 parse without sanitising first _______________________________
            molecule = Chem.MolFromSmiles(original_smiles, sanitize=False)
            if molecule is None:  # unparsable string
                print('Yikes 1')
                return None

            # # — 2 replace wildcards with carbon ________________________________
            # for atom in molecule.GetAtoms():
            #     if atom.GetAtomicNum() == 0:
            #         # atom.SetAtomicNum(6)
            #         atom.SetAtomicNum(85)

            # — 3 full sanitisation ___________________________________________
            Chem.SanitizeMol(molecule)

            # — 4 dataset-specific filter _____________________________________
            if (
                target_name in ["FFV", "Density"]
                and molecule.GetNumAtoms(onlyExplicit=False) > 110
            ):
                return None

            # — 5 conformer generation sanity check ___________________________
            # molecule = Chem.AddHs(molecule)
            embed_status: int = AllChem.EmbedMolecule(
                molecule, maxAttempts=10, clearConfs=True
            )
            if embed_status != 0:
                return None

            # — 6 return canonicalised SMILES _________________________________
            return Chem.MolToSmiles(
                molecule, canonical=True, isomericSmiles=True
            )

        except Exception:
            # Anything weird → drop the row
            traceback.print_exc()
            return None

    return _process


if __name__ == '__main__':
    host_train_df = pd.read_csv('data/from_host/train.csv')
    host_train_df['SMILES'] = host_train_df['SMILES'].map(canonicalize_smiles)

    TARGET_NAMES_TO_EXTRA_DATA_CONFIG_PATHS = {
        'Tg': 'configs/TabM_D_Regressor_data_Tg_434216.json',
        'FFV': 'configs/TabM_D_Regressor_data_FFV_49.json',
        'Tc': 'configs/TabM_D_Regressor_data_Tc_237.json',
        'Density': 'configs/TabM_D_Regressor_data_Density_230.json',
        'Rg': 'configs/TabM_D_Regressor_data_Rg_13361.json',
        # 'Tg': 'configs/RealMLP_TD_Regressor_data_Tg_428058.json',
        # 'FFV': 'configs/RealMLP_TD_Regressor_data_FFV_52.json',
        # 'Tc': 'configs/RealMLP_TD_Regressor_data_Tc_252.json',
        # 'Density': 'configs/RealMLP_TD_Regressor_data_Density_245.json',
        # 'Rg': 'configs/RealMLP_TD_Regressor_data_Rg_13772.json',
    }
    for target_name, extra_data_config_path in tqdm(TARGET_NAMES_TO_EXTRA_DATA_CONFIG_PATHS.items()):
        # LOAD EXTRA DATA CONFIGS.
        with open(TARGET_NAMES_TO_EXTRA_DATA_CONFIG_PATHS[target_name], 'r') as extra_data_config_file:
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

        # LOAD EXTRA DATA.
        valid_host_train_mask = host_train_df[target_name].notna() & host_train_df[target_name].notnull()
        target_train_df: pd.DataFrame = load_extra_data(
            extra_data_configs = extra_dataset_configs,
            host_train_df = host_train_df[valid_host_train_mask],
            target_name = target_name,
            max_similarity = .99)
        
        # target_train_df = target_train_df.sample(n=100)
        # print(target_train_df)
        
        # FILTER EXTRA DATA.
        preprocess_and_embed = build_smiles_processor(target_name)
        subset_df: pl.DataFrame = (
            pl.from_pandas(target_train_df)
            # keep only rows with a label for this target
            .drop_nulls(subset=target_name)
            # rename the column to the generic name expected downstream
            .with_columns(pl.col(target_name).alias("TARGET"))
            # convert SMILES → canonicalised; invalid rows become null
            .with_columns(
                pl.col("SMILES")
                .map_elements(preprocess_and_embed, return_dtype=pl.Utf8)
                .alias("SMILES")
            )
            # remove rows that failed preprocessing
            .drop_nulls(subset=["SMILES"])
            # output only what Uni-Mol 2 needs
            .select(["SMILES", "TARGET"])
        )

        print(len(subset_df), len(target_train_df))

        # SAVE EXTRA DATA.
        output_path = f'uni_mol/unimol_datasets/extra_data_TabM/{target_name}.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        subset_df.write_csv(output_path)
        print(f'Saved {len(subset_df)} rows to {output_path}')