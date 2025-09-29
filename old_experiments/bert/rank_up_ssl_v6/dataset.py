# dataset_rankup.py
import os, hashlib, random
from functools import lru_cache
from typing import Optional, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# -------------------- Utilities (kept from your code and extended) --------------------

def canonicalize_smiles(smiles: str) -> str | None:
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
    test_fingerprints = [
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

def _apply_augmentation_strategy(smiles: str, strategy: str, n_augmentations: int) -> List[str]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [smiles]
        augmented: List[str] = []

        if strategy == 'enumeration':
            for _ in range(max(1, n_augmentations)):
                enum_smiles = Chem.MolToSmiles(
                    mol, canonical=False, doRandom=True, isomericSmiles=True
                )
                augmented.append(enum_smiles)
        elif strategy == 'kekulize':
            try:
                Chem.Kekulize(mol)
                kek_smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
                augmented.append(kek_smiles)
            except Exception:
                pass
        elif strategy == 'stereo_enum':
            for _ in range(max(1, n_augmentations // 2)):
                Chem.RemoveStereochemistry(mol)
                no_stereo = Chem.MolToSmiles(mol)
                augmented.append(no_stereo)

        return list(set(augmented)) or [smiles]
    except Exception:
        return [smiles]

def augment_smiles_frame(
    df: pd.DataFrame,
    smiles_column: str = 'SMILES',
    augmentation_strategies: List[str] = ['enumeration', 'kekulize', 'stereo_enum'],
    n_augmentations: int = 10,
    preserve_original: bool = True,
    random_seed: Optional[int] = None
) -> pd.DataFrame:
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    augmented_rows: List[dict] = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        original_smiles = row[smiles_column]
        if preserve_original:
            r0 = row.to_dict()
            r0['augmentation_strategy'] = 'original'
            r0['is_augmented'] = False
            r0['original_smiles'] = original_smiles
            augmented_rows.append(r0)

        for strategy in augmentation_strategies:
            for aug_smiles in _apply_augmentation_strategy(original_smiles, strategy, n_augmentations):
                if aug_smiles != original_smiles:
                    r = row.to_dict()
                    r[smiles_column] = aug_smiles
                    r['augmentation_strategy'] = strategy
                    r['is_augmented'] = True
                    r['original_smiles'] = original_smiles
                    augmented_rows.append(r)

    out = pd.DataFrame(augmented_rows).reset_index(drop=True)
    print(f"Original size: {len(df)}, Augmented size: {len(out)}")
    return out

# -------------------- Labeled datasets (unchanged API) --------------------

class SMILESDataset(Dataset):
    def __init__(
        self,
        augmented_smiles: list[str],
        original_smiles: list[str],
        labels: list[float],
        sample_weights: list[float],
        tokenizer,
        max_length: int = 256
    ):
        self.augmented_smiles = augmented_smiles
        self.original_smiles = original_smiles
        self.labels = labels
        self.sample_weights = sample_weights
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.augmented_smiles)

    def __getitem__(self, idx: int) -> dict:
        augmented_smiles = self.tokenizer.cls_token + self.augmented_smiles[idx]
        original_smiles = self.original_smiles[idx]
        label_value = self.labels[idx]
        sample_weight_value = self.sample_weights[idx]

        encoding = self.tokenizer(
            augmented_smiles,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        blake2b_hasher = hashlib.blake2b(digest_size=8)
        blake2b_hasher.update(original_smiles.encode('utf-8'))
        original_smiles_id = int.from_bytes(blake2b_hasher.digest(), byteorder='big', signed=True)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_value, dtype=torch.float32),
            'original_smiles_id': original_smiles_id,
            'sample_weight': sample_weight_value
        }

# -------------------- Unlabeled dataset for RankUp (weak/strong views) --------------------

class UnlabeledPairDataset(Dataset):
    """
    Returns weak/strong augmented tokenizations for each SMILES.
    Pairing is formed inside the training loop (to allow sub-sampling).
    """
    def __init__(
        self,
        smiles_list: list[str],
        tokenizer,
        weak_strategies: List[str] = ['enumeration'],
        strong_strategies: List[str] = ['enumeration', 'kekulize', 'stereo_enum'],
        max_length: int = 256
    ):
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.weak_strategies = weak_strategies
        self.strong_strategies = strong_strategies
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.smiles_list)

    def _augment_once(self, smiles: str, strategies: List[str]) -> str:
        strategy = random.choice(strategies) if strategies else 'enumeration'
        variants = _apply_augmentation_strategy(smiles, strategy, n_augmentations=1)
        return variants[0] if variants else smiles

    def _tokenize(self, smiles: str) -> dict:
        smiles_with_cls = self.tokenizer.cls_token + smiles
        enc = self.tokenizer(
            smiles_with_cls,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
        }

    def __getitem__(self, idx: int) -> dict:
        base_smiles = self.smiles_list[idx]
        weak_smiles = self._augment_once(base_smiles, self.weak_strategies)
        strong_smiles = self._augment_once(base_smiles, self.strong_strategies)

        weak_tokens = self._tokenize(weak_smiles)
        strong_tokens = self._tokenize(strong_smiles)

        return {
            'weak_input_ids': weak_tokens['input_ids'],
            'weak_attention_mask': weak_tokens['attention_mask'],
            'strong_input_ids': strong_tokens['input_ids'],
            'strong_attention_mask': strong_tokens['attention_mask'],
        }

# -------------------- Builders --------------------

def load_extra_data(
    extra_data_configs: list[dict],
    target_name: str,
    train_smiles: list[str],
    test_smiles: list[str],
    max_similarity: float
) -> pd.DataFrame:
    added_smiles: list[str] = []
    extra_labels: list[float] = []
    extra_sample_weights: list[float] = []

    sorted_cfgs = sorted(extra_data_configs, key=lambda c: c['dataset_weight'], reverse=True)
    for cfg in sorted_cfgs:
        df = pd.read_csv(cfg['filepath'])
        df['SMILES'] = df['SMILES'].map(canonicalize_smiles)

        df = df[~df['SMILES'].isin(added_smiles)]
        if cfg['purge_extra_train_smiles_overlap']:
            df = df[~df['SMILES'].isin(train_smiles)]
        df = df[~df['SMILES'].isin(test_smiles)]

        if len(df) == 0:
            continue

        # Label merging as in your original extra-data logic
        raw_col = f'{target_name}_label'
        scaled_col = f'{target_name}_rescaled_label'
        pred_col = f'{target_name}_pred'

        if raw_col not in df.columns and scaled_col not in df.columns:
            # No labels in this file; skip for labeled path (it will be used via unlabeled pool below)
            continue

        # Remove (near) dup vs test by Tanimoto
        sims = compute_max_tanimoto_per_train(
            train_smiles=df['SMILES'].tolist(),
            test_smiles=test_smiles
        )
        df = df[sims < max_similarity]

        if raw_col in df.columns and scaled_col in df.columns:
            raw_labels = df[raw_col]
            scaled_labels = df[scaled_col]
            labels = (raw_labels * cfg['raw_label_weight']) + (scaled_labels * (1 - cfg['raw_label_weight']))
        elif raw_col in df.columns:
            labels = df[raw_col]
        elif scaled_col in df.columns:
            labels = df[scaled_col]
        else:
            continue

        if pred_col in df.columns and labels.notna().any():
            mae_value = mean_absolute_error(labels.dropna(), df.loc[labels.notna(), pred_col])
            abs_err = (labels - df[pred_col]).abs()
            mae_ratios = abs_err / max(mae_value, 1e-8)
            keep_mask = mae_ratios < cfg['max_error_ratio']
            df = df[keep_mask]
            labels = labels[keep_mask]

        added_smiles.extend(df['SMILES'].tolist())
        extra_labels.extend(labels.tolist())
        extra_sample_weights.extend([cfg['dataset_weight']] * len(df))

    return pd.DataFrame({
        'SMILES': added_smiles,
        target_name: extra_labels,
        'sample_weight': extra_sample_weights
    })

def build_unlabeled_pool(
    target_name: str,
    host_df: pd.DataFrame,
    extra_data_configs: list[dict],
    unlabeled_smiles_csv_path: Optional[str],
    exclude_smiles: set[str],
) -> list[str]:
    smiles_set: set[str] = set()

    # 1) Supplemental unlabeled CSV
    if unlabeled_smiles_csv_path:
        u = pd.read_csv(unlabeled_smiles_csv_path)
        if 'SMILES' in u.columns:
            u['SMILES'] = u['SMILES'].map(canonicalize_smiles)
            smiles_set.update(s for s in u['SMILES'].dropna().tolist())

    # 2) Host data rows with NaN label for this target
    host_nan = host_df[host_df[target_name].isna()].copy()
    if 'SMILES' in host_nan.columns:
        host_nan['SMILES'] = host_nan['SMILES'].map(canonicalize_smiles)
        smiles_set.update(s for s in host_nan['SMILES'].dropna().tolist())

    # 3) Extra datasets rows with NaN label for this target
    for cfg in extra_data_configs or []:
        df = pd.read_csv(cfg['filepath'])
        if 'SMILES' not in df.columns:
            continue
        df['SMILES'] = df['SMILES'].map(canonicalize_smiles)

        label_col = f'{target_name}_label'
        scaled_col = f'{target_name}_rescaled_label'
        if label_col in df.columns:
            nan_mask = df[label_col].isna()
        elif scaled_col in df.columns:
            nan_mask = df[scaled_col].isna()
        else:
            # If dataset doesn’t even contain target labels, consider all rows unlabeled for this target
            nan_mask = pd.Series([True] * len(df))

        unl = df[nan_mask]
        smiles_set.update(s for s in unl['SMILES'].dropna().tolist())

    # Exclusions (e.g., evaluation/test split)
    smiles_list = [s for s in smiles_set if s and (s not in exclude_smiles)]
    return smiles_list

def get_train_test_datasets(
    csv_path: str,
    extra_dataset_configs: list[dict],
    model_path: str,
    target_name: str,
    fold_count: int,
    fold_id: int,
    train_augmentation_kwargs: dict,
    test_augmentation_kwargs: dict,
):
    # LOAD & FILTER LABELED DATA
    raw_df = pd.read_csv(csv_path)
    filtered_df = raw_df[raw_df[target_name].notna()].copy()

    # SPLIT
    kf = KFold(fold_count, shuffle=True, random_state=42)
    splits = list(kf.split(filtered_df))
    train_idx, test_idx = splits[fold_id]
    raw_train_df = filtered_df.iloc[train_idx].copy()
    test_df = filtered_df.iloc[test_idx].copy()

    # EXTRA LABELED
    extra_df = load_extra_data(
        extra_data_configs=extra_dataset_configs,
        target_name=target_name,
        train_smiles=raw_train_df['SMILES'].to_list(),
        test_smiles=test_df['SMILES'].to_list(),
        max_similarity=0.99
    )

    raw_train_df['sample_weight'] = 1.0
    test_df['sample_weight'] = 1.0

    raw_train_df = pd.concat(
        [raw_train_df[['SMILES', target_name, 'sample_weight']], extra_df],
        ignore_index=True
    )

    # SCALE
    scaler = StandardScaler()
    raw_train_df[target_name] = scaler.fit_transform(raw_train_df[target_name].values.reshape(-1, 1))
    test_df[target_name] = scaler.transform(test_df[target_name].values.reshape(-1, 1))

    # AUGMENT
    aug_train = augment_smiles_frame(raw_train_df, **train_augmentation_kwargs)
    aug_test = augment_smiles_frame(test_df, **test_augmentation_kwargs)

    # DATASETS
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    train_ds = SMILESDataset(
        augmented_smiles=aug_train['SMILES'].tolist(),
        original_smiles=aug_train['original_smiles'].tolist(),
        labels=aug_train[target_name].tolist(),
        sample_weights=aug_train['sample_weight'].astype(float).tolist(),
        tokenizer=tokenizer
    )
    test_ds = SMILESDataset(
        augmented_smiles=aug_test['SMILES'].tolist(),
        original_smiles=aug_test['original_smiles'].tolist(),
        labels=aug_test[target_name].tolist(),
        sample_weights=aug_test['sample_weight'].astype(float).tolist(),
        tokenizer=tokenizer
    )
    return train_ds, test_ds, scaler

def get_train_test_and_unlabeled_datasets(
    csv_path: str,
    extra_data_config: list[dict],
    model_path: str,
    target_name: str,
    fold_count: int,
    fold_id: int,
    train_augmentation_kwargs: dict,
    test_augmentation_kwargs: dict,
    unlabeled_smiles_csv_path: Optional[str] = None,
    weak_strategies: List[str] = ['enumeration'],
    strong_strategies: List[str] = ['enumeration', 'kekulize', 'stereo_enum'],
):
    extra_dataset_configs = []
    extra_dataset_count = int(len(extra_data_config.keys()) / 5)
    for extra_dataset_index in range(extra_dataset_count):
        extra_dataset_configs.append({
            'filepath': extra_data_config[f'filepath_{extra_dataset_index}'],
            'raw_label_weight': extra_data_config[f'raw_label_weight_{extra_dataset_index}'],
            'dataset_weight': extra_data_config[f'dataset_weight_{extra_dataset_index}'],
            'max_error_ratio': extra_data_config[f'max_error_ratio_{extra_dataset_index}'],
            'purge_extra_train_smiles_overlap': extra_data_config[f'purge_extra_train_smiles_overlap_{extra_dataset_index}'],
        })

    train_ds, test_ds, scaler = get_train_test_datasets(
        csv_path=csv_path,
        extra_dataset_configs=extra_dataset_configs,
        model_path=model_path,
        target_name=target_name,
        fold_count=fold_count,
        fold_id=fold_id,
        train_augmentation_kwargs=train_augmentation_kwargs,
        test_augmentation_kwargs=test_augmentation_kwargs,
    )

    # Build unlabeled pool
    host_df = pd.read_csv(csv_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # # Exclude eval SMILES to avoid transductive leakage to ARC (optional but safer)
    # exclude = set(pd.read_csv(csv_path)['SMILES'].iloc[list(
    #     KFold(fold_count, shuffle=True, random_state=42).split(host_df[host_df[target_name].notna()])[fold_id][1]
    # )].tolist())

    unlabeled_smiles = build_unlabeled_pool(
        target_name=target_name,
        host_df=host_df,
        extra_data_configs=extra_dataset_configs,
        unlabeled_smiles_csv_path=unlabeled_smiles_csv_path,
        # exclude_smiles=exclude,
        exclude_smiles=[]
    )

    unlabeled_ds = UnlabeledPairDataset(
        smiles_list=unlabeled_smiles,
        tokenizer=tokenizer,
        weak_strategies=weak_strategies,
        strong_strategies=strong_strategies
    )
    return train_ds, test_ds, unlabeled_ds, scaler
