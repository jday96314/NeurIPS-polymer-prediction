import pandas as pd
from typing import Optional, List, Union
import numpy as np
import random
from rdkit import Chem
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from transformers import AutoTokenizer
from random import sample
from sklearn.preprocessing import StandardScaler
import os
import hashlib
from sklearn.metrics import mean_absolute_error
from rdkit.Chem import Descriptors, GraphDescriptors, MACCSkeys, rdFingerprintGenerator, AllChem, rdmolops
from functools import lru_cache
from rdkit import Chem, DataStructs
from typing import Iterable, List, Mapping, Sequence, Tuple

class SMILESDataset(Dataset):
    def __init__(
            self, 
            augmented_smiles: list[str], 
            original_smiles: list[str], 
            labels: list[float], 
            sample_weights: list[float],
            tokenizer, 
            max_length: int=256):
        self.augmented_smiles = augmented_smiles
        self.original_smiles = original_smiles
        self.labels = labels
        self.sample_weights = sample_weights
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.augmented_smiles)
    
    def __getitem__(self, idx):
        augmented_smiles = self.tokenizer.cls_token + self.augmented_smiles[idx]
        original_smiles = self.original_smiles[idx]
        label = self.labels[idx]
        sample_weight = self.sample_weights[idx]
        
        # Tokenize the SMILES string
        if not os.environ.get('SPLIT_CHARACTERS', False):
            encoding = self.tokenizer(
                augmented_smiles,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
        else:
            split_smiles = list(augmented_smiles)
            encoding = self.tokenizer(
                split_smiles,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                is_split_into_words=True,
                return_tensors='pt',
            )

        # Hash original smiles to compute ID.
        blake2b_hasher = hashlib.blake2b(digest_size=8)     # 8 bytes = 64 bits
        blake2b_hasher.update(original_smiles.encode('utf-8'))
        original_smiles_id = int.from_bytes(blake2b_hasher.digest(), byteorder='big', signed=True)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float32),
            'original_smiles_id': original_smiles_id,
            'sample_weight': sample_weight
        }

def augment_smiles_dataset(
        df: pd.DataFrame,
        smiles_column: str = 'SMILES',
        augmentation_strategies: List[str] = ['enumeration', 'kekulize', 'stereo_enum'],
        n_augmentations: int = 10,
        preserve_original: bool = True,
        random_seed: Optional[int] = None) -> pd.DataFrame:
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    def apply_augmentation_strategy(smiles: str, strategy: str) -> List[str]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return [smiles]
            
            augmented = []
            
            if strategy == 'enumeration':
                # Standard SMILES enumeration
                for _ in range(n_augmentations):
                    enum_smiles = Chem.MolToSmiles(mol, 
                                                 canonical=False, 
                                                 doRandom=True,
                                                 isomericSmiles=True)
                    augmented.append(enum_smiles)
            
            elif strategy == 'kekulize':
                # Kekulization variants
                try:
                    Chem.Kekulize(mol)
                    kek_smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
                    augmented.append(kek_smiles)
                except:
                    pass
            
            elif strategy == 'stereo_enum':
                # Stereochemistry enumeration
                for _ in range(n_augmentations // 2):
                    # Remove stereochemistry
                    Chem.RemoveStereochemistry(mol)
                    no_stereo = Chem.MolToSmiles(mol)
                    augmented.append(no_stereo)
            
            return list(set(augmented))  # Remove duplicates
            
        except Exception as e:
            print(f"Error in {strategy} for {smiles}: {e}")
            return [smiles]
    
    augmented_rows = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        original_smiles = row[smiles_column]
        sample_weight = row['sample_weight']
        
        if preserve_original:
            original_row = row.to_dict()
            original_row['augmentation_strategy'] = 'original'
            original_row['is_augmented'] = False
            original_row['original_smiles'] = original_smiles
            original_row['sample_weight'] = sample_weight
            augmented_rows.append(original_row)
        
        for strategy in augmentation_strategies:
            strategy_smiles = apply_augmentation_strategy(original_smiles, strategy)
            
            for aug_smiles in strategy_smiles:
                if aug_smiles != original_smiles:
                    new_row = row.to_dict().copy()
                    new_row[smiles_column] = aug_smiles
                    new_row['augmentation_strategy'] = strategy
                    new_row['is_augmented'] = True
                    new_row['original_smiles'] = original_smiles
                    new_row['sample_weight'] = sample_weight
                    augmented_rows.append(new_row)
    
    augmented_df = pd.DataFrame(augmented_rows)
    augmented_df = augmented_df.reset_index(drop=True)
    
    print(f"Original size: {len(df)}, Augmented size: {len(augmented_df)}")
    print(f"Augmentation factor: {len(augmented_df) / len(df):.2f}x")
    
    return augmented_df

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

    extra_labels = []
    extra_sample_weights = []
    added_smiles = []
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

        # RECORD DATASET.
        added_smiles.extend(raw_extra_data_df['SMILES'].tolist())
        extra_labels.extend(labels)
        
        dataset_weight = extra_data_config['dataset_weight']
        extra_sample_weights.extend([dataset_weight for _ in range(len(labels))])

    extra_data_df = pd.DataFrame({
        'SMILES': added_smiles,
        target_name: extra_labels,
        'sample_weight': extra_sample_weights
    })

    return extra_data_df

def get_train_test_datasets(
        csv_path, 
        extra_data_config,
        model_path,
        target_name, 
        fold_count, 
        fold_id, 
        train_augmentation_kwargs,
        test_augmentation_kwargs,):
    # LOAD & FILTER DATA.
    raw_df = pd.read_csv(csv_path)
    filtered_df = raw_df[raw_df[target_name].notna()]

    # SPLIT INTO TRAIN/TEST PARTITIONS.
    if fold_count > 1:
        kf = KFold(fold_count, shuffle=True, random_state=42)
        splits = list(kf.split(filtered_df))
        train_indices, test_indices = splits[fold_id]
    else:
        train_indices = list(range(len(filtered_df)))
        test_indices = np.random.choice(train_indices, size=int(0.1*len(train_indices)), replace=False)
    
    raw_train_df = filtered_df.iloc[train_indices]
    test_df = filtered_df.iloc[test_indices]
    
    # ADD EXTRA TRAINING DATA.
    extra_train_df = load_extra_data(
        extra_data_config, 
        target_name=target_name,
        train_smiles=raw_train_df['SMILES'].to_list(),
        test_smiles=test_df['SMILES'].to_list(),
        max_similarity=0.99
    )

    raw_train_df['sample_weight'] = [1 for _ in range(len(raw_train_df))]
    test_df['sample_weight'] = [1 for _ in range(len(test_df))]
    
    raw_train_df = pd.concat([
        raw_train_df[['SMILES', target_name, 'sample_weight']],
        extra_train_df
    ])

    # SCALE LABELS.
    label_scaler = StandardScaler()
    raw_train_df[target_name] = label_scaler.fit_transform(
        raw_train_df[target_name].values.reshape(-1, 1))
    
    test_df[target_name] = label_scaler.transform(
        test_df[target_name].values.reshape(-1, 1))
    
    # AUGMENT TRAINING & TEST DATA.
    augmented_train_df = augment_smiles_dataset(raw_train_df, **train_augmentation_kwargs)
    augmented_test_df = augment_smiles_dataset(test_df, **test_augmentation_kwargs)

    # CREATE DATASETS.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    train_dataset = SMILESDataset(
        augmented_smiles = augmented_train_df['SMILES'].tolist(), 
        original_smiles = augmented_train_df['original_smiles'].tolist(), 
        labels = augmented_train_df[target_name].tolist(), 
        sample_weights = augmented_train_df['sample_weight'].tolist(),
        tokenizer = tokenizer
    )
    
    test_dataset = SMILESDataset(
        augmented_smiles = augmented_test_df['SMILES'].tolist(),
        original_smiles = augmented_test_df['original_smiles'].tolist(), 
        labels = augmented_test_df[target_name].tolist(),
        sample_weights = augmented_train_df['sample_weight'].tolist(),
        tokenizer = tokenizer
    )

    return train_dataset, test_dataset, label_scaler

if __name__ == '__main__':
    train_dataset, test_dataset, label_scaler = get_train_test_datasets(
        csv_path = 'data/from_host/train.csv', 
        model_path = 'DeepChem/ChemBERTa-77M-MTR',
        target_name = 'Tg', 
        fold_count = 10, 
        fold_id = 0, 
        augmentation_kwargs = {})
    
    max_token_count = 0
    for sample in train_dataset:
        token_count = sum(sample['attention_mask'])
        max_token_count = max(max_token_count, token_count)

    print(max_token_count)