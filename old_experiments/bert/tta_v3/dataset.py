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

class SMILESDataset(Dataset):
    def __init__(self, augmented_smiles: list[str], original_smiles: list[str], labels: list[float], tokenizer, max_length=256):
        self.augmented_smiles = augmented_smiles
        self.original_smiles = original_smiles
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.augmented_smiles)
    
    def __getitem__(self, idx):
        augmented_smiles = self.tokenizer.cls_token + self.augmented_smiles[idx]
        original_smiles = self.original_smiles[idx]
        label = self.labels[idx]
        
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
        
        if preserve_original:
            original_row = row.to_dict()
            original_row['augmentation_strategy'] = 'original'
            original_row['is_augmented'] = False
            original_row['original_smiles'] = original_smiles
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
                    augmented_rows.append(new_row)
    
    augmented_df = pd.DataFrame(augmented_rows)
    augmented_df = augmented_df.reset_index(drop=True)
    
    print(f"Original size: {len(df)}, Augmented size: {len(augmented_df)}")
    print(f"Augmentation factor: {len(augmented_df) / len(df):.2f}x")
    
    return augmented_df

def get_train_test_datasets(
        csv_path, 
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
    kf = KFold(fold_count, shuffle=True, random_state=42)
    splits = list(kf.split(filtered_df))

    train_indices, test_indices = splits[fold_id]
    raw_train_df = filtered_df.iloc[train_indices]
    test_df = filtered_df.iloc[test_indices]

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
        tokenizer = tokenizer)
    
    test_dataset = SMILESDataset(
        augmented_smiles = augmented_test_df['SMILES'].tolist(),
        original_smiles = augmented_test_df['original_smiles'].tolist(), 
        labels = augmented_test_df[target_name].tolist(),
        tokenizer = tokenizer)

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