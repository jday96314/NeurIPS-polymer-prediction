import argparse, json, os, random, sys
import traceback
from typing import List, Tuple
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, GraphDescriptors, MACCSkeys, rdFingerprintGenerator, AllChem, rdmolops
from typing import Iterable, List, Mapping, Sequence, Tuple
from functools import lru_cache

import numpy as np, pandas as pd, torch, optuna
from rdkit import Chem
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch import nn, Tensor
from torch.nn import functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import joblib
from tqdm import tqdm

#region graph helpers
def bond_type_to_int(bond: Chem.Bond) -> int:
    mapping = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3,
    }
    return mapping.get(bond.GetBondType(), 0)


def smiles_to_graph(smiles: str, label: float, weight: float|None = None) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Bad SMILES: {smiles}")
    atom_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    edge_indices, edge_attributes = [], []
    for bond in mol.GetBonds():
        start_index, end_index = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond_type_to_int(bond)
        edge_indices += [(start_index, end_index), (end_index, start_index)]
        edge_attributes += [bond_type, bond_type]
    return Data(
        x=torch.tensor(atom_nums, dtype=torch.long),
        edge_index=torch.tensor(edge_indices, dtype=torch.long).t(),
        edge_attr=torch.tensor(edge_attributes, dtype=torch.long),
        y=torch.tensor([label], dtype=torch.float32),
        weight=weight
    )


#region model
def _maybe_dropout(rate: float) -> nn.Module:
    return nn.Dropout(rate) if rate > 0.0 else nn.Identity()

def _scatter_sum(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """
    Fully-featured replacement for torch_scatter.scatter_add.
    Allocates a zero tensor of shape (dim_size, src.size(1)) and
    accumulates `src` rows whose destinations are in `index`.
    """
    out = torch.zeros(
        dim_size,
        src.size(1),
        dtype = src.dtype,
        device = src.device,
    )
    out.scatter_add_(
        dim = 0,
        index = index.unsqueeze(-1).expand_as(src),
        src = src,
    )
    return out

class BasicDMPNN(nn.Module):
    def __init__(
        self,
        atom_emb: int,
        bond_emb: int,
        msg_dim: int,
        msg_passes: int,
        msg_layer_count: int,
        out_hidden: int,
        emb_drop: float,
        msg_drop: float,
        head_drop: float,
    ):
        super().__init__()
        self.atom_embeddings = nn.Sequential(
            nn.Embedding(119, atom_emb),
            _maybe_dropout(emb_drop),
        )
        self.bond_embeddings = nn.Sequential(
            nn.Embedding(4, bond_emb),
            _maybe_dropout(emb_drop),
        )
        self.msg_init = nn.Linear(atom_emb + bond_emb, msg_dim)
        if msg_layer_count == 1:
            self.msg_update = nn.Linear(atom_emb + bond_emb + msg_dim, msg_dim)
        elif msg_layer_count == 2:
            self.msg_update = nn.Sequential(
                nn.Linear(atom_emb + bond_emb + msg_dim, msg_dim),
                nn.ReLU(),
                nn.Linear(msg_dim, msg_dim),
                # nn.LayerNorm(msg_dim),
            )
        else:
            raise ValueError("msg_layer_count must be 1 or 2")

        self.msg_passes = msg_passes
        self.msg_dropout = _maybe_dropout(msg_drop)

        self.readout = nn.Sequential(
            _maybe_dropout(head_drop),
            nn.Linear(msg_dim, out_hidden),
            nn.ReLU(),
            nn.Linear(out_hidden, 1),
        )

    def forward(self, data: Data | Batch) -> Tensor:
        atom = self.atom_embeddings(data.x)          # (num_atoms, atom_emb)
        bond = self.bond_embeddings(data.edge_attr)  # (num_edges, bond_emb)
        src, dst = data.edge_index                   # (2, num_edges)

        # 1. initial edge → message
        msg = F.relu(self.msg_init(torch.cat([atom[src], bond], dim=1)))

        # 2. message-passing iterations
        for _ in range(self.msg_passes):
            agg = _scatter_sum(msg, dst, dim_size=atom.size(0))
            upd_in = torch.cat([atom[src], bond, agg[src]], dim=1)
            msg = F.relu(self.msg_update(upd_in))
            msg = self.msg_dropout(msg)

        # 3. node & molecule readout
        node_state = _scatter_sum(msg, dst, dim_size=atom.size(0))
        num_molecules = int(data.batch.max().item()) + 1
        mol_state = _scatter_sum(node_state, data.batch, dim_size=num_molecules)

        return self.readout(mol_state).squeeze(-1)


#region training utils
def train_epoch(model, loader, loss_fn, lr_schedule, optimizer, device='cuda'):
    model.train()
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        # loss = loss_fn(model(batch), batch.y.squeeze().to(device))
        losses = loss_fn(model(batch), batch.y.squeeze().to(device))
        loss = (losses * batch.weight).sum() / batch.weight.sum()
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_schedule.step()


@torch.no_grad()
def test_model(model, loader, scaler, device='cuda') -> float:
    model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        y_hat = model(batch)
        preds.append(y_hat.cpu().numpy())
        labels.append(batch.y.cpu().numpy())
    preds = scaler.inverse_transform(np.concatenate(preds).reshape(-1, 1))
    labels = scaler.inverse_transform(np.concatenate(labels).reshape(-1, 1))
    return mean_absolute_error(labels, preds)

def prepare_dataloaders(train_df, test_df, target_name, batch_size):
    # SCALE TARGETS.
    scaler = StandardScaler()
    y_train = scaler.fit_transform(train_df[target_name].values[:, None]).ravel()
    y_val = scaler.transform(test_df[target_name].values[:, None]).ravel()

    # FORM GRAPHS.
    train_weights = train_df['weight'].to_list()
    train_graphs = [smiles_to_graph(s, y, w) for s, y, w in zip(train_df.SMILES, y_train, train_weights)]
    test_graphs = [smiles_to_graph(s, y) for s, y in zip(test_df.SMILES, y_val)]

    # CREATE DATALOADERS.
    train_loader = DataLoader(
        train_graphs, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0)
    test_loader = DataLoader(
        test_graphs, 
        batch_size=max(64, 2 * batch_size), 
        shuffle=False, 
        num_workers=0)
    
    return train_loader, test_loader, scaler

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
        'weight': extra_sample_weights
    })

    return extra_data_df

#region train & test

def train_test_models(
        extra_data_config: list[dict], 
        train_test_path: str, 
        target_name: str, 
        model_config: dict,
        output_directory_suffix: str = ""):
    # UNPACK CONFIG.
    ModelClass = BasicDMPNN # if model_type == "basic" else ResidualDMPNN

    atom_emb = model_config["atom_emb"]
    bond_emb = 16
    msg_dim = model_config["msg_dim"]
    msg_layer_count = model_config.get("msg_layer_count", 1)
    out_hidden = model_config["out_hidden"]
    msg_passes = model_config["msg_passes"]

    emb_drop = model_config.get("emb_drop", 0)
    msg_drop = model_config.get("msg_drop", 0)
    head_drop = model_config.get("head_drop", 0)

    batch_size = model_config["batch_size"]
    epochs = model_config["epochs"]
    max_lr = model_config["max_lr"]

    loss_fn = nn.MSELoss(reduction='none')

    # LOAD DATA.
    train_test_df = pd.read_csv(train_test_path)
    train_test_df['SMILES'] = train_test_df['SMILES'].map(canonicalize_smiles)
    train_test_df = train_test_df.dropna(subset=target_name)
    train_test_df['weight'] = [1 for _ in range(len(train_test_df))]

    # TRAIN & TEST MODELS.
    FOLD_COUNT = 5
    kf = KFold(n_splits=FOLD_COUNT, shuffle=True, random_state=42)
    maes = []
    models = []
    scalers = []
    for fold_index, (train_index, test_index) in enumerate(kf.split(train_test_df)):
        # PREPARE DATA.
        train_df = train_test_df.iloc[train_index]
        test_df = train_test_df.iloc[test_index]

        extra_train_df = load_extra_data(
            extra_data_config, 
            target_name=target_name,
            train_smiles=train_df['SMILES'].to_list(),
            test_smiles=test_df['SMILES'].to_list(),
            max_similarity=0.99
        )
        train_df = pd.concat([
            train_df[['SMILES', target_name, 'weight']],
            extra_train_df
        ])

        train_loader, test_loader, scaler = prepare_dataloaders(train_df, test_df, target_name, batch_size)

        # TRAIN.
        model = ModelClass(
            atom_emb,
            bond_emb=bond_emb,
            msg_dim=msg_dim,
            msg_layer_count=msg_layer_count,
            msg_passes=msg_passes,
            out_hidden=out_hidden,
            emb_drop=emb_drop,
            msg_drop=msg_drop,
            head_drop=head_drop,
        ).to('cuda')

        optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=1e-4)
        lr_schedule = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=len(train_loader) * epochs,
            pct_start=0.05,
            anneal_strategy="cos",
        )
        
        # TEST.
        for _ in tqdm(range(epochs), desc = f'Fold {fold_index+1}'):
            train_epoch(model, train_loader, loss_fn, lr_schedule, optimizer)

        # RECORD RESULTS.
        mae = test_model(model, test_loader, scaler)
        maes.append(mae)
        print(f'Fold {fold_index+1}/{FOLD_COUNT} MAE = {mae}')

        models.append(model)
        scalers.append(scaler)

    # LOG STATS.
    avg_mae = np.mean(maes)
    print(f'Avg MAE ({target_name}):', avg_mae)

    # MAYBE SAVE MODELS.
    if output_directory_suffix is not None:
        output_directory_path = f'models/d_mpnn_{target_name}_{int(avg_mae*10000)}{output_directory_suffix}'
        os.makedirs(output_directory_path, exist_ok=True)
        for fold_index, (scaler, model) in enumerate(zip(scalers, models)):
            model_path = f'{output_directory_path}/fold_{fold_index}.pth'
            torch.save(model.state_dict(), model_path)
            
            scaler_path = f'{output_directory_path}/fold_{fold_index}_scaler.pkl'
            joblib.dump(scaler, scaler_path)

    return avg_mae

#region Tuning

def get_optimal_data_config(target_name, model_config, trial_count):
    def objective(trial: optuna.Trial) -> float:
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

        model_config['batch_size'] = trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64])

        # TRAIN & TEST.
        mae = train_test_models(
            extra_data_config=extra_data_configs, 
            train_test_path='data/from_host_v2/train.csv', 
            target_name=target_name, 
            model_config=model_config,
            output_directory_suffix=None
        )
        return mae
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=trial_count, show_progress_bar=True)

    print(f"\nBest wMAE: {study.best_value:.5f}")
    print("Params:")
    print(json.dumps(study.best_params, indent=4))

    output_filepath = f'configs/DMPNN_data_{target_name}_{int(study.best_value * 10000)}.json'
    with open(output_filepath, 'w') as output_file:
        json.dump(study.best_params, output_file, indent=4)


def get_optimal_hyperparameters(target_name, extra_data_config_path, trial_count):
    def sample_dropout(trial: optuna.Trial, name: str, max_dropout_rate: float) -> float:
        if trial.suggest_categorical(f"use_{name}_dropout", [False, True]):
            return trial.suggest_float(f"{name}_drop", 0.05, max_dropout_rate)
        return 0.0

    def objective(trial: optuna.Trial) -> float:
        # GET HYPERPARAMETERS.
        # "Rg": { # 1.4444
        #     "atom_emb": 28,
        #     "msg_dim": 889,
        #     "out_hidden": 352,
        #     "msg_layer_count": 2,
        #     "msg_passes": 8,
        #     "use_emb_dropout": True,
        #     "emb_drop": 0.1229751649981316,
        #     "use_msg_dropout": False,
        #     "use_head_dropout": False,
        #     "batch_size": 32,
        #     "epochs": 116,
        #     "max_lr": 0.0002001702351251582
        # }
        model_config = {
            "model_type": "basic",

            # "msg_layer_count": trial.suggest_categorical("msg_layer_count", [2]),
            # "atom_emb": trial.suggest_int("atom_emb", 16, 512, log=True),
            # "msg_dim": trial.suggest_int("msg_dim", 96, 1024, log=True),
            # "out_hidden": trial.suggest_int("out_hidden", 32, 512, step=32),
            # "msg_passes": trial.suggest_int("msg_passes", 2, 8),
            # "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256]),
            "msg_layer_count": trial.suggest_categorical("msg_layer_count", [1]), # Tweaked for Rg
            "atom_emb": trial.suggest_int("atom_emb", 16, 256, log=True),
            "msg_dim": trial.suggest_int("msg_dim", 128, 1280, log=True),
            "out_hidden": trial.suggest_int("out_hidden", 64, 512, step=32),
            "msg_passes": trial.suggest_int("msg_passes", 4, 12),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128]),

            "emb_drop": sample_dropout(trial, "emb", max_dropout_rate=0.15),
            "msg_drop": sample_dropout(trial, "msg", max_dropout_rate=0.25),
            "head_drop": sample_dropout(trial, "head", max_dropout_rate=0.25),
            "epochs": trial.suggest_int("epochs", 50, 500),
            "max_lr": trial.suggest_float("max_lr", 5e-6, 2e-3, log=True)
        }

        # GET EXTRA DATA CONFIG.
        with open(TARGETS_TO_EXTRA_DATA_CONFIG_PATHS[args.target_name], 'r') as extra_data_config_file:
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

        # TRAIN & TEST.
        mae = train_test_models(
            extra_data_config=extra_dataset_configs, 
            train_test_path='data/from_host_v2/train.csv', 
            target_name=target_name, 
            model_config=model_config,
            output_directory_suffix=None
        )
        return float(mae)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=trial_count, show_progress_bar=True)

    print(f"\nBest wMAE: {study.best_value:.5f}")
    print("Params:")
    print(json.dumps(study.best_params, indent=4))

    output_filepath = f'configs/DMPNN_{target_name}_{int(study.best_value * 10000)}.json'
    with open(output_filepath, 'w') as output_file:
        json.dump(study.best_params, output_file, indent=4)

#region main

'''
Tg:
    Best wMAE: 43.62608
    Params:
    {
        "filepath_0": "data_filtering/standardized_datasets/Tg/dmitry_2.csv",
        "raw_label_weight_0": 0.9733355676382754,
        "dataset_weight_0": 0.26110619379096356,
        "max_error_ratio_0": 0.532846732793262,
        "purge_extra_train_smiles_overlap_0": False,
        "filepath_1": "data_filtering/standardized_datasets/Tg/dmitry_3.csv",
        "raw_label_weight_1": 0.741345028845899,
        "dataset_weight_1": 0.48442771996639244,
        "max_error_ratio_1": 4.947646557572168,
        "purge_extra_train_smiles_overlap_1": True,
        "filepath_2": "data_filtering/standardized_datasets/Tg/host_extra.csv",
        "raw_label_weight_2": 0.9641796094059099,
        "dataset_weight_2": 0.744128522507409,
        "max_error_ratio_2": 4.849485812854964,
        "purge_extra_train_smiles_overlap_2": True,
        "filepath_3": "data_filtering/standardized_datasets/Tg/LAMALAB.csv",
        "raw_label_weight_3": 0.9784045197761072,
        "dataset_weight_3": 0.6812232348030162,
        "max_error_ratio_3": 0.6360920449408985,
        "purge_extra_train_smiles_overlap_3": True,
        "batch_size": 4
    }
Rg:
    Best wMAE: 1.34202
    Params:
    {
        "filepath_0": "data_filtering/standardized_datasets/Rg/RadonPy.csv",
        "raw_label_weight_0": 0.23437974118559457,
        "dataset_weight_0": 0.4696110572019423,
        "max_error_ratio_0": 1.4195849661056434,
        "purge_extra_train_smiles_overlap_0": True,
        "batch_size": 64
    }
Tc:
    Best wMAE: 0.02412
    Params:
    {
        "filepath_0": "data_filtering/standardized_datasets/Tc/host_extra.csv",
        "raw_label_weight_0": 0.6329054841626748,
        "dataset_weight_0": 0.4871761884907006,
        "max_error_ratio_0": 1.3640122781271269,
        "purge_extra_train_smiles_overlap_0": True,
        "filepath_1": "data_filtering/standardized_datasets/Tc/RadonPy.csv",
        "raw_label_weight_1": 0.9595751635642056,
        "dataset_weight_1": 0.5936392048080116,
        "max_error_ratio_1": 0.5525840358722306,
        "purge_extra_train_smiles_overlap_1": True,
        "filepath_2": "data_filtering/standardized_datasets/Tc/RadonPy_filtered.csv",
        "raw_label_weight_2": 0.5622006417875236,
        "dataset_weight_2": 0.03906202384716695,
        "max_error_ratio_2": 1.4211329923746656,
        "purge_extra_train_smiles_overlap_2": False,
        "batch_size": 16
    }
Density:
    Best wMAE: 0.02182
    Params:
    {
        "filepath_0": "data_filtering/standardized_datasets/Density/dmitry.csv",
        "raw_label_weight_0": 0.050668688300460674,
        "dataset_weight_0": 0.13266431854512828,
        "max_error_ratio_0": 0.945453973939569,
        "purge_extra_train_smiles_overlap_0": False,
        "filepath_1": "data_filtering/standardized_datasets/Density/RadonPy.csv",
        "raw_label_weight_1": 0.14663046215525077,
        "dataset_weight_1": 0.9898563790893301,
        "max_error_ratio_1": 3.363492163150198,
        "purge_extra_train_smiles_overlap_1": False,
        "batch_size": 32
    }
'''

if __name__ == "__main__":
    # GET COMMANDLINE ARGS.
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-name", type=str, default="Tc")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # GET HYPERPARAMS.
    # TARGETS_TO_MODEL_CONFIGS = { # Tuned without extra data.
    #     "Tg": { # 50.2 base, 48.1 leaks, 48.8 host extra, 47.3 full
    #         "model_type": "basic",
    #         "atom_emb": 40,
    #         "msg_dim": 242,
    #         "out_hidden": 62,
    #         "msg_passes": 5,
    #         "emb_drop": 0,
    #         "msg_drop": 0,
    #         "head_drop": 0.2691323902671593,
    #         "batch_size": 32,
    #         "epochs": 73,
    #         "max_lr": 0.00043134633685439284
    #     },
    #     "FFV": { # 0.00452 base, N/A leaks, 0.00434 host extra, 0.00433 full (same as host)
    #         'model_type': 'basic', 
    #         'atom_emb': 131, 
    #         'msg_dim': 515, 
    #         'out_hidden': 608, 
    #         'msg_passes': 7, 
    #         'emb_drop': 0.055918540183476806, 
    #         'msg_drop': 0, 
    #         'head_drop': 0.05, 
    #         'batch_size': 8, 
    #         'epochs': 243, 
    #         'max_lr': 5.8668769484779167e-05
    #     },
    #     "Tc": { # 0.0256 base, 0.0260 leaks, 0.0260 host extra, 0.0260 full
    #         "model_type": "basic",
    #         "atom_emb": 545,
    #         "msg_dim": 723,
    #         "out_hidden": 222,
    #         "msg_passes": 5,
    #         "emb_drop": 0,
    #         "msg_drop": 0.20329077329185974,
    #         "head_drop": 0.10640722466508153,
    #         "batch_size": 8,
    #         "epochs": 82,
    #         "max_lr": 0.00013045038309030167
    #     },
    #     "Density": { # 0.0237 base, 0.0273 leaks, N/A host, 0.0274 full
    #         "model_type": "basic",
    #         "atom_emb": 31,
    #         "msg_dim": 305,
    #         "out_hidden": 786,
    #         "msg_passes": 5,
    #         "emb_drop": 0,
    #         "msg_drop": 0,
    #         "head_drop": 0,
    #         "batch_size": 4,
    #         "epochs": 418,
    #         "max_lr": 0.0002463720710124256
    #     },
    #     "Rg": { # 1.46 - 1.52 base (no extra)
    #         "model_type": "basic",
    #         "atom_emb": 51,
    #         "msg_dim": 926,
    #         "out_hidden": 369,
    #         "msg_passes": 4,
    #         "emb_drop": 0,
    #         "msg_drop": 0,
    #         "head_drop": 0.10354626111613492,
    #         "batch_size": 32,
    #         "epochs": 220,
    #         "max_lr": 6.578182373221e-05
    #     }
    # }
    # TARGETS_TO_MODEL_CONFIGS = { # Re-tuned with extra data (CV wMAE = 0.0590, LB = 0.066).
    #     "Tg": { # 43.595
    #         "model_type": "basic",
    #         "atom_emb": 455,
    #         "msg_dim": 97,
    #         "out_hidden": 128,
    #         "msg_passes": 3,
    #         "use_emb_dropout": True,
    #         "emb_drop": 0.07971650425969778,
    #         "use_msg_dropout": False,
    #         "use_head_dropout": True,
    #         "head_drop": 0.07706917345977361,
    #         "batch_size": 16,
    #         "epochs": 458,
    #         "max_lr": 1.7568158575438196e-05
    #     },
    #     "FFV": { # 0.00438
    #         'model_type': 'basic', 
    #         "atom_emb": 135,
    #         "msg_dim": 956,
    #         "out_hidden": 256,
    #         "msg_passes": 6,
    #         "use_emb_dropout": False,
    #         "use_msg_dropout": False,
    #         "use_head_dropout": True,
    #         "head_drop": 0.18157995019962409,
    #         "batch_size": 8,
    #         "epochs": 216,
    #         "max_lr": 1.7925926625654525e-05
    #     },
    #     "Tc": { # 0.02428
    #         "model_type": "basic",
    #         "atom_emb": 54,
    #         "msg_dim": 455,
    #         "out_hidden": 256,
    #         "msg_passes": 8,
    #         "use_emb_dropout": True,
    #         "emb_drop": 0.06153220662036694,
    #         "use_msg_dropout": True,
    #         "msg_drop": 0.08610554567427026,
    #         "use_head_dropout": True,
    #         "head_drop": 0.11941615204224441,
    #         "batch_size": 16,
    #         "epochs": 119,
    #         "max_lr": 0.0005179197208472095
    #     },
    #     "Density": { # 0.02257
    #         "model_type": "basic",
    #         "atom_emb": 102,
    #         "msg_dim": 395,
    #         "out_hidden": 224,
    #         "msg_passes": 5,
    #         "use_emb_dropout": False,
    #         "use_msg_dropout": True,
    #         "msg_drop": 0.07305572012066236,
    #         "use_head_dropout": False,
    #         "batch_size": 16,
    #         "epochs": 364,
    #         "max_lr": 5.229183242320349e-05
    #     },
    #     "Rg": { # 1.3631
    #         "model_type": "basic",
    #         "atom_emb": 26,
    #         "msg_dim": 833,
    #         "out_hidden": 256,
    #         "msg_passes": 5,
    #         "use_emb_dropout": True,
    #         "emb_drop": 0.12492099902514082,
    #         "use_msg_dropout": False,
    #         "use_head_dropout": True,
    #         "head_drop": 0.20503078366943217,
    #         "batch_size": 8,
    #         "epochs": 340,
    #         "max_lr": 1.9286139050798306e-05
    #     }
    # }
    # wMAE weights = weights = [0.00205237749659811, 0.6239248659724906, 2.219975435294379, 1.0640941761320575, 0.046558118429742154]
    TARGETS_TO_MODEL_CONFIGS = { # Tuned with extra data & double message passing layer count (CV wMAE = 0.0590, LB = 0.069).
        "Tg": { # 42.685
            "model_type": "basic",
            "atom_emb": 99,
            "msg_dim": 474,
            "msg_layer_count": 2,
            "out_hidden": 416,
            "msg_passes": 5,
            "use_emb_dropout": True,
            "emb_drop": 0.08589469198489456,
            "use_msg_dropout": True,
            "msg_drop": 0.17715533351841595,
            "use_head_dropout": True,
            "head_drop": 0.1232299087533877,
            "batch_size": 16,
            "epochs": 158,
            "max_lr": 2.2758461249712942e-05
        },
        "FFV": { # 0.00432
            'model_type': 'basic', 
            "atom_emb": 49,
            "msg_dim": 675,
            "msg_layer_count": 2,
            "out_hidden": 416,
            "msg_passes": 3,
            "use_emb_dropout": False,
            "use_msg_dropout": False,
            "use_head_dropout": False,
            "batch_size": 16,
            "epochs": 118,
            "max_lr": 7.927976918794681e-05
        },
        "Tc": { # 0.02398
            "model_type": "basic",
            "atom_emb": 30,
            "msg_dim": 367,
            "msg_layer_count": 2,
            "out_hidden": 224,
            "msg_passes": 6,
            "use_emb_dropout": False,
            "use_msg_dropout": False,
            "use_head_dropout": True,
            "head_drop": 0.17161918643168925,
            "batch_size": 32,
            "epochs": 356,
            "max_lr": 3.714500888724834e-05
        },
        "Density": { # 0.02146
            "atom_emb": 16,
            "msg_dim": 384,
            "out_hidden": 160,
            "msg_layer_count": 2,
            "msg_passes": 4,
            "use_emb_dropout": False,
            "use_msg_dropout": True,
            "msg_drop": 0.058408865494437544,
            "use_head_dropout": False,
            "batch_size": 16,
            "epochs": 203,
            "max_lr": 0.00017368702481115287
        },
        # "Rg": { # 1.4444
        #     "atom_emb": 28,
        #     "msg_dim": 889,
        #     "out_hidden": 352,
        #     "msg_layer_count": 2,
        #     "msg_passes": 8,
        #     "use_emb_dropout": True,
        #     "emb_drop": 0.1229751649981316,
        #     "use_msg_dropout": False,
        #     "use_head_dropout": False,
        #     "batch_size": 32,
        #     "epochs": 116,
        #     "max_lr": 0.0002001702351251582
        # }
        "Rg": { # 1.3659
            "msg_layer_count": 1,
            "atom_emb": 36,
            "msg_dim": 318,
            "out_hidden": 448,
            "msg_passes": 7,
            "batch_size": 16,
            "use_emb_dropout": False,
            "use_msg_dropout": False,
            "use_head_dropout": True,
            "head_drop": 0.07830174982408508,
            "epochs": 356,
            "max_lr": 2.7487203986202153e-05
        }
    }
    
    model_config = TARGETS_TO_MODEL_CONFIGS[args.target_name]

    # get_optimal_data_config(
    #     target_name=args.target_name, 
    #     model_config=model_config, 
    #     trial_count=6
    # )

    # wMAE = 0.05950 with tuned extra (pure automated)
    # wMAE = 0.05948 with tuned extra (manual FFV override)
    TARGETS_TO_EXTRA_DATA_CONFIG_PATHS = {
        "Tg": 'configs/DMPNN_data_Tg_436260.json', # 44.3307
        # "FFV": 'configs/DMPNN_data_FFV_43.json', # 0.004433 
        "FFV": 'configs/DMPNN_data_FFV_manual.json', # 0.004340
        "Tc": 'configs/DMPNN_data_Tc_241.json', # 0.02431
        "Density": "configs/LGBMRegressor_data_Density_262.json", # 0.022447
        # "Density": "configs/DMPNN_data_Density_215.json", # 0.0233
        "Rg": "configs/DMPNN_data_Rg_13420.json" # 1.3707
    }

    # get_optimal_hyperparameters(
    #     target_name=args.target_name, 
    #     extra_data_config_path=TARGETS_TO_EXTRA_DATA_CONFIG_PATHS[args.target_name], 
    #     trial_count=35
    # )

    #'''
    # LOAD EXTRA DATA CONFIG.
    with open(TARGETS_TO_EXTRA_DATA_CONFIG_PATHS[args.target_name], 'r') as extra_data_config_file:
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

    train_test_models(
        extra_data_config=extra_dataset_configs,
        train_test_path='data/from_host_v2/train.csv',
        target_name=args.target_name,
        model_config=model_config,
        # output_directory_suffix=f'_tuned_extra',
        output_directory_suffix=f'_retuned_extra',
        # output_directory_suffix=None
    )
    #'''