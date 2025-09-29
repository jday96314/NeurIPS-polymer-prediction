import argparse, json, os, random, sys
import traceback
from typing import List, Tuple

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
from torch_scatter import scatter_add
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


def smiles_to_graph(smiles: str, label: float) -> Data:
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
    )


#region model
def _maybe_dropout(rate: float) -> nn.Module:
    return nn.Dropout(rate) if rate > 0.0 else nn.Identity()


class BasicDMPNN(nn.Module):
    """No-residual baseline."""

    def __init__(
        self,
        atom_emb: int,
        bond_emb: int,
        msg_dim: int,
        msg_passes: int,
        out_hidden: int,
        emb_drop: float,
        msg_drop: float,
        head_drop: float,
    ):
        super().__init__()
        self.atom_embeddings = nn.Sequential(nn.Embedding(119, atom_emb), _maybe_dropout(emb_drop))
        self.bond_embeddings = nn.Sequential(nn.Embedding(4, bond_emb), _maybe_dropout(emb_drop))
        self.msg_init = nn.Linear(atom_emb + bond_emb, msg_dim)
        self.msg_update = nn.Linear(atom_emb + bond_emb + msg_dim, msg_dim)
        self.msg_passes = msg_passes
        self.msg_dropout = _maybe_dropout(msg_drop)
        self.readout = nn.Sequential(
            _maybe_dropout(head_drop),
            nn.Linear(msg_dim, out_hidden),
            nn.ReLU(),
            nn.Linear(out_hidden, 1),
        )

    def forward(self, data: Data | Batch) -> Tensor:
        atom = self.atom_embeddings(data.x)
        bond = self.bond_embeddings(data.edge_attr)
        src, dst = data.edge_index
        msg = F.relu(self.msg_init(torch.cat([atom[src], bond], 1)))

        for _ in range(self.msg_passes):
            agg = scatter_add(msg, dst, dim=0, dim_size=atom.size(0))
            upd_in = torch.cat([atom[src], bond, agg[src]], 1)
            msg = F.relu(self.msg_update(upd_in))
            msg = self.msg_dropout(msg)

        node_state = scatter_add(msg, dst, dim=0, dim_size=atom.size(0))
        mol_state = scatter_add(node_state, data.batch, dim=0)
        return self.readout(mol_state).squeeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, msg_dim: int, atom_dim: int, bond_dim: int, drop: float):
        super().__init__()
        self.lin = nn.Linear(atom_dim + bond_dim + msg_dim, msg_dim)
        self.norm = nn.LayerNorm(msg_dim)
        self.drop = _maybe_dropout(drop)

    def forward(self, msg, atom, bond, src, dst):
        inc = scatter_add(msg, dst, dim=0, dim_size=atom.size(0))
        upd = F.relu(self.lin(torch.cat([atom[src], bond, inc[src]], 1)))
        upd = self.drop(upd)
        return self.norm(msg + upd)


class ResidualDMPNN(nn.Module):
    """Residual/LayerNorm variant."""

    def __init__(
        self,
        atom_emb: int,
        bond_emb: int,
        msg_dim: int,
        msg_passes: int,
        out_hidden: int,
        emb_drop: float,
        msg_drop: float,
        head_drop: float,
    ):
        super().__init__()
        self.atom_embeddings = nn.Sequential(nn.Embedding(119, atom_emb), _maybe_dropout(emb_drop))
        self.bond_embeddings = nn.Sequential(nn.Embedding(4, bond_emb), _maybe_dropout(emb_drop))
        self.msg_init = nn.Linear(atom_emb + bond_emb, msg_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(msg_dim, atom_emb, bond_emb, msg_drop) for _ in range(msg_passes)]
        )
        self.readout = nn.Sequential(
            _maybe_dropout(head_drop),
            nn.Linear(msg_dim, out_hidden),
            nn.ReLU(),
            nn.Linear(out_hidden, 1),
        )

    def forward(self, data: Data | Batch) -> Tensor:
        atom, bond = self.atom_embeddings(data.x), self.bond_embeddings(data.edge_attr)
        src, dst = data.edge_index
        msg = F.relu(self.msg_init(torch.cat([atom[src], bond], 1)))
        for blk in self.blocks:
            msg = blk(msg, atom, bond, src, dst)
        node_state = scatter_add(msg, dst, dim=0, dim_size=atom.size(0))
        mol_state = scatter_add(node_state, data.batch, dim=0)
        return self.readout(mol_state).squeeze(-1)


#region training utils
def train_epoch(model, loader, loss_fn, lr_schedule, optimizer, device='cuda'):
    model.train()
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(batch), batch.y.squeeze().to(device))
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
    train_graphs = [smiles_to_graph(s, y) for s, y in zip(train_df.SMILES, y_train)]
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

#region train & test

def train_test_models(
        extra_train_path: str | None, 
        train_test_path: str, 
        target_name: str, 
        config: dict,
        output_directory_suffix: str = ""):
    # UNPACK CONFIG.
    model_type = config.get('model_type', 'basic')
    ModelClass = BasicDMPNN if model_type == "basic" else ResidualDMPNN

    atom_emb = config["atom_emb"]
    bond_emb = 16
    msg_dim = config["msg_dim"]
    out_hidden = config["out_hidden"]
    msg_passes = config["msg_passes"]

    emb_drop = config.get("emb_drop", 0)
    msg_drop = config.get("msg_drop", 0)
    head_drop = config.get("head_drop", 0)

    batch_size = config["batch_size"]
    epochs = config["epochs"]
    max_lr = config["max_lr"]

    loss_fn = nn.MSELoss()

    # LOAD DATA.
    train_test_df = pd.read_csv(train_test_path)
    # train_test_df = train_test_df.sample(n=1000)
    train_test_df = train_test_df.dropna(subset=target_name)

    extra_train_df = None
    if extra_train_path is not None:
        extra_train_df = pd.read_csv(extra_train_path)
        extra_train_df = extra_train_df.dropna(subset=target_name)
        print(f'Using {len(extra_train_df)} extra train rows.')

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

        if (extra_train_df is not None) and (len(extra_train_df) > 0):
            train_df = pd.concat([train_df, extra_train_df])

        train_loader, test_loader, scaler = prepare_dataloaders(train_df, test_df, target_name, batch_size)

        # TRAIN.
        model = ModelClass(
            atom_emb,
            bond_emb=bond_emb,
            msg_dim=msg_dim,
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

    # SAVE MODELS.
    output_directory_path = f'models/d_mpnn_{target_name}_{int(avg_mae*10000)}{output_directory_suffix}'
    os.makedirs(output_directory_path, exist_ok=True)
    for fold_index, (scaler, model) in enumerate(zip(scalers, models)):
        model_path = f'{output_directory_path}/fold_{fold_index}.pth'
        torch.save(model.state_dict(), model_path)
        
        scaler_path = f'{output_directory_path}/fold_{fold_index}_scaler.pkl'
        joblib.dump(scaler, scaler_path)

#region main

if __name__ == "__main__":
    # GET COMMANDLINE ARGS.
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, default="data/from_host/train.csv")
    parser.add_argument("--target-name", type=str, default="Tg")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # GET HYPERPARAMS.
    # Class weights: [2.05237750e-03, 6.23924866e-01, 2.21997544e+00, 1.06409418e+00, 4.65581184e-02]
    # >>> np.average([50.2, 0.00452, 0.0256, 0.0237, 1.49], weights=[2.05237750e-03, 6.23924866e-01, 2.21997544e+00, 1.06409418e+00, 4.65581184e-02])
    # np.float64(0.06502329441964554)
    TARGETS_TO_CONFIGS = {
        "Tg": { # 50.2 base, 48.1 leaks, 48.8 host extra, 47.3 full
            "model_type": "basic",
            "atom_emb": 40,
            "msg_dim": 242,
            "out_hidden": 62,
            "msg_passes": 5,
            "emb_drop": 0,
            "msg_drop": 0,
            "head_drop": 0.2691323902671593,
            "batch_size": 32,
            "epochs": 73,
            "max_lr": 0.00043134633685439284
        },
        "FFV": { # 0.00452 base, N/A leaks, 0.00434 host extra, 0.00433 full (same as host)
            'model_type': 'basic', 
            'atom_emb': 131, 
            'msg_dim': 515, 
            'out_hidden': 608, 
            'msg_passes': 7, 
            'emb_drop': 0.055918540183476806, 
            'msg_drop': 0, 
            'head_drop': 0.05, 
            'batch_size': 8, 
            'epochs': 243, 
            'max_lr': 5.8668769484779167e-05
        },
        "Tc": { # 0.0256 base, 0.0260 leaks, 0.0260 host extra, 0.0260 full
            "model_type": "basic",
            "atom_emb": 545,
            "msg_dim": 723,
            "out_hidden": 222,
            "msg_passes": 5,
            "emb_drop": 0,
            "msg_drop": 0.20329077329185974,
            "head_drop": 0.10640722466508153,
            "batch_size": 8,
            "epochs": 82,
            "max_lr": 0.00013045038309030167
        },
        "Density": { # 0.0237 base, 0.0273 leaks, N/A host, 0.0274 full
            "model_type": "basic",
            "atom_emb": 31,
            "msg_dim": 305,
            "out_hidden": 786,
            "msg_passes": 5,
            "emb_drop": 0,
            "msg_drop": 0,
            "head_drop": 0,
            "batch_size": 4,
            "epochs": 418,
            "max_lr": 0.0002463720710124256
        },
        "Rg": { # 1.46 - 1.52 base (no extra)
            "model_type": "basic",
            "atom_emb": 51,
            "msg_dim": 926,
            "out_hidden": 369,
            "msg_passes": 4,
            "emb_drop": 0,
            "msg_drop": 0,
            "head_drop": 0.10354626111613492,
            "batch_size": 32,
            "epochs": 220,
            "max_lr": 6.578182373221e-05
        }
    }
    config = TARGETS_TO_CONFIGS[args.target_name]

    EXTRA_TRAIN_DATASETS = [
        'data/from_host_v2/train_tc-natsume_full-dmitry_extra.csv', # Tg, Tc, Density
        'data/from_host_v2/train_host_extra.csv', # Tg, FFV, Tc
        'data/from_host_v2/train_host_plus_leaks_extra.csv', # Tg, FFV, Tc, Density
    ]
    for dataset_index, train_dataset_path in enumerate(EXTRA_TRAIN_DATASETS):
        print(f'Training with {train_dataset_path}')
        train_test_models(
            extra_train_path=train_dataset_path,
            train_test_path=args.csv_path,
            target_name=args.target_name,
            config=config,
            output_directory_suffix=f'_ds{dataset_index}'
        )