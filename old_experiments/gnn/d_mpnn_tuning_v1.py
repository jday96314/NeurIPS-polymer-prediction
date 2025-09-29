#!/usr/bin/env python3
"""
Optuna-driven *5-fold* hyper-parameter search for D-MPNN variants.

Tunable knobs
-------------
• model_type            – "basic" (no residual) or "residual"
• max_lr                – OneCycleLR peak learning rate (log-uniform)
• batch_size            – training mini-batch (categorical)
• epochs                – total training epochs (int)
• atom_embedding_size
• message_size
• output_hidden_size
• message_pass_count
• dropout enable flags + dropout rates at three sites
"""

from __future__ import annotations

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
from tqdm import tqdm

# --------------------------- graph helpers ---------------------------------- #
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
    atom_nums = [a.GetAtomicNum() for a in mol.GetAtoms()]
    e_idx, e_attr = [], []
    for b in mol.GetBonds():
        s, t = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bt = bond_type_to_int(b)
        e_idx += [(s, t), (t, s)]
        e_attr += [bt, bt]
    return Data(
        x=torch.tensor(atom_nums, dtype=torch.long),
        edge_index=torch.tensor(e_idx, dtype=torch.long).t(),
        edge_attr=torch.tensor(e_attr, dtype=torch.long),
        y=torch.tensor([label], dtype=torch.float32),
    )


# --------------------------- model definitions ------------------------------ #
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


# ----------------------------- training utils -------------------------------- #
def train_epoch(model, loader, loss_fn, opt, sched, device):
    model.train()
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(batch), batch.y.squeeze().to(device))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()


@torch.no_grad()
def fold_mae(model, loader, loss_fn, scaler, device) -> float:
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


# --------------------------- Optuna helpers ---------------------------------- #
def sample_dropout(trial: optuna.Trial, name: str, max_dropout_rate: float) -> float:
    if trial.suggest_categorical(f"use_{name}_dropout", [False, True]):
        return trial.suggest_float(f"{name}_drop", 0.05, max_dropout_rate)
    return 0.0


def objective(
        trial: optuna.Trial, 
        df: pd.DataFrame, 
        target: str, 
        old_config: dict,
        device: str) -> float:
    '''
    # model_type = trial.suggest_categorical("model_type", ["basic", "residual"])
    model_type = trial.suggest_categorical("model_type", ["basic"])

    # ----- hyper-parameters -----
    # atom_emb = trial.suggest_int("atom_emb", 32, 128, step=32)
    atom_emb = trial.suggest_int("atom_emb", 16, 512, log=True)
    # msg_dim = trial.suggest_int("msg_dim", 64, 512, step=64)
    # msg_dim = trial.suggest_int("msg_dim", 32, 768, step=32)
    msg_dim = trial.suggest_int("msg_dim", 96, 1024, log=True)
    # out_hidden = trial.suggest_int("out_hidden", 64, 512, step=64)
    out_hidden = trial.suggest_int("out_hidden", 32, 512, step=32)
    msg_passes = trial.suggest_int("msg_passes", 2, 8)

    emb_drop = sample_dropout(trial, "emb", max_dropout_rate=0.15)
    msg_drop = sample_dropout(trial, "msg", max_dropout_rate=0.25)
    head_drop = sample_dropout(trial, "head", max_dropout_rate=0.25)

    # batch_size = trial.suggest_categorical("batch_size", [16, 32, 48, 64])
    # batch_size = trial.suggest_categorical("batch_size", [16, 32, 48, 64, 128])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256])
    # epochs = trial.suggest_int("epochs", 3, 200)
    epochs = trial.suggest_int("epochs", 3, 300, log=True)
    # max_lr = trial.suggest_float("max_lr", 1e-5, 2e-4, log=True)
    # max_lr = trial.suggest_float("max_lr", 2e-5, 4e-4, log=True)
    max_lr = trial.suggest_float("max_lr", 5e-5, 1e-3, log=True)
    '''

    model_type = trial.suggest_categorical("model_type", ["basic"])

    old_atom_emb = old_config["atom_emb"]
    old_msg_dim = old_config["msg_dim"]
    old_out_hidden = old_config["out_hidden"]
    old_msg_passes = old_config["msg_passes"]
    atom_emb = trial.suggest_int("atom_emb", old_atom_emb//2, old_atom_emb*2, log=True)
    msg_dim = trial.suggest_int("msg_dim", old_msg_dim//2, min(old_msg_dim*2, 1536), log=True)
    out_hidden = trial.suggest_int("out_hidden", old_out_hidden//2, old_out_hidden*2, log=True)
    # msg_passes = trial.suggest_int("msg_passes", max(old_msg_passes-1, 1), old_msg_passes+1)
    msg_passes = trial.suggest_int("msg_passes", max(old_msg_passes-2, 1), old_msg_passes+2)

    old_emb_drop = old_config.get("emb_drop", 0)
    old_msg_drop = old_config.get("msg_drop", 0)
    old_head_drop = old_config.get("head_drop", 0)
    emb_drop = trial.suggest_categorical("emb_drop", [0, old_emb_drop, old_emb_drop + 0.05])
    msg_drop = trial.suggest_categorical("msg_drop", [0, old_msg_drop, old_msg_drop + 0.05])
    head_drop = trial.suggest_categorical("head_drop", [0, old_head_drop, old_head_drop + 0.05])

    # batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256])
    # epochs = trial.suggest_int("epochs", 3, 300, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64]) # TODO: Just tweaked for density!
    epochs = trial.suggest_int("epochs", 100, 500) # TODO: Just tweaked for density!
    max_lr = trial.suggest_float("max_lr", 5e-6, 2e-3, log=True)
    ###################################

    ModelClass = BasicDMPNN if model_type == "basic" else ResidualDMPNN
    loss_fn = nn.MSELoss()
    device_t = torch.device(device)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    maes = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(df)):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

        # ----- scaling -----
        scaler = StandardScaler()
        y_train = scaler.fit_transform(train_df[target].values[:, None]).ravel()
        y_val = scaler.transform(val_df[target].values[:, None]).ravel()

        # ----- graphs -----
        train_graphs = [smiles_to_graph(s, y) for s, y in zip(train_df.SMILES, y_train)]
        val_graphs = [smiles_to_graph(s, y) for s, y in zip(val_df.SMILES, y_val)]

        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_graphs, batch_size=max(64, 2 * batch_size), shuffle=False, num_workers=0)

        try:
            model = ModelClass(
                atom_emb,
                bond_emb=16,
                msg_dim=msg_dim,
                msg_passes=msg_passes,
                out_hidden=out_hidden,
                emb_drop=emb_drop,
                msg_drop=msg_drop,
                head_drop=head_drop,
            ).to(device_t)

            opt = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=1e-4)
            sched = OneCycleLR(
                opt,
                max_lr=max_lr,
                total_steps=len(train_loader) * epochs,
                pct_start=0.05,
                anneal_strategy="cos",
            )

            for _ in range(epochs):
                train_epoch(model, train_loader, loss_fn, opt, sched, device_t)

            maes.append(fold_mae(model, val_loader, loss_fn, scaler, device_t))
        except:
            traceback.print_exc()
            maes.append(100)
            break

    return float(np.mean(maes))


# ---------------------------------- main ------------------------------------ #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, default="data/from_host/train.csv")
    parser.add_argument("--target-name", type=str, default="Tg")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path).dropna(subset=[args.target_name])

    TARGETS_TO_OLD_CONFIGS = {
        "Tg": {
            "model_type": "basic",
            "atom_emb": 28,
            "msg_dim": 183,
            "out_hidden": 96,
            "msg_passes": 4,
            "use_emb_dropout": False,
            "use_msg_dropout": True,
            "msg_drop": 0.14320400596701305,
            "use_head_dropout": True,
            "head_drop": 0.2191323902671593,
            "batch_size": 256,
            "epochs": 100,
            "max_lr": 0.0004335406035920781
        },
        "FFV": {
            'model_type': 'basic', 
            'atom_emb': 141, 
            'msg_dim': 343, 
            'out_hidden': 399, 
            'msg_passes': 7, 
            'use_emb_dropout': True, 
            'emb_drop': 0.055918540183476806, 
            'use_msg_dropout': False, 
            'use_head_dropout': False, 
            'batch_size': 8, 
            'epochs': 289,
            'max_lr': 0.0016113660177597576
        },
        "Tc": {
            "model_type": "basic",
            "atom_emb": 493,
            "msg_dim": 844,
            "out_hidden": 352,
            "msg_passes": 6,
            "use_emb_dropout": True,
            "emb_drop": 0.11351848000238966,
            "use_msg_dropout": True,
            "msg_drop": 0.20329077329185974,
            "use_head_dropout": True,
            "head_drop": 0.10640722466508153,
            "batch_size": 256,
            "epochs": 247,
            "max_lr": 0.00018091641332601836
        },
        "Density": {
            "model_type": "basic",
            "atom_emb": 26,
            "msg_dim": 351,
            "out_hidden": 448,
            "msg_passes": 7,
            "use_emb_dropout": False,
            "use_msg_dropout": False,
            "use_head_dropout": False,
            "batch_size": 64,
            "epochs": 244,
            "max_lr": 0.00044098150082522787
        },
        "Rg": {
            "model_type": "basic",
            "atom_emb": 72,
            "msg_dim": 1007,
            "out_hidden": 352,
            "msg_passes": 5,
            "use_emb_dropout": False,
            "use_msg_dropout": True,
            "msg_drop": 0.05115597036497517,
            "use_head_dropout": True,
            "head_drop": 0.10354626111613492,
            "batch_size": 16,
            "epochs": 265,
            "max_lr": 0.00033729239982180413
        }
    }
    old_config = TARGETS_TO_OLD_CONFIGS[args.target_name]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
    study.optimize(
        lambda tr: objective(tr, df, args.target_name, old_config, args.device),
        n_trials=args.trials,
        show_progress_bar=True
    )

    print(f"\nBest validation MAE (for {args.target_name}):", study.best_value)
    print(json.dumps(study.best_params, indent=2))

'''
Best validation MAE (for Tg): 49.06496353149414
{
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
}
FFV: {'model_type': 'basic', 'atom_emb': 13
1, 'msg_dim': 515, 'out_hidden': 608, 'msg_passes': 7, 'emb_drop':
 0.055918540183476806, 'msg_drop': 0, 'head_drop': 0.05, 'batch_si
ze': 8, 'epochs': 243, 'max_lr': 5.8668769484779167e-05}. Best is 
trial 0 with value: 0.004473870806396008
Best validation MAE (for Tc): 0.025429774820804597
{
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
}
Best validation MAE (for Density): 0.022699838504195213
{
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
}
Best validation MAE (for Rg): 1.4544209718704224
{
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
'''
if __name__ == "__main__":
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    main()
