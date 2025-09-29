#!/usr/bin/env python3
"""
Train per-target XGBoost ranking models on polymer data like PI1M_500.csv.

- Builds features from SMILES (tries to import your project's `fingerprinting_v5.get_features_dataframe`;
  if unavailable, falls back to RDKit descriptors + Morgan fingerprints).
- For each target column, trains an XGBRanker with objective='rank:pairwise'.
- Uses a single random split: 85% train / 15% validation.
- Forms ranking groups by chunking each split into fixed-size groups (default 8).
- Reports NDCG@5/@10/@all, Spearman rho, and sampled pairwise accuracy on the validation set.
- Saves models and a JSON metrics summary.

Usage (Ubuntu, Python ≥3.11):

    python train_xgb_ranker.py \
        --csv_path /path/to/PI1M_500.csv \
        --output_dir ./xgb_ranker_runs/run1 \
        --group_size 8 \
        --validation_fraction 0.15 \
        --random_seed 1337

If your environment has xgboost>=2.0, GPU will be used via device='cuda' automatically.
Otherwise it will fall back to tree_method='gpu_hist'.

Assumptions:
- Input CSV has a 'SMILES' column and 5 targets: Tg, FFV, Tc, Density, Rg.
- You can override the targets via --targets.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from scipy.stats import spearmanr
import numpy as np
import pandas as pd
import xgboost as xgb

import sys
sys.path.append('./')
from fingerprinting_v5 import get_features_dataframe


# Wrapper that tries project featurizer first, then falls back
def build_features(smiles_series: pd.Series) -> pd.DataFrame:
    smiles_df = pd.DataFrame({"SMILES": smiles_series.astype(str).values})
    features_df = get_features_dataframe(
        smiles_df=smiles_df,
        morgan_fingerprint_dim=1024,
        atom_pair_fingerprint_dim=0,
        torsion_dim=0,
        use_maccs_keys=False,
        use_graph_features=False,
        backbone_sidechain_detail_level=0,
    )
    # Some project versions return a Polars DF; coerce to pandas if needed
    if not isinstance(features_df, pd.DataFrame):
        features_df = pd.DataFrame(features_df)
    return features_df

# ------------------ Ranking utilities ------------------
def make_random_split(n_rows: int, validation_fraction: float, rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:
    indices = list(range(n_rows))
    rng.shuffle(indices)
    split_point = int(round((1.0 - validation_fraction) * n_rows))
    train_indices = np.array(indices[:split_point], dtype=int)
    valid_indices = np.array(indices[split_point:], dtype=int)
    return train_indices, valid_indices

def make_groups_cover_all(n_items: int, group_size: int) -> List[int]:
    """
    Partition n_items into contiguous groups with size ~ group_size (>=2 each),
    ensuring the sum of group sizes == n_items.
    If the remainder would be 1, merge it into the previous group.
    """
    if group_size < 2:
        raise ValueError("group_size must be >= 2 for pairwise ranking.")
    if n_items < 2:
        raise ValueError(f"Need at least 2 items to form a ranking group (n_items={n_items}).")

    full_groups = n_items // group_size
    remainder = n_items - full_groups * group_size
    group_sizes = [group_size] * full_groups

    if remainder == 1:
        if group_sizes:
            group_sizes[-1] += 1  # merge the singleton
        else:
            # n_items == 1 already handled above, but keep defensive branch
            raise ValueError("Cannot form groups: only one item.")
    elif remainder >= 2:
        group_sizes.append(remainder)

    assert sum(group_sizes) == n_items, f"Group sizes {group_sizes} sum to {sum(group_sizes)} != {n_items}"
    return group_sizes

def build_group_vector_for_indices(indices: np.ndarray, group_size: int) -> List[int]:
    return make_groups_cover_all(len(indices), group_size)

def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, group_sizes: List[int], k: int | None = None) -> float:
    """
    Compute mean NDCG@k across groups. If k is None, use full group size.
    """
    start = 0
    ndcgs: List[float] = []
    for g in group_sizes:
        end = start + g
        true_g = y_true[start:end]
        pred_g = y_pred[start:end]
        if k is None:
            k_eff = g
        else:
            k_eff = min(k, g)

        # Sort by predicted relevance descending
        order = np.argsort(-pred_g)
        gains = (2.0 ** true_g[order] - 1.0)  # Using "exponential gain"; can also use linear gain true_g
        discounts = 1.0 / np.log2(np.arange(2, 2 + k_eff))
        dcg = float(np.sum(gains[:k_eff] * discounts))

        # Ideal DCG
        ideal_order = np.argsort(-true_g)
        ideal_gains = (2.0 ** true_g[ideal_order] - 1.0)
        idcg = float(np.sum(ideal_gains[:k_eff] * discounts))
        ndcgs.append(0.0 if idcg == 0.0 else (dcg / idcg))
        start = end
    return float(np.mean(ndcgs)) if ndcgs else float("nan")

def sampled_pairwise_accuracy(y_true: np.ndarray, y_pred: np.ndarray, group_sizes: List[int], max_pairs: int = 10000, rng: random.Random | None = None) -> float:
    """
    Sample up to max_pairs pairs within groups, returning the fraction where sign(pred_i - pred_j)
    matches sign(y_i - y_j). Ties are ignored.
    """
    if rng is None:
        rng = random.Random(1337)
    # Build index ranges for groups
    starts = np.cumsum([0] + group_sizes[:-1])
    total_pairs = 0
    correct = 0
    pair_budget = max_pairs
    for g_idx, g in enumerate(group_sizes):
        if g < 2:
            continue
        start = int(starts[g_idx])
        end = start + g
        idxs = np.arange(start, end, dtype=int)

        # Number of possible pairs in this group
        possible = g * (g - 1) // 2
        take = min(possible, pair_budget)
        if take <= 0:
            break

        # Sample unique pairs without replacement (approximate: random draws + dedup)
        seen = set()
        sampled = 0
        while sampled < take:
            i, j = rng.sample(list(idxs), 2)
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            sampled += 1

        for (i, j) in seen:
            dy = y_true[i] - y_true[j]
            dp = y_pred[i] - y_pred[j]
            if dy == 0 or dp == 0:
                continue
            if (dy > 0 and dp > 0) or (dy < 0 and dp < 0):
                correct += 1
            total_pairs += 1

        pair_budget -= take
        if pair_budget <= 0:
            break
    return float(correct) / float(total_pairs) if total_pairs > 0 else float("nan")

# ------------------ Training per target ------------------
@dataclass
class RankerConfig:
    n_estimators: int = 1000
    learning_rate: float = 0.05
    max_depth: int = 8
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50
    # GPU handling is done inside the builder

def build_xgbranker_gpu(cfg: RankerConfig):
    params = dict(
        objective="rank:pairwise",
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        eval_metric="rmse",
        random_state=0,
    )
    model = xgb.XGBRanker(
        **params,
        tree_method="hist",
        device="cuda",
    )
    _ = model.get_xgb_params()
    return model

def train_one_target(
    df: pd.DataFrame,
    target_name: str,
    validation_fraction: float,
    group_size: int,
    random_seed: int,
    cfg: RankerConfig,
    output_dir: str,
) -> dict:
    rng = random.Random(random_seed)

    # Keep rows with this target
    mask = ~df[target_name].isna()
    df_target = df.loc[mask].reset_index(drop=True)
    if len(df_target) < 10:
        raise ValueError(f"Not enough rows with non-NaN {target_name} to train a ranker.")

    # Build features
    features_df = build_features(df_target["SMILES"])
    labels = df_target[target_name].to_numpy(dtype=float)

    # Random split
    train_idx, valid_idx = make_random_split(len(df_target), validation_fraction, rng)

    X_train = features_df.iloc[train_idx].to_numpy(dtype=float)
    y_train = labels[train_idx]
    X_valid = features_df.iloc[valid_idx].to_numpy(dtype=float)
    y_valid = labels[valid_idx]

    # Make groups
    train_group = build_group_vector_for_indices(train_idx, group_size=group_size)
    valid_group = build_group_vector_for_indices(valid_idx, group_size=group_size)

    # Train
    model = build_xgbranker_gpu(cfg)
    model.fit(
        X_train,
        y_train,
        group=train_group,
        eval_set=[(X_valid, y_valid)],
        eval_group=[valid_group],
        verbose=False
    )

    # Predict on validation
    valid_pred = model.predict(X_valid)

    # Metrics
    try:
        rho, _ = spearmanr(y_valid, valid_pred)
    except Exception:
        rho = float("nan")

    ndcg5 = ndcg_at_k(y_valid, valid_pred, valid_group, k=5)
    ndcg10 = ndcg_at_k(y_valid, valid_pred, valid_group, k=10)
    ndcg_full = ndcg_at_k(y_valid, valid_pred, valid_group, k=None)
    pacc = sampled_pairwise_accuracy(y_valid, valid_pred, valid_group, max_pairs=10000, rng=rng)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{target_name}_xgbranker.json")
    model.save_model(model_path)
    metrics = {
        "target": target_name,
        "n_train": int(len(train_idx)),
        "n_valid": int(len(valid_idx)),
        "group_size": int(group_size),
        "spearman_rho_valid": float(rho) if rho == rho else None,
        "ndcg@5_valid": float(ndcg5) if ndcg5 == ndcg5 else None,
        "ndcg@10_valid": float(ndcg10) if ndcg10 == ndcg10 else None,
        "ndcg@all_valid": float(ndcg_full) if ndcg_full == ndcg_full else None,
        "pairwise_accuracy_valid": float(pacc) if pacc == pacc else None,
        "model_path": model_path,
    }
    with open(os.path.join(output_dir, f"{target_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=str, nargs="*", default=["Tg", "FFV", "Tc", "Density", "Rg"], help="Target column names to train.")
    parser.add_argument("--validation_fraction", type=float, default=0.15, help="Fraction of data to use for validation.")
    parser.add_argument("--group_size", type=int, default=8, help="Items per ranking group (>=2).")
    parser.add_argument("--random_seed", type=int, default=1337, help="Random seed for split and sampling.")
    parser.add_argument("--n_estimators", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--reg_alpha", type=float, default=0.0)
    parser.add_argument("--reg_lambda", type=float, default=1.0)

    args = parser.parse_args()

    OUTPUT_DIR_PATH = 'gbdt_ranking/runs/temp_run'
    DATA_PATH = 'data_filtering/relabeled_datasets/PI1M_500.csv'

    # Load data
    df = pd.read_csv(DATA_PATH)

    # Determine default targets if not provided
    if args.targets is None or len(args.targets) == 0:
        candidates = [c for c in df.columns if c != "SMILES"]
        args.targets = candidates

    cfg = RankerConfig(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
    )

    all_metrics: List[dict] = []
    for target_name in args.targets:
        print(f"\n=== Training XGBRanker for target: {target_name} ===")
        metrics = train_one_target(
            df=df,
            target_name=target_name,
            validation_fraction=args.validation_fraction,
            group_size=args.group_size,
            random_seed=args.random_seed,
            cfg=cfg,
            output_dir=OUTPUT_DIR_PATH,
        )
        print(json.dumps(metrics, indent=2))
        all_metrics.append(metrics)

    # Save overall summary
    summary_path = os.path.join(OUTPUT_DIR_PATH, "summary_metrics.json")
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved summary metrics to: {summary_path}")

if __name__ == "__main__":
    main()