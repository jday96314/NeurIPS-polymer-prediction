import argparse
import json
import os
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import optuna
from scipy.stats import spearmanr

import sys
sys.path.append('./')
from fingerprinting_v5 import get_features_dataframe

# ------------------ CLI ------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="data_filtering/relabeled_datasets/PI1M_50000.csv")
    parser.add_argument("--output_dir", type=str, default="gbdt_ranking/runs/tuning_default")
    parser.add_argument("--n_trials", type=int, default=40)
    parser.add_argument("--targets", type=str, nargs="*", default=["Tg", "FFV", "Tc", "Density", "Rg"])
    parser.add_argument("--validation_fraction", type=float, default=0.15)
    parser.add_argument("--random_seed", type=int, default=1337)
    parser.add_argument("--timeout_seconds", type=int, default=0, help="0 disables timeout")
    parser.add_argument("--opt_metric", type=str, default="ndcg10", choices=["ndcg5", "ndcg10", "ndcg_all", "spearman"])
    parser.add_argument("--group_size_min", type=int, default=6)
    parser.add_argument("--group_size_max", type=int, default=14)
    return parser.parse_args()

# ------------------ Utility: split & groups ------------------
def make_random_split(n_rows: int, validation_fraction: float, rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:
    indices = list(range(n_rows))
    rng.shuffle(indices)
    split_point = int(round((1.0 - validation_fraction) * n_rows))
    train_indices = np.array(indices[:split_point], dtype=int)
    valid_indices = np.array(indices[split_point:], dtype=int)
    return train_indices, valid_indices

def make_groups_cover_all(n_items: int, group_size: int) -> List[int]:
    if group_size < 2:
        raise ValueError("group_size must be >= 2 for pairwise ranking.")
    if n_items < 2:
        raise ValueError(f"Need at least 2 items to form a ranking group (n_items={n_items}).")

    full_groups = n_items // group_size
    remainder = n_items - full_groups * group_size
    group_sizes = [group_size] * full_groups

    if remainder == 1:
        if group_sizes:
            group_sizes[-1] += 1
        else:
            raise ValueError("Cannot form groups: only one item.")
    elif remainder >= 2:
        group_sizes.append(remainder)

    assert sum(group_sizes) == n_items, f"Group sizes {group_sizes} sum to {sum(group_sizes)} != {n_items}"
    return group_sizes

# ------------------ Metrics ------------------
def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, group_sizes: List[int], k: int | None = None, gain: str = "exp") -> float:
    start = 0
    ndcgs: List[float] = []
    for g in group_sizes:
        end = start + g
        true_g = y_true[start:end]
        pred_g = y_pred[start:end]
        k_eff = g if k is None else min(k, g)

        order = np.argsort(-pred_g)
        if gain == "exp":
            gains = (2.0 ** true_g[order] - 1.0)
        else:
            gains = true_g[order]
        discounts = 1.0 / np.log2(np.arange(2, 2 + k_eff))
        dcg = float(np.sum(gains[:k_eff] * discounts))

        ideal_order = np.argsort(-true_g)
        if gain == "exp":
            ideal_gains = (2.0 ** true_g[ideal_order] - 1.0)
        else:
            ideal_gains = true_g[ideal_order]
        idcg = float(np.sum(ideal_gains[:k_eff] * discounts))
        ndcgs.append(0.0 if idcg == 0.0 else (dcg / idcg))
        start = end
    return float(np.mean(ndcgs)) if ndcgs else float("nan")

# ------------------ Feature config ------------------
@dataclass
class FeatureConfig:
    morgan_fingerprint_dim: int = 1024
    atom_pair_fingerprint_dim: int = 0
    torsion_dim: int = 0
    use_maccs_keys: bool = False
    use_graph_features: bool = False
    backbone_sidechain_detail_level: int = 0

def build_features(smiles_series: pd.Series, feat_cfg: FeatureConfig) -> pd.DataFrame:
    smiles_df = pd.DataFrame({"SMILES": smiles_series.astype(str).values})
    features_df = get_features_dataframe(
        smiles_df=smiles_df,
        morgan_fingerprint_dim=feat_cfg.morgan_fingerprint_dim,
        atom_pair_fingerprint_dim=feat_cfg.atom_pair_fingerprint_dim,
        torsion_dim=feat_cfg.torsion_dim,
        use_maccs_keys=feat_cfg.use_maccs_keys,
        use_graph_features=feat_cfg.use_graph_features,
        backbone_sidechain_detail_level=feat_cfg.backbone_sidechain_detail_level,
    )
    if not isinstance(features_df, pd.DataFrame):
        features_df = pd.DataFrame(features_df)
    return features_df

# ------------------ Model builder ------------------
def build_xgbranker_gpu(params: Dict) -> "xgboost.sklearn.XGBRanker":
    import xgboost as xgb
    try:
        model = xgb.XGBRanker(
            **params,
            tree_method="hist",
            device="cuda",
        )
        _ = model.get_xgb_params()
    except Exception:
        model = xgb.XGBRanker(
            **params,
            tree_method="gpu_hist",
        )
    return model

# ------------------ Suggest spaces ------------------
def suggest_feature_params(trial: optuna.Trial) -> FeatureConfig:
    # Widened space so you can later narrow by hardcoding defaults.
    morgan = trial.suggest_categorical("morgan_fingerprint_dim", [0, 1024, 2048])
    atom_pair = trial.suggest_categorical("atom_pair_fingerprint_dim", [0, 1024])
    torsion = trial.suggest_categorical("torsion_dim", [0, 256])
    maccs = trial.suggest_categorical("use_maccs_keys", [False, True])
    graph = trial.suggest_categorical("use_graph_features", [False, True])
    backbone_detail = trial.suggest_categorical("backbone_sidechain_detail_level", [0, 1, 2])
    return FeatureConfig(
        morgan_fingerprint_dim=morgan,
        atom_pair_fingerprint_dim=atom_pair,
        torsion_dim=torsion,
        use_maccs_keys=maccs,
        use_graph_features=graph,
        backbone_sidechain_detail_level=backbone_detail,
    )

def suggest_model_params(trial: optuna.Trial) -> Dict:
    params = dict(
        objective="rank:pairwise",
        eval_metric="rmse",
        n_estimators=trial.suggest_int("n_estimators", 400, 2500),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
        max_depth=trial.suggest_int("max_depth", 4, 12),
        min_child_weight=trial.suggest_float("min_child_weight", 1e-3, 20.0, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 20.0, log=True),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),
        random_state=0,
    )
    return params

# ------------------ Per-target training ------------------
def train_one_target_with_cfg(
    df: pd.DataFrame,
    target_name: str,
    feat_cfg: FeatureConfig,
    model_params: Dict,
    validation_fraction: float,
    group_size: int,
    random_seed: int,
) -> Dict:
    rng = random.Random(random_seed)
    mask = ~df[target_name].isna()
    df_target = df.loc[mask].reset_index(drop=True)
    if len(df_target) < 10:
        return {"ok": False, "n": int(len(df_target))}

    features_df = build_features(df_target["SMILES"], feat_cfg)
    labels = df_target[target_name].to_numpy(dtype=float)

    train_idx, valid_idx = make_random_split(len(df_target), validation_fraction, rng)
    X_train = features_df.iloc[train_idx].to_numpy(dtype=float)
    y_train = labels[train_idx]
    X_valid = features_df.iloc[valid_idx].to_numpy(dtype=float)
    y_valid = labels[valid_idx]

    train_group = make_groups_cover_all(len(train_idx), group_size)
    valid_group = make_groups_cover_all(len(valid_idx), group_size)

    model = build_xgbranker_gpu(model_params)
    model.set_params(early_stopping_rounds=100)
    model.fit(
        X_train, y_train,
        group=train_group,
        eval_set=[(X_valid, y_valid)],
        eval_group=[valid_group],
        verbose=False,
    )
    valid_pred = model.predict(X_valid)

    # External metrics (we'll pick one later to optimize)
    ndcg5 = ndcg_at_k(y_valid, valid_pred, valid_group, k=5, gain="exp")
    ndcg10 = ndcg_at_k(y_valid, valid_pred, valid_group, k=10, gain="exp")
    ndcg_all = ndcg_at_k(y_valid, valid_pred, valid_group, k=None, gain="exp")
    try:
        rho, _ = spearmanr(y_valid, valid_pred)
    except Exception:
        rho = float("nan")

    return {
        "ok": True,
        "n": int(len(df_target)),
        "ndcg5": float(ndcg5) if ndcg5 == ndcg5 else None,
        "ndcg10": float(ndcg10) if ndcg10 == ndcg10 else None,
        "ndcg_all": float(ndcg_all) if ndcg_all == ndcg_all else None,
        "spearman": float(rho) if rho == rho else None,
        "best_iteration": int(getattr(model, "best_iteration", -1)),
        "model": model,
        "valid_pred": valid_pred,
        "y_valid": y_valid,
    }

# ------------------ Study objective ------------------
def build_objective(
    df: pd.DataFrame,
    targets: List[str],
    validation_fraction: float,
    random_seed: int,
    opt_metric: str,
    group_size_min: int,
    group_size_max: int,
):
    def objective(trial: optuna.Trial) -> float:
        feat_cfg = suggest_feature_params(trial)
        model_params = suggest_model_params(trial)
        group_size = trial.suggest_int("group_size", group_size_min, group_size_max, step=2)

        per_target_scores: List[float] = []
        for target in targets:
            res = train_one_target_with_cfg(
                df=df,
                target_name=target,
                feat_cfg=feat_cfg,
                model_params=model_params,
                validation_fraction=validation_fraction,
                group_size=group_size,
                random_seed=random_seed,
            )
            if not res["ok"]:
                return -1e9  # impossible low score
            score = {
                "ndcg5": res["ndcg5"],
                "ndcg10": res["ndcg10"],
                "ndcg_all": res["ndcg_all"],
                "spearman": res["spearman"],
            }[opt_metric]
            if score is None or not np.isfinite(score):
                return -1e9
            per_target_scores.append(float(score))

        mean_score = float(np.mean(per_target_scores))
        trial.set_user_attr("mean_score", mean_score)
        trial.set_user_attr("per_target_scores", per_target_scores)
        # Store the chosen feature config for convenience
        trial.set_user_attr("feature_config", asdict(feat_cfg))
        return mean_score
    return objective

# ------------------ Final retrain & save ------------------
def retrain_and_save_best(
    df: pd.DataFrame,
    targets: List[str],
    validation_fraction: float,
    random_seed: int,
    best_params: Dict,
    best_feat_cfg: FeatureConfig,
    best_group_size: int,
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)
    summary: List[Dict] = []

    for target in targets:
        res = train_one_target_with_cfg(
            df=df,
            target_name=target,
            feat_cfg=best_feat_cfg,
            model_params=best_params,
            validation_fraction=validation_fraction,
            group_size=best_group_size,
            random_seed=random_seed,
        )
        model = res["model"]
        model_path = os.path.join(output_dir, f"{target}_best_xgbranker.json")
        model.save_model(model_path)

        summary.append({
            "target": target,
            "ndcg5_valid": res["ndcg5"],
            "ndcg10_valid": res["ndcg10"],
            "ndcg_all_valid": res["ndcg_all"],
            "spearman_valid": res["spearman"],
            "best_iteration": res["best_iteration"],
            "model_path": model_path,
        })

    with open(os.path.join(output_dir, "best_feature_config.json"), "w") as f:
        json.dump(asdict(best_feat_cfg), f, indent=2)

    with open(os.path.join(output_dir, "best_model_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)

    with open(os.path.join(output_dir, "best_summary_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

# ------------------ Main ------------------
def main():
    args = parse_args()
    rng = random.Random(args.random_seed)

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.csv_path)

    # Filter targets to those present
    targets = [t for t in args.targets if t in df.columns]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.random_seed))
    objective = build_objective(
        df=df,
        targets=targets,
        validation_fraction=args.validation_fraction,
        random_seed=args.random_seed,
        opt_metric=args.opt_metric,
        group_size_min=args.group_size_min,
        group_size_max=args.group_size_max,
    )

    if args.timeout_seconds and args.timeout_seconds > 0:
        study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout_seconds, show_progress_bar=False)
    else:
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    # Persist study artifacts
    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    trials_csv_path = os.path.join(args.output_dir, "optuna_trials.csv")
    trials_df.to_csv(trials_csv_path, index=False)

    best_trial = study.best_trial
    best_feat_cfg_dict = best_trial.user_attrs.get("feature_config", {})
    best_feat_cfg = FeatureConfig(**best_feat_cfg_dict)
    best_group_size = int(best_trial.params["group_size"])
    # Extract model params from best trial
    best_model_params = suggest_model_params(best_trial)  # re-materialize with the same names
    # Overwrite with fixed bits that aren't part of trial params:
    best_model_params["objective"] = "rank:pairwise"
    best_model_params["eval_metric"] = "rmse"
    best_model_params["random_state"] = 0

    # Save best_config bundle
    best_bundle = {
        "mean_score": best_trial.user_attrs.get("mean_score", None),
        "per_target_scores": best_trial.user_attrs.get("per_target_scores", None),
        "feature_config": best_feat_cfg_dict,
        "model_params": best_model_params,
        "group_size": best_group_size,
        "opt_metric": args.opt_metric,
    }
    with open(os.path.join(args.output_dir, "best_config.json"), "w") as f:
        json.dump(best_bundle, f, indent=2)

    # Retrain final models with the best config and save
    final_dir = os.path.join(args.output_dir, "final_models")
    retrain_and_save_best(
        df=df,
        targets=targets,
        validation_fraction=args.validation_fraction,
        random_seed=args.random_seed,
        best_params=best_model_params,
        best_feat_cfg=best_feat_cfg,
        best_group_size=best_group_size,
        output_dir=final_dir,
    )

    print(f"\nBest mean {args.opt_metric}: {best_trial.value:.6f}")
    print(f"Saved trial history to: {trials_csv_path}")
    print(f"Saved best_config.json and final model artifacts in: {args.output_dir}")

if __name__ == "__main__":
    main()