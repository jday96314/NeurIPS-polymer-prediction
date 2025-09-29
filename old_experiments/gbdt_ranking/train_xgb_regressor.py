from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

import sys
sys.path.append("./")
from fingerprinting_v5 import get_features_dataframe, load_extra_data  # reusing your module

# ----------------------------- Helpers -----------------------------

def load_feature_config(feature_config_json_path: str) -> Dict:
    with open(feature_config_json_path, "r") as f:
        feature_config = json.load(f)
    return feature_config


def build_features_from_smiles(smiles_series: pd.Series, features_config: Dict) -> pd.DataFrame:
    smiles_df = pd.DataFrame({"SMILES": smiles_series.astype(str).values})
    features_df = get_features_dataframe(smiles_df=smiles_df, **features_config)
    if not isinstance(features_df, pd.DataFrame):
        features_df = pd.DataFrame(features_df)
    return features_df


def align_feature_frames(feature_frames: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Ensure all frames have the same column set and order by taking the union of columns,
    filling missing columns with zeros. This guards against rare schema drift.
    """
    all_columns = set()
    for frame in feature_frames:
        all_columns.update(frame.columns.tolist())
    ordered_columns = sorted(list(all_columns))
    aligned = []
    for frame in feature_frames:
        missing = [c for c in ordered_columns if c not in frame.columns]
        if missing:
            for c in missing:
                frame[c] = 0.0
        aligned.append(frame[ordered_columns].copy())
    return aligned


def build_residual_regressor(params: argparse.Namespace) -> xgb.XGBRegressor:
    kwargs = dict(
        objective="reg:absoluteerror",
        # n_estimators=params.xgb_n_estimators,
        # learning_rate=params.xgb_learning_rate,
        # max_depth=params.xgb_max_depth,
        # subsample=params.xgb_subsample,
        # colsample_bytree=params.xgb_colsample_bytree,
        # reg_alpha=params.xgb_reg_alpha,
        # reg_lambda=params.xgb_reg_lambda,
        # random_state=0,
        eval_metric="mae",
        # early_stopping_rounds=params.early_stopping_rounds,
    )
    # Prefer new device API; fallback to gpu_hist if needed
    try:
        model = xgb.XGBRegressor(**kwargs, tree_method="hist", device="cuda")
        _ = model.get_xgb_params()
        return model
    except Exception:
        return xgb.XGBRegressor(**kwargs, tree_method="gpu_hist")


def load_ranker_for_target(ranker_models_dir: str, target_name: str) -> xgb.Booster:
    model_path = os.path.join(ranker_models_dir, f"{target_name}_best_xgbranker.json")
    if not os.path.exists(model_path):
        # also try "{target}_xgbranker.json" in case your naming differs
        alt_path = os.path.join(ranker_models_dir, f"{target_name}_xgbranker.json")
        model_path = alt_path if os.path.exists(alt_path) else os.path.join(ranker_models_dir, f"{target_name}.json")
    booster = xgb.Booster()
    booster.load_model(model_path)
    return booster


def predict_ranker_scores(booster: xgb.Booster, feature_frame: pd.DataFrame) -> np.ndarray:
    dmatrix = xgb.DMatrix(feature_frame.values)
    scores = booster.predict(dmatrix)
    return scores


# ----------------------------- Core per-target/ per-fold -----------------------------

def run_cv_for_target(
    host_df: pd.DataFrame,
    target_name: str,
    features_config: Dict,
    ranker_booster: xgb.Booster,
    params: argparse.Namespace,
    extra_data_configs: Optional[List[Dict]],
    output_dir_for_target: str,
) -> Dict:
    os.makedirs(output_dir_for_target, exist_ok=True)

    # Filter rows with this target present
    valid_mask = ~host_df[target_name].isna()
    df_target = host_df.loc[valid_mask].reset_index(drop=True)

    # Precompute host features (do this once; small data shown, but scales OK)
    host_features_df = build_features_from_smiles(df_target["SMILES"], features_config)
    # Precompute ranker scores for the entire host set
    host_ranker_scores = predict_ranker_scores(ranker_booster, host_features_df)
    host_labels = df_target[target_name].to_numpy(float)
    host_smiles = df_target["SMILES"].astype(str).tolist()

    # KFold on host rows only
    kf = KFold(n_splits=params.n_splits, shuffle=True, random_state=params.random_seed)

    # Hold OOF predictions and per-fold metrics
    oof_pred_full = np.zeros(len(df_target), dtype=float)
    oof_iso_pred_full = np.zeros(len(df_target), dtype=float)
    per_fold_metrics: List[Dict] = []

    # Save per-fold predictions dataframe rows
    oof_rows = []

    for fold_index, (train_idx, test_idx) in enumerate(kf.split(df_target), start=1):
        print(f"[{target_name}] Fold {fold_index}/{params.n_splits}: train={len(train_idx)}, test={len(test_idx)}")

        # Host train/test splits
        host_train_features = host_features_df.iloc[train_idx].reset_index(drop=True)
        host_test_features = host_features_df.iloc[test_idx].reset_index(drop=True)

        host_train_labels = host_labels[train_idx]
        host_test_labels = host_labels[test_idx]

        host_train_scores = host_ranker_scores[train_idx]
        host_test_scores = host_ranker_scores[test_idx]

        host_train_smiles = [host_smiles[i] for i in train_idx]
        host_test_smiles = [host_smiles[i] for i in test_idx]

        # 1) Isotonic regression on host train fold ONLY
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(host_train_scores, host_train_labels)

        iso_pred_train = iso.predict(host_train_scores)
        iso_pred_test = iso.predict(host_test_scores)

        # 2) Residual targets for host train
        host_train_residuals = host_train_labels - iso_pred_train

        # Load + prepare extra data (filtered vs test fold to avoid leakage)
        extra_features_df = None
        extra_labels_series = None
        extra_sample_weights = None

        if extra_data_configs is not None and len(extra_data_configs) > 0:
            extra_features_df, extra_labels_series, extra_sample_weights = load_extra_data(
                extra_data_configs=extra_data_configs,
                features_config=features_config,
                target_name=target_name,
                train_smiles=host_train_smiles,
                test_smiles=host_test_smiles,
                max_similarity=0.99
            )
            print(f"  Loaded extra training data: {len(extra_features_df)} rows")

        # If extra exists, compute its ranker scores and residuals w.r.t. THIS fold's isotonic calibrator
        if extra_features_df is not None and extra_labels_series is not None and len(extra_features_df) > 0:
            # Align schemas between host and extra
            host_train_features_aligned, extra_features_aligned = align_feature_frames([host_train_features, extra_features_df])
            host_train_features = host_train_features_aligned
            extra_features_df = extra_features_aligned

            extra_ranker_scores = predict_ranker_scores(ranker_booster, extra_features_df)
            iso_pred_extra = iso.predict(extra_ranker_scores)
            extra_residuals = extra_labels_series.to_numpy(float) - iso_pred_extra

            # Build training matrices
            X_train = pd.concat([host_train_features, extra_features_df], axis=0, ignore_index=True)
            y_train = np.concatenate([host_train_residuals, extra_residuals], axis=0)

            sample_weights = np.concatenate([
                np.ones(len(host_train_features), dtype=float),
                np.array(extra_sample_weights, dtype=float)
            ])
        else:
            X_train = host_train_features
            y_train = host_train_residuals
            sample_weights = np.ones(len(host_train_features), dtype=float)

        # For early stopping, validate on host test fold residuals only
        X_valid = host_test_features
        y_valid_residual = host_test_labels - iso_pred_test

        # Align schema between train and valid
        X_train_aligned, X_valid_aligned = align_feature_frames([X_train, X_valid])
        X_train = X_train_aligned
        X_valid = X_valid_aligned

        # Build residual regressor
        residual_model = build_residual_regressor(params)
        residual_model.fit(
            X_train.values, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_valid.values, y_valid_residual)],
            verbose=False,
        )

        # Inference on test fold: final prediction = isotonic(score) + residual_model(features)
        residual_pred_test = residual_model.predict(X_valid.values)
        final_pred_test = iso_pred_test + residual_pred_test

        # Record OOF predictions and metrics
        oof_pred_full[test_idx] = final_pred_test
        oof_iso_pred_full[test_idx] = iso_pred_test
        fold_metrics = {
            'mae': float(mean_absolute_error(host_test_labels, final_pred_test)),
            'isotonic_mae': float(mean_absolute_error(host_test_labels, iso_pred_test)),
        }
        per_fold_metrics.append({"fold": fold_index, **fold_metrics})

        # Save fold artifacts
        fold_dir = os.path.join(output_dir_for_target, f"fold_{fold_index}")
        os.makedirs(fold_dir, exist_ok=True)
        joblib.dump(iso, os.path.join(fold_dir, f"{target_name}_isotonic.joblib"))
        residual_model.save_model(os.path.join(fold_dir, f"{target_name}_residual_xgb.json"))

        # Collect rows for CSV
        for local_i, global_i in enumerate(test_idx):
            oof_rows.append({
                "fold": fold_index,
                "index": int(global_i),
                "SMILES": host_smiles[global_i],
                "y_true": float(host_labels[global_i]),
                "ranker_score": float(host_ranker_scores[global_i]),
                "iso_pred": float(oof_iso_pred_full[global_i]),
                "residual_pred": float(residual_pred_test[local_i]),
                "final_pred": float(final_pred_test[local_i]),
            })

    # Aggregate metrics across folds
    maes = [m["mae"] for m in per_fold_metrics]

    summary = {
        "target": target_name,
        "n_host_rows": int(len(df_target)),
        "cv_mae_mean": float(np.mean(maes)),
        "per_fold": per_fold_metrics,
    }

    # Save OOF predictions for this target
    oof_df = pd.DataFrame(oof_rows)
    oof_csv_path = os.path.join(output_dir_for_target, f"{target_name}_oof_predictions.csv")
    oof_df.to_csv(oof_csv_path, index=False)

    # Save summary metrics
    with open(os.path.join(output_dir_for_target, f"{target_name}_cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ----------------------------- Main -----------------------------

def load_extra_data_configs(config_json_path: str) -> List[Dict]:
    with open(config_json_path, "r") as extra_data_configs_file:
        raw_extra_data_configs = json.load(extra_data_configs_file)

    extra_dataset_configs = []
    extra_dataset_count = int(len(raw_extra_data_configs.keys()) / 5)
    for extra_dataset_index in range(extra_dataset_count):
        extra_dataset_configs.append({
            'filepath': raw_extra_data_configs[f'filepath_{extra_dataset_index}'],
            'raw_label_weight': raw_extra_data_configs[f'raw_label_weight_{extra_dataset_index}'],
            'dataset_weight': raw_extra_data_configs[f'dataset_weight_{extra_dataset_index}'],
            'max_error_ratio': raw_extra_data_configs[f'max_error_ratio_{extra_dataset_index}'],
            'purge_extra_train_smiles_overlap': raw_extra_data_configs[f'purge_extra_train_smiles_overlap_{extra_dataset_index}'],
        })

    return extra_dataset_configs

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default='data/from_host/train.csv', help="Host CSV with SMILES + targets.")
    parser.add_argument("--ranker_models_dir", type=str, default='gbdt_ranking/runs/xgb_ranker_tuning_50k_25trials/final_models', help="Directory with per-target XGBRanker .json files.")
    parser.add_argument("--feature_config_json", type=str, default='gbdt_ranking/runs/xgb_ranker_tuning_50k_25trials/final_models/best_feature_config.json', help="JSON file containing feature generation config.")
    parser.add_argument("--output_dir", type=str, default='gbdt_ranking/runs/debug', help="Where to save models/preds/metrics.")

    parser.add_argument("--targets", type=str, nargs="*", default=["Tg", "FFV", "Tc", "Density", "Rg"])
    parser.add_argument("--random_seed", type=int, default=1337)
    parser.add_argument("--n_splits", type=int, default=5)

    # Fixed XGB regression params (override-able via CLI if you want)
    parser.add_argument("--xgb_n_estimators", type=int, default=2000)
    parser.add_argument("--xgb_learning_rate", type=float, default=0.03)
    parser.add_argument("--xgb_max_depth", type=int, default=8)
    parser.add_argument("--xgb_subsample", type=float, default=0.8)
    parser.add_argument("--xgb_colsample_bytree", type=float, default=0.8)
    parser.add_argument("--xgb_reg_alpha", type=float, default=0.0)
    parser.add_argument("--xgb_reg_lambda", type=float, default=1.0)
    parser.add_argument("--early_stopping_rounds", type=int, default=100)

    return parser.parse_args()

def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load host data and feature config
    host_df = pd.read_csv(args.csv_path)
    features_config = load_feature_config(args.feature_config_json)


    # Ensure targets exist in host_df
    targets = [t for t in args.targets if t in host_df.columns]

    all_targets_summary: List[Dict] = []

    for target_name in targets:
        print(f"\n=== Processing target: {target_name} ===")

        # Load trained ranker booster for this target
        ranker_booster = load_ranker_for_target(args.ranker_models_dir, target_name)

        # Per-target output dir
        target_dir = os.path.join(args.output_dir, target_name)
        os.makedirs(target_dir, exist_ok=True)

        # Load extra data configs
        extra_data_configs_path = {
            'Tg': 'configs_backup/XGBRegressor_data_Tg_447786.json',
            'FFV': 'configs_backup/XGBRegressor_data_FFV_51.json',
            'Tc': 'configs_backup/XGBRegressor_data_Tc_248.json',
            'Density': 'configs_backup/XGBRegressor_data_Density_254.json',
            'Rg': 'configs_backup/XGBRegressor_data_Rg_14271.json',
        }[target_name]
        extra_dataset_configs = load_extra_data_configs(extra_data_configs_path)
        
        # Run CV loop and residual model training
        summary = run_cv_for_target(
            host_df=host_df,
            target_name=target_name,
            features_config=features_config,
            ranker_booster=ranker_booster,
            params=args,
            extra_data_configs=extra_dataset_configs,
            output_dir_for_target=target_dir,
        )
        print(json.dumps(summary, indent=2))
        all_targets_summary.append(summary)

    # Save global summary
    global_summary = {
        "targets": all_targets_summary,
        "overall_mae_mean": float(np.mean([s["cv_mae_mean"] for s in all_targets_summary])),
    }
    with open(os.path.join(args.output_dir, "overall_summary.json"), "w") as f:
        json.dump(global_summary, f, indent=2)

    print("\n=== Overall summary ===")
    print(json.dumps(global_summary, indent=2))


if __name__ == "__main__":
    main()