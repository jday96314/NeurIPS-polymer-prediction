#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Uniform model soup across multiple 5-fold runs.

- Python >= 3.11
- Assumes PyTorch .pth checkpoints are compatible (same architecture).
- Designed for Ubuntu.

Outputs:
  models/{YYYYMMDD_HHMMSS}_{model_id}_{run_count}/
    ├── polymer_bert_v2_{Target}.pth
    ├── scaler_{Target}.pkl
    ├── sources.txt
    └── soup_manifest.json
"""

from __future__ import annotations

import os
import re
import io
import json
import math
import pickle
import datetime as dt
from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple, Iterable, Optional
from tqdm import tqdm
import joblib

import torch
import numpy as np

# =========================
# Utilities
# =========================

def get_model_identifier_from_path(run_dir: str) -> str:
    """
    Parse something like: models/20250827_214805_modern_3epochs_2thresh_mlm_finetune_1e
    and return 'modern'. Fallback to 'soup' if not found.
    """
    base_name = os.path.basename(run_dir.rstrip("/"))
    m = re.search(r"\d{8}_\d{6}_([A-Za-z]+)", base_name)
    return m.group(1) if m else "soup"


def make_output_directory(model_identifier: str, model_count: int) -> str:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("models", f"{timestamp}_{model_identifier}_merged_{model_count}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def list_fold_dirs(run_dir: str) -> list[str]:
    folds = []
    if not os.path.isdir(run_dir):
        return folds
    for name in sorted(os.listdir(run_dir)):
        if re.fullmatch(r"fold_\d+", name) and os.path.isdir(os.path.join(run_dir, name)):
            folds.append(os.path.join(run_dir, name))
    return folds


def discover_target_names(run_dirs: list[str]) -> list[str]:
    """
    Scan fold_0 (or any fold) to find files like polymer_bert_v2_{Target}.pth
    """
    target_names = set()
    for run_dir in run_dirs:
        for fold_dir in list_fold_dirs(run_dir):
            for f in os.listdir(fold_dir):
                if f.startswith("polymer_bert_v2_") and f.endswith(".pth"):
                    # Extract target between last '_' and '.pth'
                    m = re.match(r"polymer_bert_v2_(.+)\.pth", f)
                    if m:
                        target_names.add(m.group(1))
            # If we saw any targets in this fold, that’s enough
            if target_names:
                break
    return sorted(target_names)


def load_checkpoint_state_dict(path: str) -> OrderedDict:
    """
    Load a checkpoint and return a clean state_dict suitable for averaging.
    Handles common wrapping keys and strips 'module.' prefixes if present.
    """
    raw_obj = torch.load(path, map_location="cpu")
    if isinstance(raw_obj, dict):
        # Common nesting possibilities
        for key_candidate in ["state_dict", "model_state_dict", "model"]:
            if key_candidate in raw_obj and isinstance(raw_obj[key_candidate], dict):
                state_dict = raw_obj[key_candidate]
                break
        else:
            # Assume it's already a state_dict
            state_dict = raw_obj
    else:
        # Rare: entire nn.Module saved; prefer .state_dict()
        try:
            state_dict = raw_obj.state_dict()  # type: ignore[attr-defined]
        except Exception:
            raise ValueError(f"Unrecognized checkpoint structure in {path}")

    # Normalize keys: strip "module." (DDP/DataParallel)
    normalized = OrderedDict()
    for k, v in state_dict.items():
        new_k = k[7:] if k.startswith("module.") else k
        normalized[new_k] = v
    return normalized


def intersect_keys(state_dicts: list[OrderedDict]) -> list[str]:
    if not state_dicts:
        return []
    key_sets = [set(sd.keys()) for sd in state_dicts]
    common = set.intersection(*key_sets)
    return sorted(common)


def average_state_dicts(state_dicts: list[OrderedDict]) -> OrderedDict:
    """
    Uniform average across tensors that:
      - are in the intersection of all state dict keys
      - share the same dtype and shape
    Non-floating tensors are copied from the first model (common for buffers).
    """
    if not state_dicts:
        raise ValueError("No state dicts provided for averaging.")

    common_keys = intersect_keys(state_dicts)
    if not common_keys:
        raise ValueError("No common parameter keys across checkpoints.")

    result = OrderedDict()
    for k in tqdm(common_keys, desc="Averaging parameters"):
        tensors = [sd[k] for sd in state_dicts]
        # Sanity: shapes must match
        shapes = {tuple(t.shape) for t in tensors}
        dtypes = {t.dtype for t in tensors}
        if len(shapes) != 1 or len(dtypes) != 1:
            # Skip incompatible tensors
            continue
        t0 = tensors[0]
        if t0.is_floating_point():
            stacked = torch.stack([t.to(torch.float32) for t in tensors], dim=0)
            mean_t = stacked.mean(dim=0).to(dtype=t0.dtype)
            result[k] = mean_t
        else:
            # Copy from the first (e.g., integer buffers); alternative is majority vote, but copy is fine.
            result[k] = t0
    return result

############ SLERP ##############

def _slerp_between_vectors(
        vector_a: torch.Tensor,
        vector_b: torch.Tensor,
        interpolation_weight: float,
        epsilon: float=1e-9
    ) -> torch.Tensor:
    """
    Spherical linear interpolation (SLERP) between two vectors.
    Operates directly on full tensors (flattened by caller).
    Falls back to linear interpolation for near-colinear or near-zero cases.
    """
    # Compute norms and guard against degenerate cases.
    norm_a = torch.linalg.norm(vector_a)
    norm_b = torch.linalg.norm(vector_b)
    if norm_a < epsilon or norm_b < epsilon:
        return (1.0 - interpolation_weight) * vector_a + interpolation_weight * vector_b

    # Work with unit directions for geodesic interpolation on the hypersphere.
    unit_a = vector_a / norm_a
    unit_b = vector_b / norm_b

    # Cosine of angle between directions; clamp for numerical stability.
    cosine_theta = torch.clamp(torch.dot(unit_a, unit_b), -1.0, 1.0)
    theta = torch.arccos(cosine_theta)

    if theta < 1e-7:
        # Directions are (almost) identical; linear is fine.
        return (1.0 - interpolation_weight) * vector_a + interpolation_weight * vector_b
    if abs(theta - torch.pi) < 1e-7:
        # Opposite directions; SLERP is ill-defined. Fall back to linear.
        return (1.0 - interpolation_weight) * vector_a + interpolation_weight * vector_b

    sin_theta = torch.sin(theta)
    coeff_a = torch.sin((1.0 - interpolation_weight) * theta) / sin_theta
    coeff_b = torch.sin(interpolation_weight * theta) / sin_theta
    return coeff_a * vector_a + coeff_b * vector_b


def _spherical_barycenter_direction(
        vectors: List[torch.Tensor],
        weights: Optional[List[float]]=None,
        epsilon: float=1e-9
    ) -> torch.Tensor:
    """
    Compute a weighted spherical 'mean direction' via online SLERP updates.
    All vectors are treated as points on a hypersphere (via normalization).
    Returns a unit vector (direction only).
    """
    # Normalize all vectors to unit length; handle zeros by fallback later.
    unit_vectors = []
    valid_mask = []
    for v in vectors:
        norm_v = torch.linalg.norm(v)
        if norm_v < epsilon:
            unit_vectors.append(v)  # keep placeholder; will be ignored
            valid_mask.append(False)
        else:
            unit_vectors.append(v / norm_v)
            valid_mask.append(True)

    # If no valid directions, return a zero vector.
    if not any(valid_mask):
        return torch.zeros_like(vectors[0])

    # Initialize with the first valid unit direction.
    mean_direction = None
    total_weight = 0.0
    if weights is None:
        weights = [1.0] * len(unit_vectors)

    for u, w, is_valid in zip(unit_vectors, weights, valid_mask):
        if not is_valid or w <= 0.0:
            continue
        if mean_direction is None:
            mean_direction = u.clone()
            total_weight = w
        else:
            # Online geodesic update toward the new direction with fractional weight.
            # This approximates the weighted Fréchet mean on S^n.
            interpolation_weight = float(w) / float(total_weight + w)
            mean_direction = _slerp_between_vectors(mean_direction, u, interpolation_weight)
            # Re-project to unit length to avoid drift.
            norm_mean = torch.linalg.norm(mean_direction)
            if norm_mean > epsilon:
                mean_direction = mean_direction / norm_mean
            total_weight += w

    if mean_direction is None:
        # Should not happen, but keep a safe fallback.
        return torch.zeros_like(vectors[0])
    return mean_direction


def average_state_dicts_slerp(
        state_dicts: List[OrderedDict],
        weights: Optional[List[float]]=None,
        epsilon: float=1e-9
    ) -> OrderedDict:
    """
    SLERP-based merge across multiple checkpoints.

    For each floating tensor:
      1) Compute unit directions of each model's tensor and a spherical mean direction via online SLERP.
      2) Compute a weighted arithmetic mean of the tensor L2 norms.
      3) Recompose: averaged_tensor = mean_direction * mean_norm, cast back to original dtype.

    Non-floating tensors are copied from the first checkpoint.

    Args:
        state_dicts: list of model state_dicts (assumed same architecture).
        weights: optional non-negative weights, one per state_dict (defaults to uniform).
        epsilon: numerical stability constant.

    Returns:
        OrderedDict containing the merged parameters.
    """
    if not state_dicts:
        raise ValueError("No state dicts provided for SLERP averaging.")

    if weights is not None and len(weights) != len(state_dicts):
        raise ValueError("Length of weights must match number of state dicts.")
    if weights is None:
        weights = [1.0] * len(state_dicts)

    # Determine the intersection of parameter keys across all models.
    common_keys = set(state_dicts[0].keys())
    for sd in state_dicts[1:]:
        common_keys &= set(sd.keys())
    common_keys = sorted(common_keys)
    if not common_keys:
        raise ValueError("No common parameter keys across checkpoints for SLERP averaging.")

    merged: OrderedDict[str, torch.Tensor] = OrderedDict()

    for parameter_key in tqdm(common_keys, desc="SLERP averaging parameters"):
        parameter_tensors = [sd[parameter_key] for sd in state_dicts]

        # All tensors must share dtype and shape; else skip this key.
        shapes = {tuple(t.shape) for t in parameter_tensors}
        dtypes = {t.dtype for t in parameter_tensors}
        if len(shapes) != 1 or len(dtypes) != 1:
            # Incompatible parameter across checkpoints; skip.
            continue

        representative_tensor = parameter_tensors[0]
        if not representative_tensor.is_floating_point():
            # Copy non-floating tensors (e.g., integer buffers) from the first state dict.
            merged[parameter_key] = representative_tensor
            continue

        original_dtype = representative_tensor.dtype
        original_shape = representative_tensor.shape

        # Flatten to 1D, operate in float32 for stability.
        flattened_vectors = [t.detach().to(torch.float32).reshape(-1) for t in parameter_tensors]

        # 1) Spherical mean of directions on the hypersphere.
        mean_direction_vector = _spherical_barycenter_direction(flattened_vectors, weights, epsilon)

        # 2) Weighted arithmetic mean of norms (magnitudes).
        norms = []
        for v in flattened_vectors:
            norms.append(torch.linalg.norm(v))
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        norms_tensor = torch.stack(norms, dim=0)
        weighted_mean_norm = (weights_tensor * norms_tensor).sum() / torch.clamp(weights_tensor.sum(), min=epsilon)

        # If mean direction is zero (all-zero tensors), fall back to linear average.
        if torch.linalg.norm(mean_direction_vector) < epsilon or torch.isnan(mean_direction_vector).any():
            stacked = torch.stack(flattened_vectors, dim=0)
            linear_mean = (stacked * weights_tensor.view(-1, 1)).sum(dim=0) / torch.clamp(weights_tensor.sum(), min=epsilon)
            averaged_tensor = linear_mean.reshape(original_shape).to(dtype=original_dtype)
        else:
            recomposed = mean_direction_vector * weighted_mean_norm
            averaged_tensor = recomposed.reshape(original_shape).to(dtype=original_dtype)

        merged[parameter_key] = averaged_tensor

    return merged

# =========================
# Scaler merging
# =========================

def load_pickle(path: str):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except:
        return joblib.load(path)


def save_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def is_standard_scaler(obj) -> bool:
    return (
        obj.__class__.__name__ == "StandardScaler"
        and hasattr(obj, "scale_")
        and hasattr(obj, "mean_")
    )


def is_minmax_scaler(obj) -> bool:
    return (
        obj.__class__.__name__ == "MinMaxScaler"
        and hasattr(obj, "data_min_")
        and hasattr(obj, "data_max_")
    )


def merge_standard_scalers(scalers: list) -> object | None:
    """
    Merge StandardScaler using weighted stats if n_samples_seen_ is available.
    Works for 1D targets; for multi-dim, merges elementwise.
    """
    try:
        # Ensure all have compatible shapes
        means = [np.array(s.mean_, dtype=np.float64) for s in scalers]
        vars_ = [np.array(getattr(s, "var_", s.scale_**2), dtype=np.float64) for s in scalers]
        counts = [
            int(getattr(s, "n_samples_seen_", 0)) if getattr(s, "n_samples_seen_", 0) is not None else 0
            for s in scalers
        ]
        if all(c > 0 for c in counts):
            total_n = sum(counts)
            weighted_mean = sum(m * n for m, n in zip(means, counts)) / float(total_n)
            # Pooled variance: sum n_i (var_i + (mu_i - mu)^2) / N
            pooled_num = sum(n * (v + (m - weighted_mean) ** 2) for v, m, n in zip(vars_, means, counts))
            pooled_var = pooled_num / float(total_n)
            merged = pickle.loads(pickle.dumps(scalers[0]))  # shallow clone via pickle
            merged.mean_ = weighted_mean
            merged.var_ = pooled_var
            merged.scale_ = np.sqrt(np.maximum(pooled_var, 0.0))
            merged.n_samples_seen_ = total_n
            return merged
        else:
            # Fallback: simple average of means/vars
            avg_mean = sum(means) / len(means)
            avg_var = sum(vars_) / len(vars_)
            merged = pickle.loads(pickle.dumps(scalers[0]))
            merged.mean_ = avg_mean
            merged.var_ = avg_var
            merged.scale_ = np.sqrt(np.maximum(avg_var, 0.0))
            # Leave n_samples_seen_ as-is or sum if present
            if any(c > 0 for c in counts):
                merged.n_samples_seen_ = sum(counts)
            return merged
    except Exception:
        return None


def merge_minmax_scalers(scalers: list) -> object | None:
    """
    Merge MinMaxScaler by taking global min of mins and max of maxes, then recompute data_range_, scale_, min_.
    """
    try:
        data_mins = [np.array(s.data_min_, dtype=np.float64) for s in scalers]
        data_maxs = [np.array(s.data_max_, dtype=np.float64) for s in scalers]
        global_min = np.min(np.stack(data_mins, axis=0), axis=0)
        global_max = np.max(np.stack(data_maxs, axis=0), axis=0)
        data_range = np.maximum(global_max - global_min, 1e-12)
        merged = pickle.loads(pickle.dumps(scalers[0]))
        merged.data_min_ = global_min
        merged.data_max_ = global_max
        merged.data_range_ = data_range
        # Respect feature_range from the representative scaler
        fr_min, fr_max = merged.feature_range
        merged.scale_ = (fr_max - fr_min) / data_range
        merged.min_ = fr_min - global_min * merged.scale_
        return merged
    except Exception:
        return None


def merge_scalers_for_target(scaler_paths: list[str]) -> Tuple[object | None, dict]:
    loaded_scalers = []
    for p in scaler_paths:
        try:
            loaded_scalers.append(load_pickle(p))
        except Exception:
            pass

    debug = {"used": [], "skipped": []}

    if not loaded_scalers:
        return None, debug

    # Try to use only compatible types
    first_type = loaded_scalers[0].__class__.__name__
    same_type = [s for s in loaded_scalers if s.__class__.__name__ == first_type]
    skipped = [s for s in loaded_scalers if s.__class__.__name__ != first_type]
    debug["skipped"] = [type(s).__name__ for s in skipped]

    scalers = same_type if same_type else [loaded_scalers[0]]
    debug["used"] = [type(s).__name__ for s in scalers]

    merged = None
    if is_standard_scaler(scalers[0]):
        merged = merge_standard_scalers(scalers)
    elif is_minmax_scaler(scalers[0]):
        merged = merge_minmax_scalers(scalers)

    # Fallback: pick the first one
    if merged is None:
        merged = scalers[0]

    return merged, debug


# =========================
# Orchestration
# =========================

def gather_paths_for_target(run_dirs: list[str], target_name: str) -> Tuple[list[str], list[str]]:
    """
    Return (model_paths, scaler_paths) for a given target across all folds of all runs.
    """
    model_paths = []
    scaler_paths = []
    for run_dir in run_dirs:
        for fold_dir in list_fold_dirs(run_dir):
            model_file = os.path.join(fold_dir, f"polymer_bert_v2_{target_name}.pth")
            scaler_file = os.path.join(fold_dir, f"scaler_{target_name}.pkl")
            if os.path.isfile(model_file):
                model_paths.append(model_file)
            if os.path.isfile(scaler_file):
                scaler_paths.append(scaler_file)
    return model_paths, scaler_paths


def build_soup_for_target(model_paths: list[str]) -> Tuple[OrderedDict, dict]:
    """
    Load and average checkpoints for a target. Returns (averaged_state_dict, diagnostics).
    """
    loaded = []
    skipped = []
    dtypes = set()
    shapes_ok = True

    for p in tqdm(model_paths, desc="Loading models"):
        try:
            sd = load_checkpoint_state_dict(p)
            loaded.append(sd)
        except Exception as e:
            skipped.append({"path": p, "reason": str(e)})

    if not loaded:
        raise RuntimeError("No valid checkpoints found to average.")

    # Optional: assert all common keys exist
    common = intersect_keys(loaded)
    if not common:
        raise RuntimeError("No common parameter keys across checkpoints.")

    averaged = average_state_dicts(loaded)
    # averaged = average_state_dicts_slerp(loaded)
    diag = {
        "num_input_models": len(model_paths),
        "num_loaded": len(loaded),
        "num_skipped": len(skipped),
        "skipped": skipped,
        "num_common_keys": len(common),
    }
    return averaged, diag


def write_sources_file(output_dir: str, run_dirs: list[str]) -> None:
    sources_path = os.path.join(output_dir, "sources.txt")
    with open(sources_path, "w", encoding="utf-8") as f:
        f.write("# Original model directories used to construct this soup\n")
        for d in run_dirs:
            f.write(d.rstrip("/") + "\n")

def sanity_check_model(model_path, scaler_path, target_name):
    from rank_up_v5.polybert_regressor import BertRegressor
    from transformers import AutoTokenizer
    from rdkit import Chem
    import pandas as pd
    from sklearn.metrics import mean_absolute_error

    def augment_smiles(smiles: str, n_augs: int):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return [smiles]
        augmented = {smiles}
        for _ in range(n_augs * 2):
            if len(augmented) >= n_augs: break
            aug_smiles = Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=True); augmented.add(aug_smiles)
            # aug_smiles = Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=True, allBondsExplicit=True, allHsExplicit=True); augmented.add(aug_smiles)
        return list(augmented)
    
    # LOAD MODEL & SCALER.
    BASE_MODEL_NAME = 'answerdotai/ModernBERT-base'
    model = BertRegressor(
        'answerdotai/ModernBERT-base',
        context_pooler_kwargs={
            "hidden_size": 768,
            "dropout_prob": 0.144,
            "activation_name": "gelu",
        },
        target_count=1
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    raw_state_dict = torch.load(model_path)
    clean_state_dict = {
        key.removeprefix("_orig_mod."): tensor
        for key, tensor in raw_state_dict.items()
    }
    model.load_state_dict(clean_state_dict)
    
    model = model.to('cuda').eval()
    scaler = joblib.load(scaler_path)

    # LOAD TEST DATA.
    test_df = pd.read_csv('data/from_host/train.csv')
    test_df = test_df[['SMILES', target_name]].dropna().reset_index()

    # INFERENCE.
    target_preds = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Sanity check"):
        augmented_smiles_list = augment_smiles(row['SMILES'], 20)
        inputs = tokenizer(augmented_smiles_list, return_tensors='pt', truncation=True, padding=True, max_length=256)
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        with torch.no_grad(): 
            _, preds = model(**inputs)
        
        scaled_preds = preds.cpu().numpy(); 
        unscaled_preds = scaler.inverse_transform(scaled_preds).flatten(); 
        final_pred = np.median(unscaled_preds)
        target_preds.append(final_pred)

    # EVALUATE.
    mae = mean_absolute_error(test_df[target_name].values, target_preds)
    print(f"[SANITY CHECK] Target: {target_name}, MAE: {mae:.4f}")


def main() -> None:
    input_run_directories = [
        "models/20250821_011341_modern_3epochs_2thresh_finetune_1e",
        # "models/20250821_194729_modern_6epochs_2thresh_finetune_1e",
        # "models/20250823_163627_modern_tuned",
        "models/20250827_214805_modern_3epochs_2thresh_mlm_finetune_1e",
    ]
    if not input_run_directories:
        raise SystemExit("No input run directories configured.")

    model_identifier = get_model_identifier_from_path(input_run_directories[0])
    output_dir = make_output_directory(model_identifier=model_identifier, model_count=len(input_run_directories))
    os.makedirs(output_dir, exist_ok=True)

    # Save a sources list
    write_sources_file(output_dir, input_run_directories)

    # Discover targets dynamically
    target_names = discover_target_names(input_run_directories)
    if not target_names:
        raise SystemExit("No target .pth files found in the provided runs.")

    soup_manifest: dict = {
        "output_dir": output_dir,
        "model_identifier": model_identifier,
        "run_directories": input_run_directories,
        "targets": {},
    }

    for target_name in target_names:
        print(f"[INFO] Processing target: {target_name}")
        model_paths, scaler_paths = gather_paths_for_target(input_run_directories, target_name)

        # Average models
        averaged_state_dict, model_diag = build_soup_for_target(model_paths)
        out_model_path = os.path.join(output_dir, f"polymer_bert_v2_{target_name}.pth")
        torch.save(averaged_state_dict, out_model_path)

        # Merge scalers (best-effort)
        merged_scaler, scaler_diag = merge_scalers_for_target(scaler_paths)
        out_scaler_path = os.path.join(output_dir, f"scaler_{target_name}.pkl")
        if merged_scaler is not None:
            save_pickle(merged_scaler, out_scaler_path)
        else:
            # If absolutely nothing usable, copy the first available scaler
            if scaler_paths:
                try:
                    first_scaler = load_pickle(scaler_paths[0])
                    save_pickle(first_scaler, out_scaler_path)
                except Exception:
                    pass

        soup_manifest["targets"][target_name] = {
            "num_models_found": len(model_paths),
            "num_scalers_found": len(scaler_paths),
            "model_paths_used": model_paths,
            "scaler_paths_used": scaler_paths,
            "model_diagnostics": model_diag,
            "scaler_diagnostics": scaler_diag,
            "output_model_path": out_model_path,
            "output_scaler_path": out_scaler_path,
        }

        # Optional sanity check
        sanity_check_model(out_model_path, out_scaler_path, target_name)

    # Save manifest
    manifest_path = os.path.join(output_dir, "soup_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(soup_manifest, f, indent=2)

    print(f"[DONE] Wrote averaged models and scalers to: {output_dir}")


if __name__ == "__main__":
    # sanity_check_model(
    #     'models/20250829_081615_modern_merged_4/polymer_bert_v2_Tg.pth',
    #     'models/20250829_081615_modern_merged_4/scaler_Tg.pkl',
    #     'Tg'
    # )

    # sanity_check_model(
    #     'models/20250821_011341_modern_3epochs_2thresh_finetune_1e/fold_0/polymer_bert_v2_Tg.pth',
    #     'models/20250821_011341_modern_3epochs_2thresh_finetune_1e/fold_0/scaler_Tg.pkl',
    #     'Tg'
    # )

    main()
