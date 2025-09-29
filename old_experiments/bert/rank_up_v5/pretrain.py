# train_rankup.py
from __future__ import annotations

import os
from datetime import datetime
from typing import Iterable, Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, OneCycleLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset import SMILESDataset, augment_smiles_dataset, TARGET_NAMES
from polybert_regressor import BertRegressor


@torch.no_grad()
def compute_per_target_std(values_2d: np.ndarray) -> np.ndarray:
    """Compute per-target standard deviations from a 2D array shaped (n_samples, n_targets)."""
    per_target_std = np.std(values_2d, axis=0, ddof=0)
    # Avoid zeros which could make the margin degenerate
    per_target_std = np.where(per_target_std <= 1e-12, 1.0, per_target_std)
    return per_target_std


def make_random_split_indices(
        num_rows: int,
        train_fraction: float,
        random_state: int
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Single random split indices for pretraining."""
    rng = np.random.default_rng(seed=random_state)
    all_indices = np.arange(num_rows)
    rng.shuffle(all_indices)
    cutoff = int(train_fraction * num_rows)
    return all_indices[:cutoff], all_indices[cutoff:]


def make_datasets_for_single_split(
        csv_path: str,
        base_model_identifier: str,
        smiles_column_name: str = 'SMILES',
        train_fraction: float = 0.85,
        random_state: int = 42,
        max_length: int = 256,
        mask_probability: float = 0.0,
        do_explicit: bool = False
        ) -> Tuple[Dataset, Dataset, np.ndarray]:
    """Load pseudolabeled data, do a single random split, return tokenized datasets and per-target stds."""
    raw_df = pd.read_csv(csv_path)#.sample(n=1000)
    # target_names = [col for col in raw_df.columns if col != smiles_column_name]
    target_names = TARGET_NAMES

    train_idx, valid_idx = make_random_split_indices(len(raw_df), train_fraction, random_state)

    train_df = raw_df.iloc[train_idx].reset_index(drop=True)
    train_df = augment_smiles_dataset(train_df, do_explicit=do_explicit)
    valid_df = raw_df.iloc[valid_idx].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_identifier)

    train_dataset = SMILESDataset(
        augmented_smiles = train_df[smiles_column_name].tolist(), # We don't care about tracking original smile IDs in this script, so we pass the same list twice.
        original_smiles = train_df[smiles_column_name].tolist(),
        labels = train_df[target_names].values.astype(np.float32),
        sample_weights=[1.0] * len(train_df),
        tokenizer = tokenizer,
        max_length = max_length,
        mask_probability = mask_probability
    )

    valid_dataset = SMILESDataset(
        augmented_smiles = valid_df[smiles_column_name].tolist(),
        original_smiles = valid_df[smiles_column_name].tolist(),
        labels = valid_df[target_names].values.astype(np.float32),
        sample_weights=[1.0] * len(valid_df),
        tokenizer = tokenizer,
        max_length = max_length
    )

    # Per-target std from the TRAIN split (used to set masking margins).
    per_target_std = compute_per_target_std(train_df[target_names].values.astype(np.float32))

    return train_dataset, valid_dataset, per_target_std, target_names


def pairwise_rankup_loss_and_metrics(
        regression_output: torch.Tensor,           # (B, T)
        regression_output_permuted: torch.Tensor,  # (B, T)
        labels: torch.Tensor,                      # (B, T) float
        labels_permuted: torch.Tensor,             # (B, T) float
        per_target_margin: torch.Tensor,           # (T,) float
        bce_loss_fn: nn.BCEWithLogitsLoss
        ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute RankUp loss:
      - logits = (pred - pred_perm)
      - targets = 1 if (label > label_perm) else 0
      - mask = 1 if abs(label - label_perm) > margin else 0
    Return mean masked loss and metrics (masked accuracy per target and overall).
    """
    # Classification logits from model predictions (no sigmoid here; BCEWithLogitsLoss handles it)
    logits = regression_output - regression_output_permuted  # (B, T)

    # Binary targets from pseudolabel rankings
    with torch.no_grad():
        target_binary = (labels > labels_permuted).to(torch.float32)  # (B, T)

        # Confidence mask: only keep pairs with sufficiently large margin between labels
        absolute_differences = torch.abs(labels - labels_permuted)    # (B, T)
        # Broadcast (T,) over batch
        loss_mask = (absolute_differences > per_target_margin).to(torch.float32)  # (B, T)

    # BCE-with-logits loss per element, masked
    per_element_loss = bce_loss_fn(logits, target_binary)  # (B, T)
    masked_loss = per_element_loss * loss_mask
    loss = masked_loss.sum() / torch.clamp(loss_mask.sum(), min=1.0)

    # Metrics
    with torch.no_grad():
        # Predicted binary labels using 0.0 threshold on logits (equivalent to prob > 0.5)
        predicted_binary = (logits > 0.0).to(torch.float32)
        correct = (predicted_binary == target_binary).to(torch.float32) * loss_mask
        per_target_correct = correct.sum(dim=0)                       # (T,)
        per_target_count = torch.clamp(loss_mask.sum(dim=0), min=1.0) # (T,)
        per_target_accuracy = per_target_correct / per_target_count   # (T,)
        overall_accuracy = correct.sum() / torch.clamp(loss_mask.sum(), min=1.0)

    metrics = {
        "per_target_accuracy": per_target_accuracy,  # (T,)
        "overall_accuracy": overall_accuracy         # scalar tensor
    }
    return loss, metrics


def train_rankup_model(
        model: BertRegressor,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        target_names: List[str],
        per_target_margin: np.ndarray,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        device: str = 'cuda'
        ) -> Dict[str, List[float]]:
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, fused=True)
    total_steps = len(train_dataloader) * max(num_epochs, 1)
    # scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps)
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, total_steps=total_steps, pct_start=0.1, anneal_strategy='linear')

    scaler = torch.amp.GradScaler()
    bce_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    # Convert margins to tensor on device
    per_target_margin_tensor = torch.as_tensor(per_target_margin, dtype=torch.float32, device=device)

    history = {
        "train_loss": [],
        "valid_loss": [],
        "valid_overall_accuracy": [],
    }
    for target_name in target_names:
        history[f"valid_acc_{target_name}"] = []

    for epoch_index in range(num_epochs):
        # ----------------
        # Train
        # ----------------
        model.train()
        running_train_loss = 0.0

        for batch in tqdm(train_dataloader, desc=f"Training epoch {epoch_index + 1}/{num_epochs}", leave=False):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)  # (B, T), float

            batch_size = labels.size(0)
            # Build a random permutation per batch
            permuted_indices = torch.randperm(batch_size, device=device)

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, regression_output = model(input_ids=input_ids, attention_mask=attention_mask)            # (B, T)
                regression_output_permuted = regression_output[permuted_indices]                         # (B, T)
                labels_permuted = labels[permuted_indices]                                               # (B, T)

                loss, _ = pairwise_rankup_loss_and_metrics(
                    regression_output=regression_output,
                    regression_output_permuted=regression_output_permuted,
                    labels=labels,
                    labels_permuted=labels_permuted,
                    per_target_margin=per_target_margin_tensor,
                    bce_loss_fn=bce_loss_fn
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_train_loss += loss.item()

        average_train_loss = running_train_loss / max(len(train_dataloader), 1)
        history["train_loss"].append(average_train_loss)

        # ----------------
        # Validation
        # ----------------
        model.eval()
        running_valid_loss = 0.0
        per_target_accuracy_totals = torch.zeros(len(target_names), dtype=torch.float64, device=device)
        per_target_count_totals = torch.zeros(len(target_names), dtype=torch.float64, device=device)
        overall_correct_total = torch.tensor(0.0, dtype=torch.float64, device=device)
        overall_count_total = torch.tensor(0.0, dtype=torch.float64, device=device)

        with torch.no_grad():
            for batch in tqdm(valid_dataloader, desc="Validation", leave=False):
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)

                batch_size = labels.size(0)
                permuted_indices = torch.randperm(batch_size, device=device)

                _, regression_output = model(input_ids=input_ids, attention_mask=attention_mask)
                regression_output_permuted = regression_output[permuted_indices]
                labels_permuted = labels[permuted_indices]

                loss, metrics = pairwise_rankup_loss_and_metrics(
                    regression_output=regression_output,
                    regression_output_permuted=regression_output_permuted,
                    labels=labels,
                    labels_permuted=labels_permuted,
                    per_target_margin=per_target_margin_tensor,
                    bce_loss_fn=bce_loss_fn
                )
                running_valid_loss += loss.item()

                # accumulate accuracies
                per_target_accuracy = metrics["per_target_accuracy"]  # (T,)
                # To compute epoch-accuracy correctly with masking, sum correct and counts
                # Reconstruct counts via accuracy * count; but we did not return counts here.
                # Instead, recompute mask & counts for proper accumulation:
                absolute_differences = torch.abs(labels - labels_permuted)
                loss_mask = (absolute_differences > per_target_margin_tensor).to(torch.float32)
                logits = regression_output - regression_output_permuted
                target_binary = (labels > labels_permuted).to(torch.float32)
                predicted_binary = (logits > 0.0).to(torch.float32)
                correct = (predicted_binary == target_binary).to(torch.float32) * loss_mask

                per_target_accuracy_totals += correct.sum(dim=0).to(torch.float64)
                per_target_count_totals += torch.clamp(loss_mask.sum(dim=0), min=0.0).to(torch.float64)

                overall_correct_total += correct.sum().to(torch.float64)
                overall_count_total += torch.clamp(loss_mask.sum(), min=0.0).to(torch.float64)

        average_valid_loss = running_valid_loss / max(len(valid_dataloader), 1)
        history["valid_loss"].append(average_valid_loss)

        # finalize epoch accuracies
        per_target_epoch_acc = (per_target_accuracy_totals / torch.clamp(per_target_count_totals, min=1.0)).cpu().numpy()
        overall_epoch_acc = (overall_correct_total / torch.clamp(overall_count_total, min=1.0)).item()
        history["valid_overall_accuracy"].append(float(overall_epoch_acc))

        print(f"Epoch {epoch_index + 1}: train_loss={average_train_loss:.5f} | "
              f"valid_loss={average_valid_loss:.5f} | overall_acc={overall_epoch_acc:.4f}")

        for target_name, acc in zip(target_names, per_target_epoch_acc):
            history[f"valid_acc_{target_name}"].append(float(acc))
        formatted = ", ".join([f"{t}: {a:.4f}" for t, a in zip(target_names, per_target_epoch_acc)])
        print(f"  Per-target validation accuracy: {formatted}")

    return history


def train_and_save_rankup(
        base_model_identifier: str = 'answerdotai/ModernBERT-base',
        context_pooler_hidden_size: int = 768,
        epoch_count: int = 3,
        learning_rate: float = 2e-5,
        input_data_path: str = 'data/from_host/train.csv',
        output_dir_suffix: str = 'temp',
        train_fraction: float = 0.85,
        random_state: int = 42,
        batch_size_train: int = 32,
        batch_size_valid: int = 64,
        num_workers: int = 4,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        margin_std_fraction: float = 0.20,   # margin = fraction * per-target-std (on TRAIN split)
        device: str = 'cuda',
        mask_probability: float = 0.0,
        do_explicit: bool = False
        ) -> None:
    """
    Train BERT with RankUp pairwise supervision on ensemble pseudolabels.
    - Single random split (train_fraction).
    - Mask comparisons whose label difference is small: |y - y'| > margin.
    - Report validation accuracy (overall + per-target).
    """
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'models/{timestamp}_{output_dir_suffix}_rankup'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Data
    train_dataset, valid_dataset, per_target_std, target_names = make_datasets_for_single_split(
        csv_path=input_data_path,
        base_model_identifier=base_model_identifier,
        train_fraction=train_fraction,
        random_state=random_state,
        mask_probability=mask_probability,
        max_length=256 if not do_explicit else 512, # Longer max length for explicit Hs + bonds
        do_explicit=do_explicit
    )
    per_target_margin = per_target_std * float(margin_std_fraction)
    print("Per-target std:", {t: f"{s:.4f}" for t, s in zip(target_names, per_target_std)})
    print("Per-target margin:", {t: f"{m:.4f}" for t, m in zip(target_names, per_target_margin)})

    train_dataloader = DataLoader(
        train_dataset,
        batch_size = batch_size_train,
        shuffle = True,
        num_workers = num_workers,
        persistent_workers = persistent_workers if num_workers > 0 else False,
        pin_memory = pin_memory
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size = batch_size_valid,
        shuffle = False,
        num_workers = num_workers,
        persistent_workers = persistent_workers if num_workers > 0 else False,
        pin_memory = pin_memory
    )

    # Model
    model = BertRegressor(
        pretrained_model_path = base_model_identifier,
        target_count = len(target_names),
        context_pooler_kwargs = {
            'hidden_size': context_pooler_hidden_size,
            'dropout_prob': 0.144,
            'activation_name': 'gelu'
        }
    )

    # Train
    history = train_rankup_model(
        model = model,
        train_dataloader = train_dataloader,
        valid_dataloader = valid_dataloader,
        target_names = target_names,
        per_target_margin = per_target_margin,
        num_epochs = epoch_count,
        learning_rate = learning_rate,
        device = device
    )

    # Save model + small summary
    fold_dir = os.path.join(output_dir, 'single_split')
    os.makedirs(fold_dir, exist_ok=True)
    model_path = os.path.join(fold_dir, 'polymer_bert_rankup.pth')
    torch.save(model.state_dict(), model_path)

    summary_lines = []
    summary_lines.append(f"overall_acc: {history['valid_overall_accuracy'][-1]:.4f}")
    for target_name in target_names:
        summary_lines.append(f"{target_name}_acc: {history[f'valid_acc_{target_name}'][-1]:.4f}")
    summary_text = "\n".join(summary_lines)
    print("Validation summary:\n" + summary_text)

    summary_path = os.path.join(output_dir, 'validation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary_text)

'''
ModernBERT, PI1M_50000:
    3 epochs, 8e-5 lr, 0.2 margin: overall_acc: 0.9978
    6 epochs, 8e-5 lr, 0.2 margin: overall_acc: 0.9987
    3 epochs, 8e-5 lr, 0.8 margin: overall_acc: 0.9999
    3 epochs, 8e-5 lr, 0.2 margin: overall_acc: 0.9963 <-- Explicit Hs + bonds

codeBERT, PI1M_50000:
    3 epochs, 8e-5 lr, 0.2 margin: overall_acc: 0.9971
    6 epochs, 8e-5 lr, 0.2 margin: overall_acc: 0.9979
    3 epochs, 8e-5 lr, 0.8 margin: overall_acc: 0.9998

polyBERT, PI1M_50000:
    3 epochs, 8e-5 lr, 0.2 margin: overall_acc: 0.9966
    6 epochs, 8e-5 lr, 0.2 margin: overall_acc: 0.9977
    3 epochs, 8e-5 lr, 0.8 margin: overall_acc: 1.0000
    3 epochs, 8e-5 lr, 0.2 margin: overall_acc: 0.9918 <-- Explicit Hs + bonds

ModernBERT, realistic_50k_full:
    3 epochs, 8e-5 lr, 0.2 margin: overall_acc: 0.9973
    6 epochs, 8e-5 lr, 0.2 margin: overall_acc: 0.9985
    3 epochs, 8e-5 lr, 0.8 margin: overall_acc: 1.0000
'''
if __name__ == '__main__':
    train_and_save_rankup(
        base_model_identifier = 'answerdotai/ModernBERT-base',
        context_pooler_hidden_size = 768,
        learning_rate = 8e-5, # Could maybe bump 2x (~tied).
        output_dir_suffix = 'modern_3epochs_8thresh_v3',

        # base_model_identifier = 'kuelumbus/polyBERT',
        # context_pooler_hidden_size = 600,
        # learning_rate = 2e-5, # Could shift 2x in either direction (~tied).
        # output_dir_suffix = 'poly_3epochs_8thresh_v3',

        # base_model_identifier = 'microsoft/codebert-base',
        # context_pooler_hidden_size = 768,
        # learning_rate = 8e-5,
        # output_dir_suffix = 'code_3epochs_2thresh_v3',

        epoch_count = 3,
        # epoch_count = 6,
        margin_std_fraction = 0.2,
        # margin_std_fraction = 0.8,
        # input_data_path = 'data_filtering/relabeled_datasets/PI1M_5000.csv',
        # input_data_path = 'data_filtering/relabeled_datasets/PI1M_50000.csv',
        input_data_path = 'data_filtering/relabeled_datasets/PI1M_50000_v3.csv',
        # input_data_path = 'data/polyOne_partitioned_mini/realistic_50k_full.csv',
        # input_data_path = 'data_filtering/relabeled_datasets/realistic_50k_full.csv',
        train_fraction = 0.85,
        random_state = 42,
        batch_size_train = 32,
        batch_size_valid = 64,
        num_workers = 4,
        persistent_workers = True,
        pin_memory = True,
        device = 'cuda',
        mask_probability = 0.0,
    )
