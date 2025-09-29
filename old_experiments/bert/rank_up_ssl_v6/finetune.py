# supervised_finetune_rankup.py
import os, json, pickle, warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from datetime import datetime
from typing import Optional

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    get_train_test_datasets,
    get_train_test_and_unlabeled_datasets,
)
from polybert_regressor import BertRegressor

# -------------------- Helpers --------------------

def prevent_file_handle_exhaustion():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        pass

def _sample_pair_indices(batch_size: int, max_pairs: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (idx_i, idx_j) for pairwise comparisons.
    If max_pairs is None -> all pairs i<j. Otherwise randomly subsample max_pairs.
    """
    device = 'cpu'
    all_i, all_j = [], []
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            all_i.append(i)
            all_j.append(j)
    if max_pairs is not None and len(all_i) > max_pairs:
        choice = np.random.choice(len(all_i), size=max_pairs, replace=False)
        all_i = [all_i[k] for k in choice]
        all_j = [all_j[k] for k in choice]
    return torch.tensor(all_i, device=device, dtype=torch.long), torch.tensor(all_j, device=device, dtype=torch.long)

def _pair_bce_with_logits(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    scores: [B, 1] ARC scores
    labels: [P] in {0,1}, for P pairs
    Computes BCEWithLogits on logits = s_i - s_j.
    """
    s = scores.squeeze(-1)  # [B]
    # NOTE: build logits for selected pairs outside to avoid duplicate slicing here
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
    return loss_fn

# -------------------- Training (with optional RankUp ARC) --------------------

def train_model(
    model: BertRegressor,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    label_scaler,
    num_epochs: int = 10,
    learning_rate: float = 2e-5,
    head_lr_multiplier: float = 10.0,
    device: str = 'cuda',
    verbose: bool = True,
    # RankUp knobs
    enable_rankup: bool = False,
    unlabeled_dataloader: Optional[DataLoader] = None,
    unlabeled_batch_ratio: float = 7.0,
    confidence_threshold: float = 0.95,
    max_labeled_pairs_per_batch: Optional[int] = 2048,
    max_unlabeled_pairs_per_batch: Optional[int] = 4096,
    arc_loss_weight: float = 0.2,
    arc_unlabeled_weight: float = 1.0,
):
    model.to(device)

    # Parameter groups: backbone/pooler vs heads
    backbone_params = list(model.backbone.parameters()) + list(model.pooler.parameters())
    head_params = list(model.output.parameters())
    if getattr(model, "enable_arc", False):
        head_params += list(model.arc_head.parameters())

    optimizer = AdamW(
        [
            {"params": backbone_params, "weight_decay": 0.01},
            {"params": head_params, "weight_decay": 0.01},
        ],
        lr=learning_rate,
        fused=True
    )

    total_steps = len(train_dataloader) * num_epochs
    tiny = 1e-8
    head_max_lr = max(learning_rate * head_lr_multiplier, tiny)

    scheduler = OneCycleLR(
        optimizer=optimizer,
        max_lr=[learning_rate, head_max_lr],
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='linear'
    )

    regression_loss_fn = torch.nn.MSELoss(reduction='none')
    bce_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
    scaler = GradScaler()

    unlabeled_iter = iter(unlabeled_dataloader) if (enable_rankup and unlabeled_dataloader is not None) else None

    for epoch_index in range(num_epochs):
        if verbose:
            print(f"\nEpoch {epoch_index + 1}/{num_epochs}")

        total_train_loss = 0.0
        total_train_mae = 0.0
        train_progress_bar = tqdm(train_dataloader, desc="Training", leave=False) if verbose else train_dataloader

        model.train()
        for _batch_index, batch in enumerate(train_progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)  # shape [B]
            sample_weights = batch['sample_weight'].to(device)  # shape [B]

            optimizer.zero_grad()

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                if enable_rankup:
                    reg_output, arc_scores = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_arc_score=True
                    )
                else:
                    reg_output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_arc_score=False
                    )
                    arc_scores = None

                # Regression loss (weighted)
                per_sample_losses = regression_loss_fn(reg_output, labels.reshape(-1, 1)).squeeze(-1)  # [B]
                regression_loss = (per_sample_losses * sample_weights).sum() / sample_weights.sum()

                total_loss = regression_loss

                # ----- RankUp: labeled pairwise ARC loss -----
                if enable_rankup and arc_scores is not None:
                    batch_size = labels.shape[0]
                    idx_i, idx_j = _sample_pair_indices(batch_size, max_pairs=max_labeled_pairs_per_batch)
                    if idx_i.numel() > 0:
                        label_i = labels[idx_i]
                        label_j = labels[idx_j]
                        gt_pair = (label_i > label_j).float()  # 1 if i>j else 0

                        logit_pairs = (arc_scores.squeeze(-1)[idx_i] - arc_scores.squeeze(-1)[idx_j])  # [P]
                        labeled_arc_loss = bce_loss_fn(logit_pairs, gt_pair)
                        total_loss = total_loss + arc_loss_weight * labeled_arc_loss

                # ----- RankUp: unlabeled FixMatch-style ARC loss -----
                if enable_rankup and unlabeled_iter is not None:
                    try:
                        u = next(unlabeled_iter)
                    except StopIteration:
                        unlabeled_iter = iter(unlabeled_dataloader)
                        u = next(unlabeled_iter)

                    weak_ids = u['weak_input_ids'].to(device)
                    weak_mask = u['weak_attention_mask'].to(device)
                    strong_ids = u['strong_input_ids'].to(device)
                    strong_mask = u['strong_attention_mask'].to(device)

                    # Weak/strong ARC scores
                    _, weak_scores = model(weak_ids, attention_mask=weak_mask, return_arc_score=True)
                    _, strong_scores = model(strong_ids, attention_mask=strong_mask, return_arc_score=True)

                    bsz_u = weak_scores.shape[0]
                    u_i, u_j = _sample_pair_indices(bsz_u, max_pairs=max_unlabeled_pairs_per_batch)
                    u_i = u_i.to(device)
                    u_j = u_j.to(device)
                    if u_i.numel() > 0:
                        # Weak probs → pseudo-labels
                        weak_logits = weak_scores.squeeze(-1)[u_i] - weak_scores.squeeze(-1)[u_j]
                        weak_probs = torch.sigmoid(weak_logits)
                        pseudo_labels = (weak_probs >= 0.5).float()
                        confidence = torch.maximum(weak_probs, 1.0 - weak_probs)
                        confident_mask = (confidence >= confidence_threshold)
                        if confident_mask.any():
                            # Enforce consistency on strong view
                            strong_logits = strong_scores.squeeze(-1)[u_i[confident_mask]] - strong_scores.squeeze(-1)[u_j[confident_mask]]
                            ulb_arc_loss = bce_loss_fn(strong_logits, pseudo_labels[confident_mask])
                            total_loss = total_loss + arc_loss_weight * arc_unlabeled_weight * ulb_arc_loss

            # Backprop
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Report metrics on-the-fly
            unscaled_labels = label_scaler.inverse_transform(labels.detach().cpu().numpy().reshape(-1, 1))
            unscaled_outputs = label_scaler.inverse_transform(reg_output.detach().float().cpu().numpy().reshape(-1, 1))
            batch_mae = mean_absolute_error(unscaled_outputs, unscaled_labels)

            total_train_loss += float(total_loss.detach().cpu())
            total_train_mae += float(batch_mae)

            if verbose:
                train_progress_bar.set_postfix({
                    'loss': f'{total_loss.item():.4f}',
                    'mae': f'{batch_mae:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_mae = total_train_mae / len(train_dataloader)

        # ---------- Evaluation ----------
        model.eval()
        total_test_loss = 0.0
        all_test_predictions, all_test_labels, all_test_ids = [], [], []
        with torch.no_grad():
            val_progress = tqdm(test_dataloader, desc="Validation", leave=False)
            for batch in val_progress:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_arc_score=False)
                losses = regression_loss_fn(outputs, labels.reshape(-1, 1))
                loss_value = losses.sum()

                unscaled_labels = label_scaler.inverse_transform(labels.cpu().numpy().reshape(-1, 1))
                unscaled_outputs = label_scaler.inverse_transform(outputs.cpu().detach().numpy().reshape(-1, 1))

                all_test_predictions.extend(unscaled_outputs.T[0])
                all_test_labels.extend(unscaled_labels.T[0])
                total_test_loss += float(loss_value.detach().cpu())
                val_progress.set_postfix({'val_loss': f'{loss_value.item():.4f}'})

        avg_test_loss = total_test_loss / len(test_dataloader)
        avg_test_mae = mean_absolute_error(all_test_predictions, all_test_labels)

        if verbose:
            print(f"Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Train MAE: {avg_train_mae:.4f} | Test MAE: {avg_test_mae:.4f}")

    return avg_test_mae, all_test_predictions, all_test_labels

# -------------------- Example wiring (CV) --------------------

def train_and_save_tuned_models(
    target_names_to_config_paths,
    input_data_path: str = 'data/from_host/train.csv',
    target_names_to_extra_data_config_paths: dict = {},
    epoch_count_override: Optional[int] = None,
    fold_count: int = 5,
    output_dir_suffix: str = 'temp',
    # RankUp toggles (you can leave them default to stay fully supervised)
    enable_rankup: bool = False,
    unlabeled_smiles_csv_path: Optional[str] = None,
    unlabeled_batch_ratio: float = 7.0,
    confidence_threshold: float = 0.95,
):
    prevent_file_handle_exhaustion()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'models/{timestamp}_{output_dir_suffix}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    all_results = {}
    for target_name, config_path in target_names_to_config_paths.items():
        print(f'\n=== Training {target_name} across {fold_count} folds (cfg: {config_path}) ===')

        with open(config_path, 'r') as f:
            cfg = json.load(f)
        base_model_identifier = cfg['base_model_identifier']
        context_pooler_hidden_size = cfg['context_pooler_hidden_size']
        pretrained_weights_path = cfg['pretrained_weights_path']
        epoch_count = cfg['epoch_count'] if (epoch_count_override is None) else epoch_count_override
        learning_rate = cfg['backbone_lr']
        head_lr_multiplier = cfg['head_lr_multiplier']
        train_batch_size = cfg['train_batch_size'] // 2

        fold_maes = []
        for fold_id in range(fold_count):
            print(f'Fold {fold_id + 1}/{fold_count}')

            # ---- Datasets & Loaders (now optionally with unlabeled) ----
            train_ds, test_ds, unlabeled_ds, label_scaler = get_train_test_and_unlabeled_datasets(
                csv_path=input_data_path,
                extra_data_config=json.load(open(target_names_to_extra_data_config_paths[target_name], 'r')),
                model_path=base_model_identifier,
                target_name=target_name,
                fold_count=fold_count,
                fold_id=fold_id,
                train_augmentation_kwargs={},
                test_augmentation_kwargs={'augmentation_strategies': ['enumeration'], 'n_augmentations': 25},
                unlabeled_smiles_csv_path=unlabeled_smiles_csv_path,
            )

            train_loader = DataLoader(
                train_ds, batch_size=train_batch_size, shuffle=True,
                num_workers=4, persistent_workers=True, pin_memory=True
            )
            test_loader = DataLoader(
                test_ds, batch_size=train_batch_size * 2, shuffle=False,
                num_workers=4, persistent_workers=True, pin_memory=False
            )

            if enable_rankup:
                ulb_batch_size = max(1, int(train_batch_size * unlabeled_batch_ratio))
                unlabeled_loader = DataLoader(
                    unlabeled_ds, batch_size=ulb_batch_size, shuffle=True,
                    num_workers=4, persistent_workers=True, pin_memory=True
                )
            else:
                unlabeled_loader = None

            # ---- Model ----
            if pretrained_weights_path is not None:
                model = BertRegressor(
                    base_model_identifier,
                    target_count=5,
                    context_pooler_kwargs={'hidden_size': context_pooler_hidden_size, 'dropout_prob': 0.144, 'activation_name': 'gelu'},
                    enable_arc=enable_rankup
                )
                state = torch.load(pretrained_weights_path, map_location='cpu')
                model.load_state_dict(state, strict=False)
                model.output = torch.nn.Linear(context_pooler_hidden_size, 1)
                model = model.to('cuda')
            else:
                model = BertRegressor(
                    base_model_identifier,
                    target_count=1,
                    context_pooler_kwargs={'hidden_size': context_pooler_hidden_size, 'dropout_prob': 0.144, 'activation_name': 'gelu'},
                    enable_arc=enable_rankup
                ).to('cuda')

            # ---- Train ----
            test_mae, oof_preds, oof_labels = train_model(
                model=model,
                train_dataloader=train_loader,
                test_dataloader=test_loader,
                label_scaler=label_scaler,
                num_epochs=epoch_count,
                learning_rate=learning_rate,
                head_lr_multiplier=head_lr_multiplier,
                device='cuda',
                verbose=True,
                enable_rankup=enable_rankup,
                unlabeled_dataloader=unlabeled_loader,
                unlabeled_batch_ratio=unlabeled_batch_ratio,
                confidence_threshold=confidence_threshold,
            )

            # ---- Save fold artifacts ----
            fold_dir = os.path.join(output_dir, f'fold_{fold_id}')
            os.makedirs(fold_dir, exist_ok=True)
            joblib.dump(label_scaler, os.path.join(fold_dir, f'scaler_{target_name}.pkl'))
            joblib.dump(oof_preds, os.path.join(fold_dir, f'oof_preds_{target_name}.pkl'))
            joblib.dump(oof_labels, os.path.join(fold_dir, f'oof_labels_{target_name}.pkl'))
            torch.save(model.state_dict(), os.path.join(fold_dir, f'polymer_bert_rankup_{target_name}.pth'))

            fold_maes.append(test_mae)
            print(f'Fold {fold_id} MAE: {test_mae:.4f}')

        all_results[target_name] = {
            'fold_maes': fold_maes,
            'mean_mae': float(np.mean(fold_maes)),
            'std_mae': float(np.std(fold_maes))
        }
        print(f'{target_name}: {np.mean(fold_maes):.4f} ± {np.std(fold_maes):.4f} MAE')

    # Weighted MAE summary (same approach as your original)
    target_names = list(target_names_to_config_paths.keys())
    class_weights = _get_class_weights_for_subset(input_data_path, target_names)
    weighted_mae = float(np.average([all_results[t]['mean_mae'] for t in target_names], weights=class_weights))

    print('\n' + '='*60)
    print('FINAL RESULTS (Cross-Validation)')
    print('='*60)
    for t in target_names:
        r = all_results[t]
        print(f'{t:8s}: {r["mean_mae"]:.4f} ± {r["std_mae"]:.4f} MAE')
    print('-'*60)
    print(f'Weighted MAE: {weighted_mae:.4f}')

    # Persist summary
    with open(os.path.join(output_dir, 'cv_results_summary.txt'), 'w') as f:
        f.write('Cross-Validation Results Summary\n' + '='*40 + '\n\n')
        f.write(f'Number of folds: {fold_count}\nTargets: {target_names}\n\n')
        for t in target_names:
            r = all_results[t]
            f.write(f'{t}:\n  Mean MAE: {r["mean_mae"]:.4f}\n  Std MAE:  {r["std_mae"]:.4f}\n  Fold MAEs: {[f"{x:.4f}" for x in r["fold_maes"]]}\n\n')
        f.write(f'Weighted MAE: {weighted_mae:.4f}\n')
        f.write(f'Class weights: {dict(zip(target_names, class_weights))}\n')
    print(f'\nResults saved to: {output_dir}')

def _get_class_weights_for_subset(csv_path: str, target_names: list[str]):
    df = pd.read_csv(csv_path)
    scale_norm, count_norm = [], []
    for t in target_names:
        vals = df[t].values
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            scale_norm.append(1.0)
            count_norm.append(1.0)
            continue
        scale_norm.append(1.0 / max(1e-9, (max(vals) - min(vals))))
        count_norm.append((1.0 / max(1, len(vals)))**0.5)
    scale_norm = np.array(scale_norm)
    count_norm = np.array(count_norm)
    weights = scale_norm * len(target_names) * count_norm / max(1e-12, sum(count_norm))
    return weights

if __name__ == '__main__':
    # Example call (mirrors your tuned CV entrypoint), now with RankUp enabled.
    train_and_save_tuned_models(
        target_names_to_config_paths={
            "Tg": "configs/ModernBERT-base_Tg_411250.json",
            "FFV": "configs/ModernBERT-base_FFV_37.json",
            "Tc": "configs/ModernBERT-base_Tc_210.json",
            "Density": "configs/ModernBERT-base_Density_197.json",
            "Rg": "configs/ModernBERT-base_Rg_10306.json",
        },
        input_data_path="data/from_host/train.csv",
        target_names_to_extra_data_config_paths={
            "Tg": "configs/avg_data_Tg.json",
            "FFV": "configs/avg_data_FFV.json",
            "Tc": "configs/avg_data_Tc.json",
            "Density": "configs/avg_data_Density.json",
            "Rg": "configs/avg_data_Rg.json",
        },
        output_dir_suffix="modern_tuned_rankup",
        enable_rankup=True,
        # unlabeled_smiles_csv_path="data/PI1M/PI1M_50000_rankup.csv",
        unlabeled_smiles_csv_path="data/polyOne_partitioned_mini/realistic_50k.csv",
        unlabeled_batch_ratio=2,
        confidence_threshold=0.95     # FixMatch-style
    )
