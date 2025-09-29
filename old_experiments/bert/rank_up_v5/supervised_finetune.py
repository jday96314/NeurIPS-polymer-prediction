import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from tqdm import tqdm
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, OneCycleLR
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
import numpy as np
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime
from torch.amp import autocast, GradScaler
import optuna
import pickle
import json
from torch.utils.data import DataLoader

from dataset import SMILESDataset, get_train_test_datasets
from polybert_regressor import BertRegressor

def train_model(
        model, 
        train_dataloader, 
        test_dataloader, 
        label_scaler, 
        num_epochs=10, 
        learning_rate=2e-5,           # interpreted as BACKBONE LR
        head_lr_multiplier=10.0,      # NEW: head LR = learning_rate * head_lr_multiplier
        device='cuda',
        verbose=True):
    model.to(device)

    # Split params: backbone+pooler vs. head
    backbone_params = list(model.backbone.parameters()) + list(model.pooler.parameters())
    head_params = list(model.output.parameters())

    optimizer = AdamW(
        [
            {"params": backbone_params, "weight_decay": 0.01},
            {"params": head_params,     "weight_decay": 0.01},
        ],
        lr=learning_rate,
        fused=True
    )

    total_steps = len(train_dataloader) * num_epochs

    # Head LR multiplier can be 0 (requested); keep it tiny-but-nonzero for the scheduler
    tiny = 1e-8
    head_max_lr = max(learning_rate * head_lr_multiplier, tiny)

    scheduler = OneCycleLR(
        optimizer=optimizer,
        max_lr=[learning_rate, head_max_lr],  # [backbone, head]
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='linear'
    )

    loss_fn = torch.nn.MSELoss(reduction='none')
    scaler = GradScaler()
    
    for epoch in range(num_epochs):
        if verbose:
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        total_train_loss = 0
        total_train_mae = 0
        if verbose:
            train_progress_bar = tqdm(train_dataloader, desc="Training", leave=False)
        else:
            train_progress_bar = train_dataloader
        
        # TRAIN.
        model.train()
        for batch_idx, batch in enumerate(train_progress_bar):
            # UNPACK DATA.
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            sample_weights = batch['sample_weight'].to(device)
            
            # UPDATE MODEL.
            optimizer.zero_grad()
            
            # print('WARNING: Ussing float16')
            # with autocast(device_type='cuda', dtype=torch.float16):
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                _, regression_output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                # loss = loss_fn(regression_output, labels.reshape(-1,1))
                losses = loss_fn(regression_output, labels.reshape(-1,1))
                loss = (losses.T[0] * sample_weights).sum() / sample_weights.sum()

            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # optimizer.step()
            # scheduler.step()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # REPORT METRICS.
            unscaled_labels = label_scaler.inverse_transform(labels.cpu().numpy().reshape(-1, 1))
            unscaled_outputs = label_scaler.inverse_transform(regression_output.float().cpu().detach().numpy().reshape(-1, 1))
            mae = mean_absolute_error(unscaled_outputs, unscaled_labels)
            
            total_train_loss += loss.item()
            total_train_mae += mae
            
            if verbose:
                train_progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mae': f'{mae:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_mae = total_train_mae / len(train_dataloader)
                
        # TEST.
        model.eval()
        total_test_loss = 0
        all_test_predictions = []
        all_test_labels = []
        all_test_ids = []
        with torch.no_grad():
            val_progress = tqdm(test_dataloader, desc="Validation", leave=False)
            
            for batch in val_progress:
                # UNPACK DATA.
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                original_smiles_ids = batch['original_smiles_id']
                
                # FORWARD PASS.
                _, regression_output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # COMPUTE METRICS.
                losses = loss_fn(regression_output, labels.reshape(-1, 1))
                loss = losses.sum()
                
                unscaled_labels = label_scaler.inverse_transform(labels.cpu().numpy().reshape(-1, 1))
                unscaled_outputs = label_scaler.inverse_transform(regression_output.cpu().detach().numpy().reshape(-1, 1))
                mae = mean_absolute_error(unscaled_outputs, unscaled_labels)
                all_test_predictions.extend(unscaled_outputs.T[0])
                all_test_labels.extend(unscaled_labels.T[0])
                all_test_ids.extend(list(original_smiles_ids.numpy()))

                total_test_loss += loss.item()
                
                val_progress.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        avg_test_loss = total_test_loss / len(test_dataloader)
        
        test_results_df = pd.DataFrame({
            'id': all_test_ids,
            'prediction': all_test_predictions,
            'label': all_test_labels,
        })
        test_results_df = test_results_df.groupby('id').agg({'prediction': 'mean', 'label': 'mean'})
        avg_test_mae = mean_absolute_error(test_results_df['prediction'], test_results_df['label'])

        if verbose:
            print(f"Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Train MAE: {avg_train_mae:.4f} | Test MAE: {avg_test_mae:.4f}")
            
    return avg_test_mae, all_test_predictions, all_test_labels

def get_target_weights(csv_path, target_names):
    df = pd.read_csv(csv_path)

    scale_normalization_factors = []
    sample_count_normalization_factors = []
    for target in target_names:
        target_values = df[target].values
        target_values = target_values[~np.isnan(target_values)]

        scale_normalization_factors.append(1 / (max(target_values) - min(target_values)))
        sample_count_normalization_factors.append((1/len(target_values))**0.5)

    scale_normalization_factors = np.array(scale_normalization_factors)
    sample_count_normalization_factors = np.array(sample_count_normalization_factors)

    target_weights = scale_normalization_factors * len(target_names) * sample_count_normalization_factors / sum(sample_count_normalization_factors)

    return target_weights

def prevent_file_handle_exhaustion():
    import os, torch, warnings
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        warnings.warn("Could not set sharing strategy; proceeding with default.")

def optimize_config(
        target_name: str,
        base_model_identifier: str,
        context_pooler_hidden_size: int,
        input_data_path: str,
        extra_data_config_path_for_target: str,
        pretrained_weight_candidates: list[str] | None,
        lr_search_range_center: float,
        n_trials: int = 30,
    ):
    """
    Per-target Optuna tuning.
      - Tunes: pretrained weight path (categorical), batch size (16/32), epoch count (1..3),
               backbone LR (±10x around default), head LR multiplier (0..50x).
      - Saves best config JSON to ./configs/
      - Saves study SQLite DB to ./studies/
    """
    prevent_file_handle_exhaustion()

    os.makedirs("configs", exist_ok=True)
    os.makedirs("studies", exist_ok=True)

    study = optuna.create_study(direction="minimize")

    def objective(trial: optuna.trial.Trial) -> float:
        # --- Search space ---
        # 1) Pretrained weights (optional)
        if pretrained_weight_candidates and len(pretrained_weight_candidates) > 0:
            chosen_weights_path = trial.suggest_categorical("pretrained_weights_path", pretrained_weight_candidates + [None])
        else:
            chosen_weights_path = None

        # 2) Batch size
        train_batch_size = trial.suggest_categorical("train_batch_size", [16, 32])

        # 3) Epoch count
        epoch_count = trial.suggest_int("epoch_count", 1, 3)

        # 4) Backbone LR (±10x around your main default)
        #    We search log-uniform in [default/10, default*10]
        lr_low = max(lr_search_range_center / 10.0, 1e-7)
        lr_high = lr_search_range_center * 10.0
        backbone_lr = trial.suggest_float("backbone_lr", lr_low, lr_high, log=True)

        # 5) Head LR multiplier (0..50); 0 -> epsilon internally
        head_lr_multiplier = trial.suggest_float("head_lr_multiplier", 0.0, 50.0)

        # --- Load extra-data config for this target ---
        with open(extra_data_config_path_for_target, 'r') as f:
            raw_extra = json.load(f)

        # Rebuild the list-of-dicts used by your get_train_test_datasets
        extra_dataset_configs = []
        extra_dataset_count = int(len(raw_extra.keys()) / 5)
        for i in range(extra_dataset_count):
            extra_dataset_configs.append({
                'filepath': raw_extra[f'filepath_{i}'],
                'raw_label_weight': raw_extra[f'raw_label_weight_{i}'],
                'dataset_weight': raw_extra[f'dataset_weight_{i}'],
                'max_error_ratio': raw_extra[f'max_error_ratio_{i}'],
                'purge_extra_train_smiles_overlap': raw_extra[f'purge_extra_train_smiles_overlap_{i}'],
            })

        # --- Datasets & loaders (uses your helper exactly as in main) ---
        test_maes = []
        for fold_id in [0, 1]:
            # CREATE MODEL.
            if chosen_weights_path is not None:
                # Load a 5-target state dict, then replace head for single-target finetune
                model = BertRegressor(
                    base_model_identifier, 
                    context_pooler_kwargs={
                        'hidden_size': context_pooler_hidden_size,
                        'dropout_prob': 0.144,
                        'activation_name': 'gelu'
                    },
                    target_count = 5 if ('polyOne' not in chosen_weights_path) else 37
                )
                state = torch.load(chosen_weights_path, map_location="cpu")
                model.load_state_dict(state, strict=False)  # tolerate head mismatch
                # Replace output head for single target finetuning
                model.output = torch.nn.Linear(context_pooler_hidden_size, 1)
                model = model.to('cuda')
            else:
                model = BertRegressor(
                    base_model_identifier, 
                    context_pooler_kwargs={
                        'hidden_size': context_pooler_hidden_size,
                        'dropout_prob': 0.144,
                        'activation_name': 'gelu'
                    },
                    target_count=1
                ).to('cuda')

            # PREPARE DATALOADERS
            train_dataset, test_dataset, label_scaler = get_train_test_datasets(
                csv_path=input_data_path, 
                extra_data_config=extra_dataset_configs,
                model_path=base_model_identifier,
                target_name=target_name,
                fold_count=5,
                fold_id=fold_id,
                train_augmentation_kwargs={},
                test_augmentation_kwargs={'augmentation_strategies': ['enumeration'], 'n_augmentations': 25},
                do_explicit=False
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=train_batch_size,
                shuffle=True,
                num_workers=4,
                persistent_workers=True,
                pin_memory=True
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=train_batch_size * 2,
                shuffle=False,
                num_workers=4,
                persistent_workers=True,
                pin_memory=False
            )

            # --- Train (with per-group OneCycleLR under the hood) ---
            #     Note: learning_rate is BACKBONE LR; head multiplier is separate
            test_mae, _, _ = train_model(
                model=model,
                train_dataloader=train_loader,
                test_dataloader=test_loader,
                label_scaler=label_scaler,
                num_epochs=epoch_count,
                learning_rate=backbone_lr,
                head_lr_multiplier=head_lr_multiplier,
                device='cuda',
                verbose=True
            )
            test_maes.append(test_mae)

        # Optuna minimizes the objective → MAE is the loss
        return float(np.mean(test_maes))

    # Run tuning
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Save best config & print it
    best = study.best_trial
    best_val = study.best_value
    best_params = dict(best.params)

    # Ensure keys exist even if choices were None
    if "pretrained_weights_path" not in best_params:
        best_params["pretrained_weights_path"] = None

    best_config = {
        "base_model_identifier": base_model_identifier,
        "context_pooler_hidden_size": context_pooler_hidden_size,
        "pretrained_weights_path": best_params["pretrained_weights_path"],
        "train_batch_size": best_params["train_batch_size"],
        "epoch_count": best_params["epoch_count"],
        "backbone_lr": best_params["backbone_lr"],
        "head_lr_multiplier": best_params["head_lr_multiplier"],
    }

    base_model_name = base_model_identifier.split("/")[-1].replace(":", "_")
    cfg_name = f"{base_model_name}_{target_name}_{int(best_val * 10000)}.json"
    cfg_path = os.path.join("configs", cfg_name)
    with open(cfg_path, "w") as f:
        json.dump(best_config, f, indent=2)

    print("\n=== Best config ===")
    print(json.dumps(best_config, indent=2))
    print(f"\nSaved best config to: {cfg_path}")

    study_pickle_path = f"studies/{base_model_name}_{target_name}.pkl"
    with open(study_pickle_path, "wb") as f:
        pickle.dump(study, f)

    print(f"Saved study pickle to: {study_pickle_path}")

    df = study.trials_dataframe()
    df.to_csv(f"studies/{base_model_name}_{target_name}_trials.csv", index=False)

def train_and_save_models(
        base_model_identifier = 'DeepChem/ChemBERTa-77M-MTR',
        pretrained_weights_path = None,
        context_pooler_hidden_size = 768,
        epoch_count=2,
        learning_rate=2.8e-4,
        train_batch_size=16,
        fold_count=5,
        input_data_path='data/from_host/train.csv',
        target_names_to_extra_data_config_paths={},
        output_dir_suffix='temp',
        do_explicit: bool = False):
    # CONFIG.
    TARGET_NAMES = ["Tg", "FFV", "Tc", "Density", "Rg"]
    
    # CREATE OUTPUT DIRECTORY.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'models/{timestamp}_{output_dir_suffix}'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # TRAIN MODELS FOR ALL TARGETS.
    all_results = {}
    for target in TARGET_NAMES:
        print(f'\n=== Training {target} model across {fold_count} folds ===')
        
        # TRAIN & TEST FOR EACH FOLD.
        fold_maes = []
        for fold_id in range(fold_count):
            print(f'Training {target} - Fold {fold_id}/{fold_count}')
            
            # CREATE FOLD-SPECIFIC OUTPUT DIRECTORY.
            fold_dir = os.path.join(output_dir, f'fold_{fold_id}')
            os.makedirs(fold_dir, exist_ok=True)
            
            # CREATE MODEL
            if pretrained_weights_path is not None:
                print(f'Loading pretrained weights from: {pretrained_weights_path}')
                model = BertRegressor(
                    base_model_identifier, 
                    context_pooler_kwargs = {
                        'hidden_size': context_pooler_hidden_size, 
                        'dropout_prob': 0.144, 
                        'activation_name': 'gelu'
                    },
                    target_count=5 if ('polyOne' not in pretrained_weights_path) else 37
                )
                model.load_state_dict(torch.load(pretrained_weights_path))
                model.output = torch.nn.Linear(context_pooler_hidden_size, 1)
                model = model.to('cuda')
            else:
                model = BertRegressor(
                    base_model_identifier, 
                    context_pooler_kwargs = {
                        'hidden_size': context_pooler_hidden_size, 
                        'dropout_prob': 0.144, 
                        'activation_name': 'gelu'
                    },
                    target_count=1
                ).to('cuda')

            # GET EXTRA DATA CONFIG.
            extra_data_config_path = target_names_to_extra_data_config_paths[target]
            with open(extra_data_config_path, 'r') as extra_data_config_file:
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
            
            # PREPARE DATALOADERS
            train_dataset, test_dataset, label_scaler = get_train_test_datasets(
                csv_path=input_data_path, 
                extra_data_config=extra_dataset_configs,
                model_path=base_model_identifier,
                target_name=target,
                fold_count=fold_count, 
                fold_id=fold_id, 
                train_augmentation_kwargs={},
                test_augmentation_kwargs={'augmentation_strategies': ['enumeration'], 'n_augmentations': 25},
                do_explicit=do_explicit
            )
            
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=train_batch_size,
                shuffle=True,
                num_workers=4,
                persistent_workers=True,
                pin_memory=True
            )
            
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=train_batch_size*2,
                shuffle=False,
                num_workers=4,
                persistent_workers=True,
                pin_memory=True
            )
            
            # TRAIN & TEST MODEL
            test_mae, oof_preds, oof_labels = train_model(
                model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                label_scaler=label_scaler,
                num_epochs=epoch_count,
                learning_rate=learning_rate,
                device='cuda'
            )
            
            # SAVE RESULTS TO FOLD-SPECIFIC DIRECTORY
            scaler_path = os.path.join(fold_dir, f'scaler_{target}.pkl')
            model_path = os.path.join(fold_dir, f'polymer_bert_v2_{target}.pth')
            oof_preds_path = os.path.join(fold_dir, f'oof_preds_{target}.pkl')
            oof_labels_path = os.path.join(fold_dir, f'oof_labels_{target}.pkl')
            
            joblib.dump(label_scaler, scaler_path)
            joblib.dump(oof_preds, oof_preds_path)
            joblib.dump(oof_labels, oof_labels_path)
            torch.save(model.state_dict(), model_path)
            
            fold_maes.append(test_mae)
            print(f'Fold {fold_id} MAE: {test_mae:.4f}')
        
            # LOG STATS.
            all_results[target] = {
                'fold_maes': fold_maes,
                'mean_mae': np.mean(fold_maes),
                'std_mae': np.std(fold_maes)
            }
        
        print(f'{target} - Mean MAE: {np.mean(fold_maes):.4f} ± {np.std(fold_maes):.4f}')
    
    # CALCULATE wMAE.
    class_weights = get_target_weights(input_data_path, TARGET_NAMES)
    
    target_mean_maes = [all_results[target]['mean_mae'] for target in TARGET_NAMES]
    weighted_mae = np.average(target_mean_maes, weights=class_weights)
    
    # PRINT STATS.
    print('\n' + '='*60)
    print('FINAL RESULTS (Cross-Validation)')
    print('='*60)
    
    for target in TARGET_NAMES:
        result = all_results[target]
        print(f'{target:8s}: {result["mean_mae"]:.4f} ± {result["std_mae"]:.4f} MAE')
    
    print('-'*60)
    print(f'Weighted MAE: {weighted_mae:.4f}')
    
    # SAVE STATS.
    summary_path = os.path.join(output_dir, 'cv_results_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('Cross-Validation Results Summary\n')
        f.write('='*40 + '\n\n')
        f.write(f'Number of folds: {fold_count}\n')
        f.write(f'Targets: {TARGET_NAMES}\n\n')
        
        for target in TARGET_NAMES:
            result = all_results[target]
            f.write(f'{target}:\n')
            f.write(f'  Mean MAE: {result["mean_mae"]:.4f}\n')
            f.write(f'  Std MAE:  {result["std_mae"]:.4f}\n')
            f.write(f'  Fold MAEs: {[f"{mae:.4f}" for mae in result["fold_maes"]]}\n\n')
        
        f.write(f'Weighted MAE: {weighted_mae:.4f}\n')
        f.write(f'Class weights: {dict(zip(TARGET_NAMES, class_weights))}\n')
    
    print(f'\nResults saved to: {output_dir}')
    print(f'Summary saved to: {summary_path}')

def train_and_save_tuned_models(
        target_names_to_config_paths, 
        input_data_path='data/from_host/train.csv', 
        target_names_to_extra_data_config_paths={},
        epoch_count_override=None,
        fold_count=5,
        output_dir_suffix='temp',
        train_augmentation_kwargs={}):
    # CREATE OUTPUT DIRECTORY.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'models/{timestamp}_{output_dir_suffix}'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")

    # TRAIN MODELS FOR ALL TARGETS.
    all_results = {}
    for target, config_path in target_names_to_config_paths.items():
        print(f'\n=== Training {target} model across 5 folds using config: {config_path} ===')
        
        # LOAD CONFIG.
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        
        base_model_identifier = cfg['base_model_identifier']
        context_pooler_hidden_size = cfg['context_pooler_hidden_size']
        pretrained_weights_path = cfg['pretrained_weights_path']
        epoch_count = cfg['epoch_count'] if (epoch_count_override is None) else epoch_count_override
        learning_rate = cfg['backbone_lr']
        head_lr_multiplier = cfg['head_lr_multiplier']
        train_batch_size = cfg['train_batch_size']
        
        print(f'Config details:\n  Base model: {base_model_identifier}\n  Context pooler size: {context_pooler_hidden_size}\n  Pretrained weights: {pretrained_weights_path}\n  Epochs: {epoch_count}\n  Backbone LR: {learning_rate}\n  Head LR multiplier: {head_lr_multiplier}\n  Train batch size: {train_batch_size}')
        
        # TRAIN & TEST FOR EACH FOLD.
        fold_maes = []
        for fold_id in range(fold_count):
            print(f'Training {target} - Fold {fold_id}/{fold_count}')
            
            # CREATE FOLD-SPECIFIC OUTPUT DIRECTORY.
            fold_dir = os.path.join(output_dir, f'fold_{fold_id}')
            os.makedirs(fold_dir, exist_ok=True)
            
            # CREATE MODEL
            if pretrained_weights_path is not None:
                print(f'Loading pretrained weights from: {pretrained_weights_path}')
                model = BertRegressor(
                    base_model_identifier, 
                    context_pooler_kwargs = {
                        'hidden_size': context_pooler_hidden_size, 
                        'dropout_prob': 0.144, 
                        'activation_name': 'gelu'
                    },
                    target_count=5 if ('polyOne' not in pretrained_weights_path) else 37
                )
                model.load_state_dict(torch.load(pretrained_weights_path))
                model.output = torch.nn.Linear(context_pooler_hidden_size, 1)
                model = model.to('cuda')
            else:
                model = BertRegressor(
                    base_model_identifier, 
                    context_pooler_kwargs = {
                        'hidden_size': context_pooler_hidden_size, 
                        'dropout_prob': 0.144, 
                        'activation_name': 'gelu'
                    },
                    target_count=1
                ).to('cuda')

            # GET EXTRA DATA CONFIG.
            extra_data_config_path = target_names_to_extra_data_config_paths[target]
            with open(extra_data_config_path, 'r') as extra_data_config_file:
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
            
            # PREPARE DATALOADERS
            train_dataset, test_dataset, label_scaler = get_train_test_datasets(
                csv_path=input_data_path, 
                extra_data_config=extra_dataset_configs,
                model_path=base_model_identifier,
                target_name=target,
                fold_count=fold_count, 
                fold_id=fold_id, 
                train_augmentation_kwargs=train_augmentation_kwargs,
                test_augmentation_kwargs={'augmentation_strategies': ['enumeration'], 'n_augmentations': 25},
                do_explicit=False
            )
            
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=train_batch_size,
                shuffle=True,
                num_workers=4,
                persistent_workers=True,
                pin_memory=True
            )
            
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=train_batch_size*2,
                shuffle=False,
                num_workers=4,
                persistent_workers=True,
                pin_memory=True
            )
            
            # TRAIN & TEST MODEL
            test_mae, oof_preds, oof_labels = train_model(
                model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                label_scaler=label_scaler,
                num_epochs=epoch_count,
                learning_rate=learning_rate,
                head_lr_multiplier=head_lr_multiplier,
                device='cuda'
            )
            
            # SAVE RESULTS TO FOLD-SPECIFIC DIRECTORY
            scaler_path = os.path.join(fold_dir, f'scaler_{target}.pkl')
            model_path = os.path.join(fold_dir, f'polymer_bert_v2_{target}.pth')
            oof_preds_path = os.path.join(fold_dir, f'oof_preds_{target}.pkl')
            oof_labels_path = os.path.join(fold_dir, f'oof_labels_{target}.pkl')
            
            joblib.dump(label_scaler, scaler_path)
            joblib.dump(oof_preds, oof_preds_path)
            joblib.dump(oof_labels, oof_labels_path)
            torch.save(model.state_dict(), model_path)
            
            fold_maes.append(test_mae)
            print(f'Fold {fold_id} MAE: {test_mae:.4f}')
        
            # LOG STATS.
            all_results[target] = {
                'fold_maes': fold_maes,
                'mean_mae': np.mean(fold_maes),
                'std_mae': np.std(fold_maes)
            }
        
        print(f'{target} - Mean MAE: {np.mean(fold_maes):.4f} ± {np.std(fold_maes):.4f}')
    
    # CALCULATE wMAE.
    target_names = target_names_to_config_paths.keys()
    class_weights = get_target_weights(input_data_path, target_names)
    
    target_mean_maes = [all_results[target]['mean_mae'] for target in target_names]
    weighted_mae = np.average(target_mean_maes, weights=class_weights)
    
    # PRINT STATS.
    print('\n' + '='*60)
    print('FINAL RESULTS (Cross-Validation)')
    print('='*60)
    
    for target in target_names:
        result = all_results[target]
        print(f'{target:8s}: {result["mean_mae"]:.4f} ± {result["std_mae"]:.4f} MAE')
    
    print('-'*60)
    print(f'Weighted MAE: {weighted_mae:.4f}')
    
    # SAVE STATS.
    summary_path = os.path.join(output_dir, 'cv_results_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('Cross-Validation Results Summary\n')
        f.write('='*40 + '\n\n')
        f.write(f'Number of folds: {fold_count}\n')
        f.write(f'Targets: {target_names}\n\n')
        
        for target in target_names:
            result = all_results[target]
            f.write(f'{target}:\n')
            f.write(f'  Mean MAE: {result["mean_mae"]:.4f}\n')
            f.write(f'  Std MAE:  {result["std_mae"]:.4f}\n')
            f.write(f'  Fold MAEs: {[f"{mae:.4f}" for mae in result["fold_maes"]]}\n\n')
        
        f.write(f'Weighted MAE: {weighted_mae:.4f}\n')
        f.write(f'Class weights: {dict(zip(target_names, class_weights))}\n')
    
    print(f'\nResults saved to: {output_dir}')
    print(f'Summary saved to: {summary_path}')

    
#region Main

def training_main_v1():
    train_and_save_models(
        fold_count=5,

        # base_model_identifier = 'answerdotai/ModernBERT-base',
        # pretrained_weights_path=[
        #     'models/20250819_223828_modern_3epochs_2thresh_rankup/single_split/polymer_bert_rankup.pth', # 0
        #     'models/20250819_223908_modern_3epochs_8thresh_rankup/single_split/polymer_bert_rankup.pth',
        #     'models/20250819_223849_modern_6epochs_2thresh_rankup/single_split/polymer_bert_rankup.pth',
        #     'models/20250826_234138_modern_3epochs_2thresh_polyOne_rankup/single_split/polymer_bert_rankup.pth', # 3
        #     'models/20250827_002753_modern_3epochs_8thresh_polyOne_rankup/single_split/polymer_bert_rankup.pth',
        #     'models/20250827_002906_modern_6epochs_2thresh_polyOne_rankup/single_split/polymer_bert_rankup.pth',
        #     'models/20250827_203349_modern_3epochs_2thresh_mlm_rankup/single_split/polymer_bert_rankup.pth', # 6
        #     'models/20250828_204112_modern_3epochs_2thresh_explicit_rankup/single_split/polymer_bert_rankup.pth',
        #     'models/20250830_161236_modern_3epochs_2thresh_p1_relabeled_rankup/single_split/polymer_bert_rankup.pth', # 8
        #     'models/20250830_161313_modern_3epochs_8thresh_p1_relabeled_rankup/single_split/polymer_bert_rankup.pth',
        #     'models/20250830_161153_modern_6epochs_2thresh_p1_relabeled_rankup/single_split/polymer_bert_rankup.pth',
        #     'models/20250912_203737_modern_3epochs_2thresh_v2.1_rankup/single_split/polymer_bert_rankup.pth', # 11
        #     'models/20250912_203714_modern_3epochs_8thresh_v2.1_rankup/single_split/polymer_bert_rankup.pth',
        #     'models/20250912_203753_modern_6epochs_2thresh_v2.1_rankup/single_split/polymer_bert_rankup.pth',
        # ][11],
        # output_dir_suffix='modern_3epochs_2thresh_finetune_1e_v2.1',
        # # ][8],
        # # output_dir_suffix='modern_3epochs_2thresh_p1_relabeled_finetune_1e',
        # # ][9],
        # # output_dir_suffix='modern_3epochs_8thresh_p1_relabeled_finetune_1e',
        # # ][10],
        # # output_dir_suffix='modern_6epochs_2thresh_p1_relabeled_finetune_1e',
        # context_pooler_hidden_size = 768,
        # epoch_count=1,
        # learning_rate=2e-5,
        # train_batch_size=8,

        base_model_identifier = 'kuelumbus/polyBERT',
        pretrained_weights_path=[
            'models/20250820_081902_poly_3epochs_2thresh_rankup/single_split/polymer_bert_rankup.pth', # 0
            'models/20250820_081658_poly_3epochs_8thresh_rankup/single_split/polymer_bert_rankup.pth',
            'models/20250820_081728_poly_6epochs_2thresh_rankup/single_split/polymer_bert_rankup.pth',
            'models/20250826_234214_poly_3epochs_2thresh_polyOne_rankup/single_split/polymer_bert_rankup.pth', # 3
            'models/20250827_002536_poly_3epochs_8thresh_polyOne_rankup/single_split/polymer_bert_rankup.pth',
            'models/20250827_002318_poly_6epochs_2thresh_polyOne_rankup/single_split/polymer_bert_rankup.pth',
            'models/20250828_204203_poly_3epochs_2thresh_explicit_rankup/single_split/polymer_bert_rankup.pth', # 6
            'models/20250912_233606_poly_3epochs_2thresh_v2.1_rankup/single_split/polymer_bert_rankup.pth',
            'models/20250913_010646_poly_3epochs_2thresh_v2_rankup/single_split/polymer_bert_rankup.pth',
        # ][7],
        # output_dir_suffix='poly_3epochs_2thresh_finetune_1e_v2.1',
        ][8],
        output_dir_suffix='poly_3epochs_2thresh_finetune_1e_v2',
        # ][4],
        # output_dir_suffix='poly_3epochs_8thresh_polyOne_finetune_2e',
        # ][5],
        # output_dir_suffix='poly_6epochs_2thresh_polyOne_finetune_2e',
        context_pooler_hidden_size = 600,
        epoch_count=1,
        # train_batch_size=8,
        # epoch_count=2, # 2 epochs, batch size 16 scored [0.0591, 0.0585, 0.0595] with configs [3, 4, 5] respectively
        # train_batch_size=8,
        # epoch_count=4,
        train_batch_size=16,
        learning_rate=1.4e-5,

        input_data_path='data/from_host/train.csv',
        target_names_to_extra_data_config_paths={
            "Tg": 'configs/avg_data_Tg.json',
            "FFV": 'configs/avg_data_FFV.json',
            "Tc": 'configs/avg_data_Tc.json',
            "Density": "configs/avg_data_Density.json",
            "Rg": "configs/avg_data_Rg.json"
        }
    )

def training_main_v2():
    train_and_save_tuned_models(
        # train_augmentation_kwargs={'augmentation_strategies': ['enumeration'], 'n_augmentations': 7},
        # train_augmentation_kwargs={'augmentation_strategies': ['enumeration'], 'n_augmentations': 10},
        # train_augmentation_kwargs={'augmentation_strategies': ['enumeration'], 'n_augmentations': 15},

        # target_names_to_config_paths={
        #     "Tg": 'configs/polyBERT_Tg_409919.json',
        #     "FFV": 'configs/polyBERT_FFV_45.json',
        #     "Tc": 'configs/polyBERT_Tc_204.json',
        #     "Density": "configs/polyBERT_Density_217.json",
        #     "Rg": "configs/polyBERT_Rg_11048.json"
        # },
        # output_dir_suffix='polyBERT_tuned_v1',

        # target_names_to_config_paths={
        #     "Tg": 'configs/polyBERT_Tg_409919_v2.1.json',
        #     "FFV": 'configs/polyBERT_FFV_45_v2.1.json',
        #     "Tc": 'configs/polyBERT_Tc_204_v2.1.json',
        #     "Density": "configs/polyBERT_Density_217_v2.1.json",
        #     "Rg": "configs/polyBERT_Rg_11048_v2.1.json"
        # },
        # output_dir_suffix='polyBERT_tuned_v2.1',

        target_names_to_config_paths={
            "Tg": 'configs/ModernBERT-base_Tg_411250_v3.json',
            "FFV": 'configs/ModernBERT-base_FFV_37_v3.json',
            "Tc": 'configs/ModernBERT-base_Tc_210_v3.json',
            "Density": "configs/ModernBERT-base_Density_197_v3.json",
            "Rg": "configs/ModernBERT-base_Rg_10306_v3.json"
        },
        output_dir_suffix='modern_tuned_v3',
        
        # target_names_to_config_paths={
        #     "Tg": 'configs/codebert-base_Tg_398570_v3.json',
        #     "FFV": 'configs/codebert-base_FFV_42_v3.json',
        #     "Tc": 'configs/codebert-base_Tc_207_v3.json',
        #     "Density": "configs/codebert-base_Density_211_v3.json",
        #     "Rg": "configs/codebert-base_Rg_10878_v3.json"
        # },
        # output_dir_suffix='codebert_tuned_v3',

        input_data_path='data/from_host/train.csv',
        target_names_to_extra_data_config_paths={
            "Tg": 'configs/avg_data_Tg.json',
            "FFV": 'configs/avg_data_FFV.json',
            "Tc": 'configs/avg_data_Tc.json',
            "Density": "configs/avg_data_Density.json",
            "Rg": "configs/avg_data_Rg.json"
        },
        fold_count=1
    )

def tuning_main():
    # base_model_identifier = 'kuelumbus/polyBERT'
    # context_pooler_hidden_size = 600
    # lr_search_range_center = 1.4e-5
    # pretrained_weight_candidates = [
    #     'models/20250820_081902_poly_3epochs_2thresh_rankup/single_split/polymer_bert_rankup.pth',
    #     'models/20250820_081658_poly_3epochs_8thresh_rankup/single_split/polymer_bert_rankup.pth',
    #     'models/20250820_081728_poly_6epochs_2thresh_rankup/single_split/polymer_bert_rankup.pth',
    # ]

    # base_model_identifier = 'answerdotai/ModernBERT-base'
    # context_pooler_hidden_size = 768
    # lr_search_range_center = 2e-5
    # pretrained_weight_candidates = [
    #     'models/20250819_223828_modern_3epochs_2thresh_rankup/single_split/polymer_bert_rankup.pth',
    #     'models/20250819_223908_modern_3epochs_8thresh_rankup/single_split/polymer_bert_rankup.pth',
    #     'models/20250819_223849_modern_6epochs_2thresh_rankup/single_split/polymer_bert_rankup.pth',
    # ]

    base_model_identifier = 'microsoft/codebert-base'
    context_pooler_hidden_size = 768
    lr_search_range_center = 2e-5
    pretrained_weight_candidates = [
        'models/20250907_015924_code_3epochs_2thresh_rankup/single_split/polymer_bert_rankup.pth',
        'models/20250907_020020_code_3epochs_8thresh_rankup/single_split/polymer_bert_rankup.pth',
        'models/20250907_020039_code_6epochs_2thresh_rankup/single_split/polymer_bert_rankup.pth',
    ]

    input_data_path = 'data/from_host/train.csv'
    target_names_to_extra_data_config_paths = {
        # "Tg": 'configs/avg_data_Tg.json',
        # "FFV": 'configs/avg_data_FFV.json',
        # "Tc": 'configs/avg_data_Tc.json',
        # "Density": "configs/avg_data_Density.json",
        "Rg": "configs/avg_data_Rg.json"
    }

    # Run per-target tuning
    for target_name, extra_cfg in target_names_to_extra_data_config_paths.items():
        print(f"\n##### Tuning target: {target_name} #####")
        optimize_config(
            target_name = target_name,
            base_model_identifier = base_model_identifier,
            context_pooler_hidden_size = context_pooler_hidden_size,
            input_data_path = input_data_path,
            extra_data_config_path_for_target = extra_cfg,
            pretrained_weight_candidates = pretrained_weight_candidates,
            lr_search_range_center = lr_search_range_center,
            n_trials = 20,
        )

'''
polyBERT (PI1M)
    baseline settings (3 pretrain epochs with thresh=0.2, 1 finetuning epoch):
        Results saved to: models/20250820_220751_poly_3epochs_2thresh_finetune      <-- 0.062 LB
        Tg      : 39.8888 ± 2.9792 MAE
        FFV     : 0.0048 ± 0.0002 MAE
        Tc      : 0.0213 ± 0.0013 MAE
        Density : 0.0199 ± 0.0051 MAE
        Rg      : 1.1672 ± 0.0744 MAE
        Weighted MAE: 0.0525
    Increased pretrain epochs to 6 (thresh=0.2, 1 finetuning epoch):
        Results saved to: models/20250820_220946_poly_6epochs_2thresh_finetune_1e   <-- 0.062 LB
        Tg      : 38.7918 ± 2.8221 MAE
        FFV     : 0.0047 ± 0.0001 MAE
        Tc      : 0.0211 ± 0.0016 MAE
        Density : 0.0198 ± 0.0037 MAE
        Rg      : 1.1313 ± 0.0590 MAE
        Weighted MAE: 0.0513
    Increased thresh to 0.8 (3 pretrain epochs, 1 finetuning epoch):
        Results saved to: models/20250820_220814_poly_3epochs_8thresh_finetune_1e   <-- 0.061 LB
        Tg      : 39.5537 ± 2.1729 MAE
        FFV     : 0.0047 ± 0.0002 MAE
        Tc      : 0.0208 ± 0.0013 MAE
        Density : 0.0196 ± 0.0042 MAE
        Rg      : 1.1800 ± 0.0565 MAE
        Weighted MAE: 0.0521
    Tuned with Optuna (20 trials per target):
        Results saved to: models/20250823_160519_polyBERT_tuned     <-- 0.061 LB
        Tg      : 39.7064 ± 2.0301 MAE
        FFV     : 0.0045 ± 0.0001 MAE
        Tc      : 0.0212 ± 0.0011 MAE
        Density : 0.0189 ± 0.0036 MAE
        Rg      : 1.1136 ± 0.0693 MAE
        Weighted MAE: 0.0514
    Explicit H & bonds (3 pretrain epochs with thresh=0.2, 1 finetuning epoch):
        Results saved to: models/20250829_002946_poly_3epochs_2thresh_explicit_finetune_1e
        Tg      : 40.3015 ± 2.8381 MAE
        FFV     : 0.0054 ± 0.0002 MAE
        Tc      : 0.0214 ± 0.0015 MAE
        Density : 0.0204 ± 0.0041 MAE
        Rg      : 1.2132 ± 0.1211 MAE
        Weighted MAE: 0.0535
polyBERT (PI1M v2)
    Tuned with prev optuna
        Results saved to: models/20250913_024900_polyBERT_tuned_v2
        Tg      : 39.1545 ± 1.0329 MAE
        FFV     : 0.0044 ± 0.0001 MAE
        Tc      : 0.0213 ± 0.0013 MAE
        Density : 0.0197 ± 0.0045 MAE
        Rg      : 1.1677 ± 0.0780 MAE
        Weighted MAE: 0.0520
    Baseline:
        Tg      : 39.5311 ± 2.2520 MAE
        FFV     : 0.0048 ± 0.0002 MAE
        Tc      : 0.0207 ± 0.0010 MAE
        Density : 0.0202 ± 0.0044 MAE
        Rg      : 1.1963 ± 0.0751 MAE
        Weighted MAE: 0.0524
Results saved to: models/20250913_055034_poly_3epochs_2thresh_finetune_1e_v2
polyBERT (PI1M v2.1)
    Tuned with prev optuna
        Results saved to: models/20250913_024947_polyBERT_tuned_v2.1
        Tg      : 39.2915 ± 1.5221 MAE
        FFV     : 0.0044 ± 0.0002 MAE
        Tc      : 0.0209 ± 0.0009 MAE
        Density : 0.0198 ± 0.0043 MAE
        Rg      : 1.1616 ± 0.0662 MAE
        Weighted MAE: 0.0518
    Baseline:
        Results saved to: models/20250913_055034_poly_3epochs_2thresh_finetune_1e_v2
        Tg      : 39.5311 ± 2.2520 MAE
        FFV     : 0.0048 ± 0.0002 MAE
        Tc      : 0.0207 ± 0.0010 MAE
        Density : 0.0202 ± 0.0044 MAE
        Rg      : 1.1963 ± 0.0751 MAE
        Weighted MAE: 0.0524
    Tuned with prev optuna, removed all but "enumeration" augmentation strategy
        Results saved to: models/20250913_055810_polyBERT_tuned_v2.1
        Tg      : 39.3482 ± 1.7873 MAE
        FFV     : 0.0044 ± 0.0001 MAE
        Tc      : 0.0208 ± 0.0009 MAE
        Density : 0.0198 ± 0.0042 MAE
        Rg      : 1.1569 ± 0.0879 MAE
        Weighted MAE: 0.0517
    7x aug (otherwise same as prev) -> 0.0520
    15x aug (otherwise same as prev) -> 0.0522
polyBERT (PI1M v2.2)
    Tuned with prev optuna, removed all but "enumeration" augmentation strategy
        Results saved to: models/20250913_132507_polyBERT_tuned_v2.2
        Tg      : 39.8146 ± 1.4608 MAE
        FFV     : 0.0044 ± 0.0001 MAE
        Tc      : 0.0207 ± 0.0011 MAE
        Density : 0.0198 ± 0.0041 MAE
        Rg      : 1.1899 ± 0.0766 MAE
        Weighted MAE: 0.0523

        
ModernBERT (PI1M)
    baseline settings (3 pretrain epochs with thresh=0.2, 1 finetuning epoch):
        Results saved to: models/20250821_011341_modern_3epochs_2thresh_finetune_1e <-- 0.059 LB
        Tg      : 39.6794 ± 2.6910 MAE
        FFV     : 0.0042 ± 0.0001 MAE
        Tc      : 0.0211 ± 0.0012 MAE
        Density : 0.0182 ± 0.0034 MAE
        Rg      : 1.1638 ± 0.1048 MAE
        Weighted MAE: 0.0517
    Increased pretrain epochs to 6 (thresh=0.2, 1 finetuning epoch):
        Results saved to: models/20250821_194729_modern_6epochs_2thresh_finetune_1e <-- 0.060 LB
        Tg      : 39.2115 ± 2.7007 MAE
        FFV     : 0.0041 ± 0.0001 MAE
        Tc      : 0.0210 ± 0.0009 MAE
        Density : 0.0182 ± 0.0027 MAE
        Rg      : 1.0979 ± 0.0356 MAE
        Weighted MAE: 0.0506
    Increased thresh to 0.8 (3 pretrain epochs, 1 finetuning epoch):
        Results saved to: models/20250821_080411_modern_3epochs_8thresh_finetune_1e <-- 0.061 LB
        Tg      : 40.9494 ± 2.6522 MAE
        FFV     : 0.0048 ± 0.0002 MAE
        Tc      : 0.0221 ± 0.0011 MAE
        Density : 0.0195 ± 0.0038 MAE
        Rg      : 1.2213 ± 0.0794 MAE
        Weighted MAE: 0.0540
    Tuned with Optuna (20 trials per target):
        Results saved to: models/20250823_163627_modern_tuned    <-- 0.060 LB
        Tg      : 39.4444 ± 2.2389 MAE
        FFV     : 0.0037 ± 0.0001 MAE
        Tc      : 0.0209 ± 0.0010 MAE
        Density : 0.0173 ± 0.0029 MAE
        Rg      : 1.0964 ± 0.0851 MAE
        Weighted MAE: 0.0503
    Pretrained with MLM objective (3 pretrain epochs with thresh=0.2, 1 finetuning epoch):
        Results saved to: models/20250827_214805_modern_3epochs_2thresh_mlm_finetune_1e
        Tg      : 40.0207 ± 3.4903 MAE
        FFV     : 0.0042 ± 0.0001 MAE
        Tc      : 0.0217 ± 0.0013 MAE
        Density : 0.0183 ± 0.0042 MAE
        Rg      : 1.1609 ± 0.0590 MAE
        Weighted MAE: 0.0522
    Explicit H & bonds (3 pretrain epochs with thresh=0.2, 1 finetuning epoch):
        Results saved to: models/20250828_231617_modern_3epochs_2thresh_explicit_finetune_1e
        Tg      : 39.8317 ± 3.2304 MAE
        FFV     : 0.0041 ± 0.0001 MAE
        Tc      : 0.0208 ± 0.0011 MAE
        Density : 0.0178 ± 0.0033 MAE
        Rg      : 1.1050 ± 0.0905 MAE
        Weighted MAE: 0.0508
ModernBERT (PI1M V2.1)
    Tuned with Optuna:
        Results saved to: models/20250912_233517_modern_tuned
        Tg      : 39.1784 ± 2.1167 MAE
        FFV     : 0.0037 ± 0.0000 MAE
        Tc      : 0.0200 ± 0.0013 MAE
        Density : 0.0168 ± 0.0030 MAE
        Rg      : 1.1188 ± 0.0461 MAE
        Weighted MAE: 0.0498
    Baseline settings:
        Results saved to: models/20250913_025240_modern_3epochs_2thresh_finetune_1e_v2.1
        Tg      : 39.2935 ± 3.1801 MAE
        FFV     : 0.0041 ± 0.0001 MAE
        Tc      : 0.0206 ± 0.0011 MAE
        Density : 0.0173 ± 0.0031 MAE
        Rg      : 1.1321 ± 0.0879 MAE
        Weighted MAE: 0.0505
    Tuned with prev optuna, removed all but "enumeration" augmentation strategy
        Results saved to: models/20250913_092729_modern_tuned_v2.1
        Tg      : 38.9266 ± 2.2490 MAE
        FFV     : 0.0037 ± 0.0000 MAE
        Tc      : 0.0204 ± 0.0012 MAE
        Density : 0.0167 ± 0.0030 MAE
        Rg      : 1.1067 ± 0.0599 MAE
        Weighted MAE: 0.0498
ModernBERT (PI1M V2.2)
    Tuned with prev optuna, removed all but "enumeration" augmentation strategy, 10x aug:
        Results saved to: models/20250913_130026_modern_tuned_v2.2
        Tg      : 39.3939 ± 1.3940 MAE
        FFV     : 0.0038 ± 0.0000 MAE
        Tc      : 0.0201 ± 0.0011 MAE
        Density : 0.0171 ± 0.0029 MAE
        Rg      : 1.1205 ± 0.0530 MAE
        Weighted MAE: 0.0501 (0.0498 rerun)
    Tuned with prev optuna, removed all but "enumeration" augmentation strategy, 15x aug:
        Results saved to: models/20250913_151251_modern_tuned_v2.2
        Tg      : 39.8206 ± 2.9826 MAE
        FFV     : 0.0037 ± 0.0000 MAE
        Tc      : 0.0203 ± 0.0008 MAE
        Density : 0.0173 ± 0.0033 MAE
        Rg      : 1.0828 ± 0.0943 MAE
        Weighted MAE: 0.0500


CodeBERT (PI1M V1):
    Tuned with prev optuna:
        Tg      : 38.2352 ± 2.4773 MAE
        FFV     : 0.0043 ± 0.0000 MAE
        Tc      : 0.0200 ± 0.0011 MAE
        Density : 0.0191 ± 0.0036 MAE
        Rg      : 1.1138 ± 0.0543 MAE
        Weighted MAE: 0.0500
CodeBERT (PI1M V2):
    Tuned with prev optuna:
        Tg      : 39.0501 ± 2.4172 MAE
        FFV     : 0.0044 ± 0.0001 MAE
        Tc      : 0.0197 ± 0.0013 MAE
        Density : 0.0181 ± 0.0028 MAE
        Rg      : 1.0895 ± 0.0696 MAE
        Weighted MAE: 0.0497
CodeBERT (PI1M V2.1):
    Tuned with prev optuna:
        Results saved to: models/20250912_203534_codebert_tuned
        Tg      : 38.0349 ± 2.4716 MAE
        FFV     : 0.0043 ± 0.0001 MAE
        Tc      : 0.0194 ± 0.0010 MAE
        Density : 0.0187 ± 0.0032 MAE
        Rg      : 1.1230 ± 0.0830 MAE
        Weighted MAE: 0.0496
    Tuned with prev optuna, removed all but "enumeration" augmentation strategy
        Results saved to: models/20250913_075246_codebert_tuned_v2.1
        Tg      : 37.8600 ± 2.5121 MAE
        FFV     : 0.0043 ± 0.0001 MAE
        Tc      : 0.0196 ± 0.0010 MAE
        Density : 0.0186 ± 0.0029 MAE
        Rg      : 1.1094 ± 0.0897 MAE
        Weighted MAE: 0.0494
CodeBERT (PI1M V2.2)
    Tuned with prev optuna, enum-only augmentation (10x):
        models/20250913_114530_codebert_tuned_v2.2
        Tg      : 37.9102 ± 1.6969 MAE
        FFV     : 0.0044 ± 0.0001 MAE
        Tc      : 0.0194 ± 0.0013 MAE
        Density : 0.0186 ± 0.0032 MAE
        Rg      : 1.0901 ± 0.0962 MAE
        Weighted MAE: 0.0491
    7x aug (otherwise same as prev) -> 0.0493
    15x aug (otherwise same as prev) -> 0.0491
        
ModernBERT (polyOne relabeled)
    baseline settings (3 pretrain epochs with thresh=0.2, 1 finetuning epoch):
        Results saved to: models/20250830_201840_modern_3epochs_2thresh_p1_relabeled_finetune_1e
        Tg      : 40.3015 ± 2.9888 MAE
        FFV     : 0.0041 ± 0.0001 MAE
        Tc      : 0.0214 ± 0.0010 MAE
        Density : 0.0185 ± 0.0039 MAE
        Rg      : 1.1468 ± 0.0827 MAE
        Weighted MAE: 0.0520
    Increased pretrain epochs to 6 (thresh=0.2, 1 finetuning epoch):
        Results saved to: models/20250830_202011_modern_6epochs_2thresh_p1_relabeled_finetune_1e
        Tg      : 39.6226 ± 3.0916 MAE
        FFV     : 0.0039 ± 0.0001 MAE
        Tc      : 0.0209 ± 0.0017 MAE
        Density : 0.0177 ± 0.0032 MAE
        Rg      : 1.1016 ± 0.0932 MAE
        Weighted MAE: 0.0506
    Increased thresh to 0.8 (3 pretrain epochs, 1 finetuning epoch):
        Results saved to: models/20250830_201940_modern_3epochs_8thresh_p1_relabeled_finetune_1e
        Tg      : 41.5989 ± 3.5487 MAE
        FFV     : 0.0046 ± 0.0002 MAE
        Tc      : 0.0223 ± 0.0011 MAE
        Density : 0.0202 ± 0.0033 MAE
        Rg      : 1.2515 ± 0.0936 MAE
        Weighted MAE: 0.0550

'''
if __name__ == '__main__':
    # tuning_main()
    # training_main_v1()
    training_main_v2()