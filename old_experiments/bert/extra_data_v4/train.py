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
        learning_rate=2e-5, 
        device='cuda',
        verbose=True):
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, fused=True)
    total_steps = len(train_dataloader) * num_epochs
    # scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps)
    scheduler = OneCycleLR(
        optimizer=optimizer,
        max_lr=learning_rate,
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
                regression_output = model(
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
                regression_output = model(
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

def get_class_weights(csv_path, target_names):
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

    class_weights = scale_normalization_factors * len(target_names) * sample_count_normalization_factors / sum(sample_count_normalization_factors)

    return class_weights

def optimize_learning_rate_and_epoch_count(
    base_model_identifier: str,
    context_pooler_hidden_size: int,
    number_of_trials: int,
    training_csv_path: str = 'data/from_host/train.csv',
    target_names: list[str] = ["Tg", "FFV", "Tc", "Density", "Rg"],
    fold_count: int = 5,
    batch_size: int = 16,
    max_epochs: int = 25,
    eval_batch_size: int = 32,
    device: str = "cuda",
    target_names_to_extra_data_config_paths={}
):
    """
    Tune learning rate and epoch count to minimize weighted MAE across targets.
    """

    # Precompute class weights once
    class_weights = get_class_weights(training_csv_path, target_names)

    def objective(trial: optuna.Trial) -> float:
        # deberta-v3-large
        learning_rate = trial.suggest_float("learning_rate", 2e-6, 4e-5, log=True)
        num_epochs = trial.suggest_int("num_epochs", 1, max_epochs)

        # # ModernBERT-large
        # learning_rate = trial.suggest_float("learning_rate", 4e-6, 4e-5, log=True)
        # num_epochs = trial.suggest_int("num_epochs", 1, max_epochs)

        # polyBERT: {'learning_rate': 2.38740929383219e-05, 'num_epochs': 8}
        # learning_rate = trial.suggest_float("learning_rate", 5e-6, 8e-5, log=True)
        # num_epochs = trial.suggest_int("num_epochs", 6, 10)

        # chemBERTa
        # • learning_rate: 7.659572276291352e-05
        # • num_epochs: 3
        # learning_rate = trial.suggest_float("learning_rate", 2e-5, 3.2e-4, log=True)
        # num_epochs = trial.suggest_int("num_epochs", 1, 5)

        # Warm-start models: one per fold, instantiated once per trial
        models_per_fold = [
            BertRegressor(
                base_model_identifier,
                context_pooler_kwargs={
                    "hidden_size": context_pooler_hidden_size,
                    "dropout_prob": 0.144,
                    "activation_name": "gelu",
                }
            ).to(device)
            for _ in range(fold_count)
        ]

        per_target_mean_maes = []

        for target in target_names:
            fold_maes = []

            for fold_id in range(fold_count):
                print(f'Training fold {fold_id} for target {target}')
                
                # GET MODEL.
                model = models_per_fold[fold_id]
                
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
                
                # PREPARE DATALOADERS.
                train_dataset, val_dataset, label_scaler = get_train_test_datasets(
                    csv_path=training_csv_path,
                    extra_data_config=extra_dataset_configs,
                    model_path=base_model_identifier,
                    target_name=target,
                    fold_count=fold_count,
                    fold_id=fold_id,
                    train_augmentation_kwargs={},
                    test_augmentation_kwargs={'augmentation_strategies': ['enumeration'], 'n_augmentations': 25},
                )

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True,
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=eval_batch_size,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True,
                )

                # TRAIN & EVALUATE MODEL.
                fold_mae, _, _ = train_model(
                    model=model,
                    train_dataloader=train_loader,
                    test_dataloader=val_loader,
                    label_scaler=label_scaler,
                    num_epochs=num_epochs,
                    learning_rate=learning_rate,
                    device=device,
                    verbose=False
                )
                fold_maes.append(fold_mae)

            per_target_mean_maes.append(np.mean(fold_maes))

        # Compute weighted MAE across targets
        weighted_mae = float(np.average(per_target_mean_maes, weights=class_weights))
        return weighted_mae

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler()
    )
    study.optimize(objective, n_trials=number_of_trials)

    # Report
    print(f"\nBest weighted MAE: {study.best_value:.4f}")
    print("Optimal hyperparameters:")
    for name, val in study.best_params.items():
        print(f"  • {name}: {val}")

    # Save study for later
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_filename = f"optuna_study_{timestamp}.pkl"
    with open(study_filename, "wb") as f:
        pickle.dump(study, f)
    print(f"Study object saved to {study_filename}")


def train_and_save_models(
        base_model_identifier = 'DeepChem/ChemBERTa-77M-MTR',
        context_pooler_hidden_size = 768,
        warmup_epoch_count=0,
        epoch_count=2,
        learning_rate=2.8e-4,
        train_batch_size=16,
        fold_count=5,
        input_data_path='data/from_host/train.csv',
        target_names_to_extra_data_config_paths={},
        train_augmentation_kwargs={},
        output_dir_suffix='temp'):
    # CONFIG.
    TARGET_NAMES = ["Tg", "FFV", "Tc", "Density", "Rg"]
    
    # CREATE OUTPUT DIRECTORY.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'models/{timestamp}_{output_dir_suffix}'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # TRAIN MODELS FOR ALL TARGETS.
    all_results = {}
    models = [
        BertRegressor(
            base_model_identifier, 
            context_pooler_kwargs = {
                'hidden_size': context_pooler_hidden_size, 
                'dropout_prob': 0.144, 
                'activation_name': 'gelu'
            }) 
        for _ 
        in range(fold_count)
    ]
    for warmup in [True, False]:
        if warmup and (warmup_epoch_count == 0):
            continue

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
                # model = BertRegressor(base_model_identifier)
                model = models[fold_id]

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
                    num_epochs=epoch_count if not warmup else warmup_epoch_count,
                    learning_rate=learning_rate,
                    device='cuda'
                )
                
                # SAVE RESULTS TO FOLD-SPECIFIC DIRECTORY
                if not warmup:
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
            if not warmup:
                all_results[target] = {
                    'fold_maes': fold_maes,
                    'mean_mae': np.mean(fold_maes),
                    'std_mae': np.std(fold_maes)
                }
        
        print(f'{target} - Mean MAE: {np.mean(fold_maes):.4f} ± {np.std(fold_maes):.4f}')
    
    # CALCULATE wMAE.
    class_weights = get_class_weights(input_data_path, TARGET_NAMES)
    
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

'''
ChemBERTa:
    Batch size 8: 0.0641, 0.0634, 0.0635 <-- Best
    Batch size 16: 0.0646, 0.0639, 0.0652
    Batch size 32: 0.0665
ModernBERT-base: 
    Batch size 8 (float16): 0.0584 <-- Best
    Batch size 8 (bfloat16): 0.0588
    Batch size 16: 0.0592 
    Batch size 32: 0.0592
polyBERT: 
    Batch size 8: 0.0604, 0.0601
    Batch size 16: 0.0593
    Batch size 32: 0.0592 <-- Best
ModernBERT-large:
    Batch size 8: 0.0587    <-- Best
    Batch size 16: 0.0593
codebert-base:
    Batch size 8: 0.0587
'''
if __name__ == '__main__':
    # print(get_class_weights('data/from_host/train.csv', ["Tg", "FFV", "Tc", "Density", "Rg"]))

    #'''
    train_and_save_models(
        # base_model_identifier = 'DeepChem/ChemBERTa-77M-MTR',
        # context_pooler_hidden_size = 384,
        # warmup_epoch_count=1,
        # epoch_count=1,
        # learning_rate=2.8e-4,
        # train_batch_size=8,
        # output_dir_suffix='chem_et_8',

        # base_model_identifier = 'answerdotai/ModernBERT-base',
        # context_pooler_hidden_size = 768,
        # warmup_epoch_count=0,
        # epoch_count=3,
        # learning_rate=2e-5,
        # train_batch_size=8,
        # output_dir_suffix='modern_et_8',

        # base_model_identifier = 'kuelumbus/polyBERT',
        # context_pooler_hidden_size = 600,
        # warmup_epoch_count=0,
        # epoch_count=8,
        # learning_rate=1.4e-5,
        # train_batch_size=32,
        # output_dir_suffix='poly_et_32',

        # base_model_identifier = 'microsoft/codebert-base',
        # context_pooler_hidden_size = 768,
        # warmup_epoch_count=0,
        # epoch_count=2,
        # learning_rate=1.82e-5,
        # train_batch_size=8,
        # output_dir_suffix='codebert_et_8',

        base_model_identifier = 'answerdotai/ModernBERT-large',
        context_pooler_hidden_size = 1024,
        warmup_epoch_count=0,
        epoch_count=1,
        learning_rate=3.6e-5,
        train_batch_size=8,
        train_augmentation_kwargs={},
        output_dir_suffix='modern_large',

        # base_model_identifier = 'microsoft/deberta-v3-large',
        # context_pooler_hidden_size = 1024,
        # warmup_epoch_count=0,
        # epoch_count=1,
        # learning_rate=9e-6,
        # train_batch_size=8,
        # train_augmentation_kwargs={'augmentation_strategies': ['enumeration'], 'n_augmentations': 20}, # 0.0608 with default, 0.0658 with 5 augmentations, 0.0606 with 20 augmentations
        # output_dir_suffix='deberta_large',

        fold_count=1,
        input_data_path='data/from_host/train.csv',
        target_names_to_extra_data_config_paths={
            "Tg": 'configs/avg_data_Tg.json',
            "FFV": 'configs/avg_data_FFV.json',
            "Tc": 'configs/avg_data_Tc.json',
            "Density": "configs/avg_data_Density.json",
            "Rg": "configs/avg_data_Rg.json"
        }
    )
    #'''

    '''
    optimize_learning_rate_and_epoch_count(
        # base_model_identifier = 'microsoft/deberta-v3-large',
        # context_pooler_hidden_size = 1024,
        # batch_size=8,
        # eval_batch_size=16,
        # max_epochs = 6,
        base_model_identifier = 'microsoft/codebert-base',
        context_pooler_hidden_size = 768,
        batch_size=8,
        eval_batch_size=16,
        max_epochs = 6,
        # base_model_identifier = 'answerdotai/ModernBERT-large',
        # context_pooler_hidden_size = 1024,
        # max_epochs = 6,
        # base_model_identifier = 'answerdotai/ModernBERT-base',
        # context_pooler_hidden_size = 768,
        # max_epochs = 8,
        # base_model_identifier = 'kuelumbus/polyBERT',
        # context_pooler_hidden_size = 600,
        # max_epochs=20,
        # base_model_identifier = 'DeepChem/ChemBERTa-77M-MTR',
        # context_pooler_hidden_size = 384,
        # max_epochs=25,
        number_of_trials = 6,
        target_names_to_extra_data_config_paths={
            "Tg": 'configs/avg_data_Tg.json',
            "FFV": 'configs/avg_data_FFV.json',
            "Tc": 'configs/avg_data_Tc.json',
            "Density": "configs/avg_data_Density.json",
            "Rg": "configs/avg_data_Rg.json"
        }
    )
    # '''