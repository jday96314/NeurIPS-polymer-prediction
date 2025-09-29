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

from dataset import SMILESDataset, get_train_test_datasets
from polybert_regressor import BertRegressor
from torch.utils.data import DataLoader

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


    loss_fn = torch.nn.MSELoss()

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
            
            # UPDATE MODEL.
            optimizer.zero_grad()
            
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                regression_output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                loss = loss_fn(regression_output, labels.reshape(-1,1))

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
        total_test_mae = 0
        all_test_predictions = []
        all_test_labels = []
        with torch.no_grad():
            val_progress = tqdm(test_dataloader, desc="Validation", leave=False)
            
            for batch in val_progress:
                # UNPACK DATA.
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # FORWARD PASS.
                regression_output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # COMPUTE METRICS.
                loss = loss_fn(regression_output, labels.reshape(-1, 1))
                
                unscaled_labels = label_scaler.inverse_transform(labels.cpu().numpy().reshape(-1, 1))
                unscaled_outputs = label_scaler.inverse_transform(regression_output.cpu().detach().numpy().reshape(-1, 1))
                mae = mean_absolute_error(unscaled_outputs, unscaled_labels)
                all_test_predictions.extend(unscaled_outputs)
                all_test_labels.extend(unscaled_labels)

                total_test_loss += loss.item()
                total_test_mae += mae
                
                val_progress.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        avg_test_loss = total_test_loss / len(test_dataloader)
        avg_test_mae = total_test_mae / len(test_dataloader)
        
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
):
    """
    Tune learning rate and epoch count to minimize weighted MAE across targets.
    """

    # Precompute class weights once
    class_weights = get_class_weights(training_csv_path, target_names)

    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters
        # learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
        # num_epochs = trial.suggest_int("num_epochs", 1, max_epochs)

        # polyBERT: {'learning_rate': 2.38740929383219e-05, 'num_epochs': 8}
        # learning_rate = trial.suggest_float("learning_rate", 5e-6, 8e-5, log=True)
        # num_epochs = trial.suggest_int("num_epochs", 6, 10)

        # chemBERTa
        # • learning_rate: 7.659572276291352e-05
        # • num_epochs: 3
        learning_rate = trial.suggest_float("learning_rate", 2e-5, 3.2e-4, log=True)
        num_epochs = trial.suggest_int("num_epochs", 1, 5)

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
                # Prepare fold datasets
                train_dataset, val_dataset, label_scaler = get_train_test_datasets(
                    csv_path=training_csv_path,
                    model_path=base_model_identifier,
                    target_name=target,
                    fold_count=fold_count,
                    fold_id=fold_id,
                    augmentation_kwargs={},
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

                model = models_per_fold[fold_id]

                # Train (in-place) and evaluate this fold
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
        fold_count=5,
        input_data_path='data/from_host/train.csv',
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
                tokenizer = AutoTokenizer.from_pretrained(base_model_identifier)
                # model = BertRegressor(base_model_identifier)
                model = models[fold_id]
                
                # PREPARE DATALOADERS
                train_dataset, test_dataset, label_scaler = get_train_test_datasets(
                    csv_path=input_data_path, 
                    model_path=base_model_identifier,
                    target_name=target, 
                    fold_count=fold_count, 
                    fold_id=fold_id, 
                    augmentation_kwargs={}
                )
                
                train_dataloader = DataLoader(
                    train_dataset,
                    batch_size=16,
                    shuffle=True,
                    num_workers=4,
                    persistent_workers=True,
                    pin_memory=True
                )
                
                test_dataloader = DataLoader(
                    test_dataset,
                    batch_size=32,
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

# polyBERT:
#   LR 2e-5:  0.0722
#   LR 1e-5:  0.0722
# ModernBERT-base:
#   LR 1e-5, 15 epochs:  
#       Normal tokenization: 0.0710
#   LR 1e-5, 3 epochs:
#       Normal tokenization: 0.0699
#       Split characters: 0.0699
#   LR 2e-5, 3 epochs:
#       Normal tokenization: 0.0689 <-- Best
#   LR 4e-5, 3 epochs:
#       Normal tokenization: 0.0714
'''
Baseline LinearLR
    0 warmup epochs (during tuning --> rerun):
        chemBERTa (0.0745 --> 0.0759)
        • learning_rate: 0.00028492427709732543
        • num_epochs: 2
        polyBERT (0.0688 --> 0.0693)
        • learning_rate: 1.415518516826981e-05
        • num_epochs: 8
        ModernBERT-base (0.0689 --> 0.0692):
        • learning_rate: 2e-5
        • num_epochs: 3
    1 warmup epoch, minus one main epoch:
        ChemBERTa: 0.0753
        polyBERT: 0.0713
        ModernBERT: 0.0705
    1 warmup epoch, same finetuning epoch count:
        ChemBERTa: 0.0750
        polyBERT: 0.0717
        ModernBERT: 0.0710
With OneCycleLR (10% warmup, within each run):
    0 warmup epochs, baseline main epochs:
        ChemBERTa: 0.0752
        polyBERT: 0.0690 <--- Best polyBERT (0.0690 rerun)
        ModernBERT: 0.0689
    1 warmup epoch, minus one main epoch:
        ChemBERTa: 0.0733 <--- Best ChemBERTa (0.0747 rerun)
        polyBERT: 0.0698
        ModernBERT: 0.0695
    1 warmup epoch, same finetuning epoch count:
        ChemBERTa: 0.0753
        polyBERT: 0.0708
        ModernBERT: 0.0692 <--- Best ModernBERT (0.0713 rerun)
'''
if __name__ == '__main__':
    print(get_class_weights('data/from_host/train.csv', ["Tg", "FFV", "Tc", "Density", "Rg"]))
    # train_and_save_models(
    #     # base_model_identifier = 'DeepChem/ChemBERTa-77M-MTR',
    #     # context_pooler_hidden_size = 384,
    #     # warmup_epoch_count=1,
    #     # epoch_count=1,
    #     # learning_rate=2.8e-4,
    #     # output_dir_suffix='chem_tuned',
    #     # base_model_identifier = 'answerdotai/ModernBERT-base',
    #     # context_pooler_hidden_size = 768,
    #     # warmup_epoch_count=1,
    #     # epoch_count=3,
    #     # learning_rate=2e-5,
    #     # output_dir_suffix='modern_tuned',
    #     base_model_identifier = 'kuelumbus/polyBERT',
    #     context_pooler_hidden_size = 600,
    #     warmup_epoch_count=0,
    #     epoch_count=8,
    #     learning_rate=1.4e-5,
    #     output_dir_suffix='poly_tuned',
    #     fold_count=5,
    #     input_data_path='data/from_host/train.csv',
    # )
    '''
    optimize_learning_rate_and_epoch_count(
        # base_model_identifier = 'answerdotai/ModernBERT-base',
        # context_pooler_hidden_size = 768,
        # max_epochs = 8,
        # base_model_identifier = 'kuelumbus/polyBERT',
        # context_pooler_hidden_size = 600,
        # max_epochs=20,
        base_model_identifier = 'DeepChem/ChemBERTa-77M-MTR',
        context_pooler_hidden_size = 384,
        max_epochs=25,
        number_of_trials = 6
    )
    # '''