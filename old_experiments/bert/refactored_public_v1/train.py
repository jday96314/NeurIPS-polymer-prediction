from tqdm import tqdm
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
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
        device='cuda'):
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, fused=True)
    # optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps)
    
    loss_fn = torch.nn.MSELoss()

    scaler = GradScaler()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        total_train_loss = 0
        total_train_mae = 0
        train_progress_bar = tqdm(train_dataloader, desc="Training", leave=False)
        
        # TRAIN.
        model.train()
        for batch_idx, batch in enumerate(train_progress_bar):
            # UNPACK DATA.
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # UPDATE MODEL.
            optimizer.zero_grad()
            
            # with autocast(device_type='cuda', dtype=torch.bfloat16):
            regression_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            loss = loss_fn(regression_output, labels.reshape(-1,1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # scaler.scale(loss).backward()
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # scaler.step(optimizer)
            # scaler.update()
            # scheduler.step()

            # REPORT METRICS.
            unscaled_labels = label_scaler.inverse_transform(labels.cpu().numpy().reshape(-1, 1))
            unscaled_outputs = label_scaler.inverse_transform(regression_output.float().cpu().detach().numpy().reshape(-1, 1))
            mae = mean_absolute_error(unscaled_outputs, unscaled_labels)
            
            total_train_loss += loss.item()
            total_train_mae += mae
            
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

                total_test_loss += loss.item()
                total_test_mae += mae
                
                val_progress.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        avg_test_loss = total_test_loss / len(test_dataloader)
        avg_test_mae = total_test_mae / len(test_dataloader)
        
        print(f"Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Train MAE: {avg_train_mae:.4f} | Test MAE: {avg_test_mae:.4f}")
            
    return avg_test_mae

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

# FP32 scores:
#   * 5-fold: 0.0750, 0.0752, 0.0757
#   * 10-fold: 0.0731, 0.0741, 0.0732
# BF16 scores:
#   * 5-fold: 0.0757, 0.0760, 0.0763
# BF16 scores (no grad scaling, fullgraph=False):
#   * 5-fold: 0.0770, 0.0780, 0.0783
# BF16 scores (with grad scaling, fullgraph=False):
#   * 5-fold: 0.0759, 0.0773, 0.0783
if __name__ == '__main__':
    # CONFIG.
    TARGET_NAMES = ["Tg", "FFV", "Tc", "Density", "Rg"]
    BASE_MODEL_PATH = 'DeepChem/ChemBERTa-77M-MTR'
    INPUT_DATA_PATH = 'data/from_host/train.csv'
    # INPUT_DATA_PATH = 'data/from_natsume/train_Tc-only_merged.csv'
    # INPUT_DATA_PATH = 'data/from_natsume/train_merged.csv'
    FOLD_COUNT = 5
    
    # CREATE OUTPUT DIRECTORY.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'models/{timestamp}_partial_merge'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # TRAIN MODELS FOR ALL TARGETS.
    all_results = {}
    models = [torch.compile(BertRegressor(BASE_MODEL_PATH)) for _ in range(FOLD_COUNT)]
    # models = [BertRegressor(BASE_MODEL_PATH) for _ in range(FOLD_COUNT)]
    for target in TARGET_NAMES:
        print(f'\n=== Training {target} model across {FOLD_COUNT} folds ===')
        
        # TRAIN & TEST FOR EACH FOLD.
        fold_maes = []
        for fold_id in range(FOLD_COUNT):
            print(f'Training {target} - Fold {fold_id}/{FOLD_COUNT}')
            
            # CREATE FOLD-SPECIFIC OUTPUT DIRECTORY.
            fold_dir = os.path.join(output_dir, f'fold_{fold_id}')
            os.makedirs(fold_dir, exist_ok=True)
            
            # CREATE MODEL
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
            # model = BertRegressor(BASE_MODEL_PATH)
            model = models[fold_id]
            
            # PREPARE DATALOADERS
            train_dataset, test_dataset, label_scaler = get_train_test_datasets(
                csv_path=INPUT_DATA_PATH, 
                model_path=BASE_MODEL_PATH,
                target_name=target, 
                fold_count=FOLD_COUNT, 
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
            test_mae = train_model(
                model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                label_scaler=label_scaler,
                num_epochs=15,
                learning_rate=1e-5,
                device='cuda'
            )
            
            # SAVE RESULTS TO FOLD-SPECIFIC DIRECTORY
            scaler_path = os.path.join(fold_dir, f'scaler_{target}.pkl')
            model_path = os.path.join(fold_dir, f'polymer_bert_v2_{target}.pth')
            
            joblib.dump(label_scaler, scaler_path)
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
    class_weights = get_class_weights(INPUT_DATA_PATH, TARGET_NAMES)
    
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
        f.write(f'Number of folds: {FOLD_COUNT}\n')
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