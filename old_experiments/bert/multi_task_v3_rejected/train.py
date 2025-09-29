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
import optuna
import pickle

from dataset import SMILESDataset, get_train_test_datasets, TARGET_NAMES
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
    total_steps = len(train_dataloader) * num_epochs
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps)
    
    loss_fn = torch.nn.MSELoss(reduction='none')

    scaler = GradScaler()
    
    for epoch in range(num_epochs):                
        # TRAIN.
        model.train()
        total_train_loss = 0
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Training epoch {epoch + 1}/{num_epochs}", leave=False)):
            # UNPACK DATA.
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            loss_mask = batch['loss_mask'].to(device)
            
            # UPDATE MODEL.
            optimizer.zero_grad()
            
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                regression_output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                masked_labels = torch.nan_to_num(labels, nan=0.0)
                loss_values = loss_fn(regression_output, masked_labels)
                masked_losses = loss_values * loss_mask
                
                # class_weights = (1 / (loss_mask.sum(0) + 1e-6))# ** 0.5
                # class_weights /= class_weights.sum()
                # masked_losses = masked_losses * class_weights

                loss = masked_losses.sum() / loss_mask.sum()
                total_train_loss += loss.item()

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
        
        avg_train_loss = total_train_loss / len(train_dataloader)
                
        # TEST.
        model.eval()
        total_test_loss = 0
        all_test_predictions = []
        all_test_labels = []
        all_test_label_masks = []
        with torch.no_grad():           
            for batch in tqdm(test_dataloader, desc="Validation", leave=False):
                # UNPACK DATA.
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                loss_mask = batch['loss_mask'].to(device)
                
                # FORWARD PASS.
                regression_output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # RECORD PREDICTIONS.
                unscaled_labels = label_scaler.inverse_transform(labels.cpu().numpy())
                unscaled_outputs = label_scaler.inverse_transform(regression_output.cpu().detach().numpy())
                all_test_predictions.extend(unscaled_outputs)
                all_test_labels.extend(unscaled_labels)
                all_test_label_masks.extend(loss_mask.cpu().numpy())

                # RECORD LOSS.
                masked_labels = torch.nan_to_num(labels, nan=0.0)
                loss_values = loss_fn(regression_output, masked_labels)
                masked_losses = loss_values * loss_mask
                loss = masked_losses.sum() / loss_mask.sum()
                total_test_loss += loss.item()

        maes_by_target = []
        formatted_maes = []
        for target_index in range(len(TARGET_NAMES)):
            target_mask = np.array(all_test_label_masks)[:,target_index].astype(bool)
            target_predictions = np.array(all_test_predictions)[:,target_index][target_mask]
            target_labels = np.array(all_test_labels)[:,target_index][target_mask]
            mae = mean_absolute_error(target_predictions, target_labels)
            maes_by_target.append(mae)
            formatted_maes.append(f'{mae:.4f}')
        
        avg_test_loss = total_test_loss / len(test_dataloader)        
        print(f"Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | MAEs: {formatted_maes}")
            
    return maes_by_target, all_test_predictions, all_test_labels

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

def train_and_save_models(
        base_model_identifier = 'DeepChem/ChemBERTa-77M-MTR',
        context_pooler_hidden_size = 768,
        epoch_count=2,
        learning_rate=2.8e-4,
        fold_count=5,
        input_data_path='data/from_host/train.csv',
        output_dir_suffix='temp'):
    # CREATE OUTPUT DIRECTORY.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'models/{timestamp}_{output_dir_suffix}_multitask'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
        
    # TRAIN & TEST FOR EACH FOLD.
    all_wmaes = []
    targets_to_maes = {target_name:[] for target_name in TARGET_NAMES}
    for fold_id in range(fold_count):
        print(f'Fold {fold_id + 1}/{fold_count}')
        
        # CREATE FOLD-SPECIFIC OUTPUT DIRECTORY.
        fold_dir = os.path.join(output_dir, f'fold_{fold_id}')
        os.makedirs(fold_dir, exist_ok=True)
        
        # CREATE MODEL
        model = BertRegressor(
            base_model_identifier, 
            target_count=len(TARGET_NAMES),
            context_pooler_kwargs = {
                'hidden_size': context_pooler_hidden_size, 
                'dropout_prob': 0.144, 
                'activation_name': 'gelu'
            }
        ) 
        
        # PREPARE DATALOADERS.
        train_dataset, test_dataset, label_scaler = get_train_test_datasets(
            csv_path=input_data_path, 
            model_path=base_model_identifier,
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
        
        # TRAIN & TEST MODEL.
        target_maes, oof_preds, oof_labels = train_model(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            label_scaler=label_scaler,
            num_epochs=epoch_count,
            learning_rate=learning_rate,
            device='cuda'
        )
        
        # CALCULATE wMAE.
        class_weights = get_class_weights(input_data_path, TARGET_NAMES)
        weighted_mae = np.average(target_maes, weights=class_weights)

        # SAVE RESULTS.
        scaler_path = os.path.join(fold_dir, f'scaler.pkl')
        model_path = os.path.join(fold_dir, f'polymer_bert_v3_{int(weighted_mae) * 1000}.pth')
        
        joblib.dump(label_scaler, scaler_path)
        torch.save(model.state_dict(), model_path)
        
        all_wmaes.append(weighted_mae)
        print(f'Fold {fold_id} wMAE: {weighted_mae:.4f}')
        for target_name, mae in zip(TARGET_NAMES, target_maes):
            print(f'\t{target_name}: MAE = {mae}')
            targets_to_maes[target_name].append(mae)

    # LOG SUMMARY.
    summary_text = f'wMAE: {np.mean(all_wmaes):.4f}\n'
    for target_name, maes in targets_to_maes.items():
        summary_text += f'\t{target_name}: MAE = {np.mean(maes)}\n'

    print(summary_text)

    summary_path = os.path.join(output_dir, 'cv_results_summary.txt')
    with open(summary_path, 'w') as summary_file:
        summary_file.write(summary_text)


if __name__ == '__main__':
    train_and_save_models(
        # base_model_identifier = 'DeepChem/ChemBERTa-77M-MTR',
        # context_pooler_hidden_size = 384,
        # epoch_count=2,
        # # 1/4 lr: 0.1370
        # # 1/2 lr: 0.1002
        # #   Equal weighting: 0.0752
        # #   1/N weighting: 0.0784
        # #   (1/N)**0.5 weighting:0.0777
        # # 1x lr: 0.0875
        # #   Equal weighting: 0.0753
        # #   1/N weighting: 0.0755
        # #   (1/N)**0.5 weighting: 0.0749
        # # 2x lr: 0.0890
        # #   Equal weighting: 0.0728 (0.0737)
        # #   1/N weighting: 0.0800 (0.0787)
        # #   (1/N)**0.5 weighting: 0.0802 (0.0788)
        # # 4x lr: 0.1178
        # learning_rate=(2.8e-4)*2,
        # output_dir_suffix='chem',
        base_model_identifier = 'answerdotai/ModernBERT-base',
        context_pooler_hidden_size = 768,
        epoch_count=3,
        # 1/2 lr:
        #   Equal weighting: 0.0703
        #   1/N weighting: 0.0716
        #   (1/N)**0.5 weighting: 0.0706
        # 1x lr:
        #   Equal weighting: 0.0719
        #   1/N weighting: 0.0738
        #   (1/N)**0.5 weighting: 0.0727
        # 2x lr:
        #   Equal weighting: 0.0700 (0.0698)
        #   1/N weighting: 0.0757
        #   (1/N)**0.5 weighting: 0.0707
        learning_rate=(2e-5)*2,
        output_dir_suffix='modern',
        # base_model_identifier = 'kuelumbus/polyBERT',
        # context_pooler_hidden_size = 600,
        # epoch_count=8,
        # learning_rate=1.4e-5,
        # output_dir_suffix='poly',
        fold_count=5,
        input_data_path='data/from_host/train.csv',
    )