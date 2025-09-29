from transformers import PreTrainedModel, AutoConfig, BertModel, BertTokenizerFast, BertConfig, AutoModel, AutoTokenizer
import pandas as pd
import torch
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import joblib
import pandas as pd
import numpy as np
from rdkit import Chem
import random
from typing import Optional, List, Union
from sklearn.metrics import mean_absolute_error
from transformers import AutoTokenizer
import torch
from torch import nn
from transformers.activations import ACT2FN
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# train = pd.read_csv('data/from_host/train.csv')
# train = pd.read_csv('data/from_natsume/train_Tc-only_merged.csv')
# train = pd.read_csv('data/from_natsume/train_merged.csv')
train = pd.read_csv('data/from_dmitry/full_merge.csv')
# train = pd.read_csv('data/from_dmitry/host_tc-natsume_full-dmitry.csv')
test = pd.read_csv('data/from_host/test.csv')

targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

def augment_smiles_dataset(df: pd.DataFrame,
                               smiles_column: str = 'SMILES',
                               augmentation_strategies: List[str] = ['enumeration', 'kekulize', 'stereo_enum'],
                               n_augmentations: int = 10,
                               preserve_original: bool = True,
                               random_seed: Optional[int] = None) -> pd.DataFrame:
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    def apply_augmentation_strategy(smiles: str, strategy: str) -> List[str]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return [smiles]
            
            augmented = []
            
            if strategy == 'enumeration':
                # Standard SMILES enumeration
                for _ in range(n_augmentations):
                    enum_smiles = Chem.MolToSmiles(mol, 
                                                 canonical=False, 
                                                 doRandom=True,
                                                 isomericSmiles=True)
                    augmented.append(enum_smiles)
            
            elif strategy == 'kekulize':
                # Kekulization variants
                try:
                    Chem.Kekulize(mol)
                    kek_smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
                    augmented.append(kek_smiles)
                except:
                    pass
            
            elif strategy == 'stereo_enum':
                # Stereochemistry enumeration
                for _ in range(n_augmentations // 2):
                    # Remove stereochemistry
                    Chem.RemoveStereochemistry(mol)
                    no_stereo = Chem.MolToSmiles(mol)
                    augmented.append(no_stereo)
            
            return list(set(augmented))  # Remove duplicates
            
        except Exception as e:
            print(f"Error in {strategy} for {smiles}: {e}")
            return [smiles]
    
    augmented_rows = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        original_smiles = row[smiles_column]
        
        if preserve_original:
            original_row = row.to_dict()
            original_row['augmentation_strategy'] = 'original'
            original_row['is_augmented'] = False
            augmented_rows.append(original_row)
        
        for strategy in augmentation_strategies:
            strategy_smiles = apply_augmentation_strategy(original_smiles, strategy)
            
            for aug_smiles in strategy_smiles:
                if aug_smiles != original_smiles:
                    new_row = row.to_dict().copy()
                    new_row[smiles_column] = aug_smiles
                    new_row['augmentation_strategy'] = strategy
                    new_row['is_augmented'] = True
                    augmented_rows.append(new_row)
    
    augmented_df = pd.DataFrame(augmented_rows)
    augmented_df = augmented_df.reset_index(drop=True)
    
    print(f"Original size: {len(df)}, Augmented size: {len(augmented_df)}")
    print(f"Augmentation factor: {len(augmented_df) / len(df):.2f}x")
    
    return augmented_df

train = augment_smiles_dataset(train)

smiles_train = train['SMILES'].to_numpy()
scalers = []

for target in targets:
    actual_targets = train[target]
    label_scaler = StandardScaler()
    train[target] = label_scaler.fit_transform(train[target].to_numpy().reshape(-1, 1))
    
    scalers.append(label_scaler)

labels = train[targets].values

smiles_train, smiles_test, labels_train, labels_test = train_test_split(
    train['SMILES'], labels, test_size=0.1, random_state=42)

joblib.dump(scalers, 'models/label_scalers_v1_full_merge.pkl')
# joblib.dump(scalers, 'models/label_scalers_v1_host_tc-natsume_full-dmitry.pkl')
class ContextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # pooler_size = getattr(config, 'pooler_hidden_size', config.hidden_size) # TODO: Experimental
        pooler_size = 600
        self.dense = nn.Linear(pooler_size, pooler_size)
        
        # dropout_prob = getattr(config, 'pooler_dropout', config.hidden_dropout_prob) # TODO: Experimental
        dropout_prob = 0.144
        self.dropout = nn.Dropout(dropout_prob)
        
        # self.activation = getattr(config, 'pooler_hidden_act', config.hidden_act) # TODO: Experimental
        self.activation = 'gelu'
        self.config = config

    def forward(self, hidden_states):
        # Extract CLS token (first token)
        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.activation](pooled_output)
        return pooled_output

class CustomModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.backbone = AutoModel.from_config(config)
        
        self.pooler = ContextPooler(config)
        
        # Final classification layer
        pooler_output_dim = getattr(config, 'pooler_hidden_size', config.hidden_size)
        self.output = torch.nn.Linear(pooler_output_dim, 1)

    def forward(
        self,
        input_ids,
        scaler,
        attention_mask=None,
        # token_type_ids=None,
        position_ids=None,
        labels=None,
    ):
        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids, # TODO: Experimental
            position_ids=position_ids,
        )

        pooled_output = self.pooler(outputs.last_hidden_state)
        
        # Final regression output
        regression_output = self.output(pooled_output)

        loss = None
        true_loss = None
        if labels is not None:
            loss_fn = torch.nn.MSELoss()

            unscaled_labels = scaler.inverse_transform(labels.cpu().numpy())
            unscaled_outputs = scaler.inverse_transform(regression_output.cpu().detach().numpy())
            
            loss = loss_fn(regression_output, labels)
            true_loss = mean_absolute_error(unscaled_outputs, unscaled_labels)
            
        return {
            "loss": loss,
            "logits": regression_output,
            "true_loss": true_loss
        }

def get_pretrained(model_path):
    config = AutoConfig.from_pretrained(model_path)
    model = CustomModel(config)

    if model_path.endswith("pytorch_model.bin"):
        model.load_state_dict(torch.load(model_path))
    else:
        model.backbone = AutoModel.from_pretrained(model_path)

    for param in model.backbone.parameters():
        param.requires_grad = True
    return model

# model_path = 'DeepChem/ChemBERTa-77M-MTR'
# model_path = 'answerdotai/ModernBERT-base' # TODO: Experimental
model_path = 'kuelumbus/polyBERT' # TODO: Experimental

model = get_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

class SMILESDataset(Dataset):
    def __init__(self, smiles_list, labels, tokenizer, max_length=512):
        self.smiles_list = smiles_list
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        smiles = self.tokenizer.cls_token + self.smiles_list[idx]
        label = self.labels[idx]
        
        # Tokenize the SMILES string
        encoding = self.tokenizer(
            smiles,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float32)
        }

def train_model(model, train_dataloader, val_dataloader, scaler, num_epochs=10, learning_rate=2e-5, device='cuda'):
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps)
    
    train_losses = []
    val_losses = []
    
    model.train()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        total_train_loss = 0
        total_true_train_loss = 0
        train_progress = tqdm(train_dataloader, desc="Training", leave=False)
        
        for batch_idx, batch in enumerate(train_progress):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                scaler=scaler,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            loss = outputs['loss']
            true_loss = outputs['true_loss']
            
            total_train_loss += loss.item()
            total_true_train_loss += true_loss
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_progress.set_postfix({
                'loss': f'{loss.item():.4f}',
                'true_loss': f'{true_loss:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_true_train_loss = total_true_train_loss / len(train_dataloader)
        
        train_losses.append(avg_train_loss)
        
        model.eval()
        total_val_loss = 0
        total_true_val_loss = 0

        with torch.no_grad():
            val_progress = tqdm(val_dataloader, desc="Validation", leave=False)
            
            for batch in val_progress:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    scaler=scaler,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss']
                true_loss = outputs['true_loss']

                total_val_loss += loss.item()
                total_true_val_loss += true_loss
                
                val_progress.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_true_loss = total_true_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | True train loss: {avg_true_train_loss:.4f} | True val loss: {avg_val_true_loss:.4f}")
        
        model.train()
    
    return train_losses, val_losses

def dropnan(smiles, targets):
    non_nan_mask = ~np.isnan(targets)
        
    targets = targets[non_nan_mask].reshape(-1, 1)
    smiles = smiles.copy()[non_nan_mask].reset_index(drop=True)

    return smiles, targets
    

for i in range(len(targets)):
    target = targets[i]
    print(f'Training {i} feature. Feature name: {target}')
    scaler = scalers[i]
    labels_train_actual = labels_train[:, i]
    print(smiles_train.shape, 'act')
    labels_test_actual = labels_test[:, i]
    # If we were to drop ALL rows containing NaNs, we would have no data to train on
    smiles_train_actual, labels_train_actual = dropnan(smiles_train, labels_train_actual)
    smiles_test_actual, labels_test_actual = dropnan(smiles_test, labels_test_actual)
    
    train_dataset = SMILESDataset(smiles_train_actual, labels_train_actual, tokenizer)
    val_dataset = SMILESDataset(smiles_test_actual, labels_test_actual, tokenizer)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train the model
    train_losses, val_losses = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        scaler=scaler,
        # num_epochs=15,
        # learning_rate=2e-5,
        num_epochs=8, # TODO: Experimental
        learning_rate=1.4e-5,
        device=device
    )
    
    print('Overall loss: ', train_losses)
    # torch.save(model.state_dict(), f'models/polymer_bert_v1_host_tc-natsume_full-dmitry_{target}.pth')
    torch.save(model.state_dict(), f'models/polymer_bert_v1_full_merge_poly_{target}.pth')
    print("Model saved successfully!")