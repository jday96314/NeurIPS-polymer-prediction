

from torch import nn
import torch
from transformers import AutoModel
from transformers.activations import ACT2FN
from sklearn.metrics import mean_absolute_error

class ContextPooler(nn.Module):
    def __init__(self, hidden_size, dropout_prob, activation_name):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = ACT2FN[activation_name]

    def forward(self, hidden_states):
        context_token = hidden_states[:, 0] # Extract CLS token (first token)

        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertRegressor(nn.Module):
    def __init__(
            self, 
            pretrained_model_path, 
            context_pooler_kwargs = {'hidden_size': 384, 'dropout_prob': 0.144, 'activation_name': 'gelu'}):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(pretrained_model_path)
        self.pooler = ContextPooler(**context_pooler_kwargs)
        
        # Final classification layer
        pooler_output_dim = context_pooler_kwargs['hidden_size']
        self.output = torch.nn.Linear(pooler_output_dim, 1)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None):
        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        pooled_output = self.pooler(outputs.last_hidden_state)        
        regression_output = self.output(pooled_output)

        return regression_output