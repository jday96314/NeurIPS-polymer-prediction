# polybert_regressor_rankup.py
from torch import nn
import torch
from transformers import AutoModel
from transformers.activations import ACT2FN

class ContextPooler(nn.Module):
    def __init__(self, hidden_size: int, dropout_prob: float, activation_name: str):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = ACT2FN[activation_name]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        context_token = hidden_states[:, 0]  # CLS
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertRegressor(nn.Module):
    """
    Drop-in: same name as your original class, but now optionally exposes an ARC head.
    If enable_arc=False, behavior is identical to your original.
    """
    def __init__(
        self,
        pretrained_model_path: str,
        target_count: int,
        context_pooler_kwargs: dict = {'hidden_size': 384, 'dropout_prob': 0.144, 'activation_name': 'gelu'},
        enable_arc: bool = False
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(pretrained_model_path)
        self.pooler = ContextPooler(**context_pooler_kwargs)

        pooler_output_dim = context_pooler_kwargs['hidden_size']
        self.output = nn.Linear(pooler_output_dim, target_count)

        # Auxiliary Ranking Classifier (ARC): scalar score
        self.enable_arc = enable_arc
        if self.enable_arc:
            self.arc_head = nn.Linear(pooler_output_dim, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        return_arc_score: bool = False
    ):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        pooled_output = self.pooler(outputs.last_hidden_state)
        regression_output = self.output(pooled_output)

        if return_arc_score and self.enable_arc:
            arc_score = self.arc_head(pooled_output)  # shape: [B, 1]
            return regression_output, arc_score
        return regression_output
