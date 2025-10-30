# src/models/transformer_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TransformerGenomicModel(nn.Module):
    """Трансформер для геномных данных с 22 фичами"""
    
    def __init__(self, input_dim=22, d_model=256, nhead=8, num_layers=6, dropout=0.2):
        super().__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        
        # Feature projection для 22 фич
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 200, d_model))  # sequence_length=200
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Multi-scale pooling
        self.attention_pool = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, regional_meta=None):
        batch_size, seq_len, _ = x.shape
        
        # Feature projection
        x = self.feature_projection(x) * np.sqrt(self.d_model)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Create mask for padding
        mask = (x.sum(dim=-1) == 0)
        
        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Global attention pooling
        query = torch.mean(x, dim=1, keepdim=True)
        attn_out, attn_weights = self.attention_pool(
            query, x, x, 
            key_padding_mask=mask
        )
        
        # Squeeze and get global representation
        global_repr = attn_out.squeeze(1)
        
        # Classification
        output = self.classifier(global_repr)
        
        return output