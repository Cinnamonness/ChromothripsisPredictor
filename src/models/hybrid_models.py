# src/models/hybrid_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)

class HybridCNNTransformer(nn.Module):
    """Гибридная CNN-Transformer архитектура для 22 фич"""
    
    def __init__(self, input_dim=22, cnn_channels=128, d_model=256, nhead=8, num_layers=4, dropout=0.2):
        super().__init__()
        
        # CNN feature extractor для 22 фич
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2),
            
            nn.Conv1d(cnn_channels * 2, cnn_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels * 4),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Projection to transformer dimension
        self.projection = nn.Linear(cnn_channels * 4, d_model)
        
        # Transformer
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model))  # после pooling
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Multi-head attention pooling
        self.attention_pool = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
    
    def forward(self, x, regional_meta=None):
        # CNN feature extraction
        x = x.transpose(1, 2)  # [batch, features, seq_len]
        cnn_features = self.cnn_layers(x)
        cnn_features = cnn_features.transpose(1, 2)  # [batch, seq_len, features]
        
        # Project to transformer dimension
        x = self.projection(cnn_features)
        
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Transformer processing
        x = self.transformer(x)
        
        # Attention pooling
        query = torch.mean(x, dim=1, keepdim=True)
        global_repr, _ = self.attention_pool(query, x, x)
        global_repr = global_repr.squeeze(1)
        
        # Classification
        output = self.classifier(global_repr)
        
        return output