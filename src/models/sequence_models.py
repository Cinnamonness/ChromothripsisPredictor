# src/models/sequence_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)

class EnhancedBiLSTM(nn.Module):
    """Улучшенная BiLSTM модель для 22 фич"""
    
    def __init__(self, config):
        super().__init__()
        
        self.input_dim = config.get('input_dim', 22)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_layers = config.get('num_layers', 2)
        self.dropout_rate = config.get('dropout_rate', 0.3)
        
        logger.info(f"EnhancedBiLSTM: input_dim={self.input_dim}, hidden_dim={self.hidden_dim}")
        
        # Feature embedding
        self.feature_embedding = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.LayerNorm(self.hidden_dim)
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            self.hidden_dim,
            self.hidden_dim // 2,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            self.hidden_dim, 
            num_heads=8, 
            dropout=0.1,
            batch_first=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 4, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and 'lstm' in name:
                nn.init.orthogonal_(param)
            elif 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x, regional_meta=None):
        batch_size, seq_len, features = x.shape
        
        # Проверка размерности
        if features != self.input_dim:
            logger.warning(f"Размерность фичей не совпадает: {features} vs {self.input_dim}")
            if features < self.input_dim:
                padding = torch.zeros(batch_size, seq_len, self.input_dim - features, device=x.device)
                x = torch.cat([x, padding], dim=2)
            else:
                x = x[:, :, :self.input_dim]
        
        # Feature embedding
        x_embedded = self.feature_embedding(x)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x_embedded)
        
        # Attention
        query = torch.mean(lstm_out, dim=1, keepdim=True)
        attended, attn_weights = self.attention(query, lstm_out, lstm_out)
        global_repr = attended.squeeze(1)
        
        # Classification
        output = self.classifier(global_repr)
        
        return output


class ChromothripsisBiLSTM(nn.Module):
    """Специализированная модель для детекции хромотрипсиса с 22 фичами"""
    
    def __init__(self, config):
        super().__init__()
        
        self.input_dim = config.get('input_dim', 22)
        self.hidden_dim = config.get('hidden_dim', 256)
        
        logger.info(f"ChromothripsisBiLSTM: input_dim={self.input_dim}, hidden_dim={self.hidden_dim}")
        
        # Улучшенное embedding для 22 фич
        self.feature_embedding = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LayerNorm(self.hidden_dim)
        )
        
        # Многослойный BiLSTM
        self.lstm1 = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True, 
                            bidirectional=True, dropout=0.2)
        self.lstm2 = nn.LSTM(self.hidden_dim * 2, self.hidden_dim, batch_first=True, 
                            bidirectional=True, dropout=0.2)
        
        # Multi-scale attention
        self.attention = nn.MultiheadAttention(self.hidden_dim * 2, num_heads=8, 
                                              dropout=0.3, batch_first=True)
        
        # Regional context integration
        self.regional_context = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 4, 1)
        )
        
        self.dropout = nn.Dropout(0.3)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
    
    def forward(self, x, regional_meta=None):
        batch_size, seq_len, features = x.shape
        
        # Проверка размерности
        if features != self.input_dim:
            if features < self.input_dim:
                padding = torch.zeros(batch_size, seq_len, self.input_dim - features, device=x.device)
                x = torch.cat([x, padding], dim=2)
            else:
                x = x[:, :, :self.input_dim]
        
        # Feature embedding
        x_embedded = self.feature_embedding(x)
        
        # LSTM processing
        lstm_out1, _ = self.lstm1(x_embedded)
        lstm_out1 = self.dropout(lstm_out1)
        
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.dropout(lstm_out2)
        
        # Attention
        mask = (x.sum(dim=-1) != 0).float()
        original_lengths = mask.sum(dim=1)
        
        max_len = x.size(1)
        attention_mask = torch.arange(max_len).expand(len(original_lengths), max_len).to(x.device) >= original_lengths.unsqueeze(1).to(x.device)
        
        attn_out, attn_weights = self.attention(
            lstm_out2, lstm_out2, lstm_out2, 
            key_padding_mask=attention_mask
        )
        
        # Multi-level pooling
        attn_out = attn_out * (~attention_mask.unsqueeze(-1)).float()
        
        avg_pool = attn_out.sum(dim=1) / original_lengths.unsqueeze(1).clamp(min=1)
        max_pool, _ = attn_out.max(dim=1)
        
        # Combine features
        x_combined = torch.cat([avg_pool, max_pool], dim=1)
        
        # Context integration
        x_context = self.regional_context(x_combined)
        
        # Classification
        output = self.classifier(x_context)
        
        return output


class ResidualLSTM(nn.Module):
    """Глубокая LSTM с residual connections для 22 фич"""
    
    def __init__(self, input_dim=22, hidden_dim=256, num_layers=4, dropout=0.3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Initial feature embedding
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
        
        # Residual LSTM layers
        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(num_layers):
            lstm = nn.LSTM(
                hidden_dim, 
                hidden_dim, 
                batch_first=True, 
                bidirectional=True,
                dropout=dropout if i < num_layers - 1 else 0
            )
            self.lstm_layers.append(lstm)
            self.layer_norms.append(nn.LayerNorm(hidden_dim * 2))
        
        # Multi-scale attention
        self.temporal_attention = nn.MultiheadAttention(
            hidden_dim * 2, 
            num_heads=8, 
            dropout=dropout,
            batch_first=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x, regional_meta=None):
        batch_size, seq_len, _ = x.shape
        
        # Feature embedding
        x = self.feature_embedding(x)
        
        # Residual LSTM processing
        for i, (lstm, norm) in enumerate(zip(self.lstm_layers, self.layer_norms)):
            residual = x
            if i == 0:
                # First layer - need to handle dimension expansion
                lstm_out, _ = lstm(x)
                x = norm(lstm_out)
            else:
                # Subsequent layers with residual connections
                lstm_out, _ = lstm(x)
                x = norm(lstm_out + residual)
        
        # Temporal attention
        query = torch.mean(x, dim=1, keepdim=True)
        attended, attn_weights = self.temporal_attention(query, x, x)
        global_repr = attended.squeeze(1)
        
        # Classification
        output = self.classifier(global_repr)
        
        return output