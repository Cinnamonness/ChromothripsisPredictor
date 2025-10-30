import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGNN(nn.Module):
    """Исправленная GNN модель с правильной обработкой батчей"""
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.conv3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim // 2)
        self.batch_norm3 = nn.BatchNorm1d(hidden_dim // 4)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_dim // 4, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        if isinstance(x, tuple):
            if len(x) == 4:
                node_features, edge_index, edge_attr, batch = x
            elif len(x) == 3:
                node_features, edge_index, edge_attr = x
                batch = None
            else:
                node_features = x
                batch = None
        else:
            node_features = x
            batch = None
        
        if torch.isnan(node_features).any():
            node_features = torch.nan_to_num(node_features, nan=0.0)
        
        x = F.relu(self.batch_norm1(self.conv1(node_features)))
        x = self.dropout(x)
        
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = self.dropout(x)
        
        if batch is not None and len(batch) == len(node_features):
            unique_batches = torch.unique(batch)
            graph_embeddings = []
            
            for batch_id in unique_batches:
                mask = (batch == batch_id)
                if mask.sum() > 0:
                    graph_embedding = x[mask].mean(dim=0, keepdim=True)
                    graph_embeddings.append(graph_embedding)
            
            if graph_embeddings:
                x = torch.cat(graph_embeddings, dim=0)
            else:
                x = x.mean(dim=0, keepdim=True)
        else:
            if x.dim() > 1 and x.size(0) > 1:
                x = x.mean(dim=0, keepdim=True)
        
        output = self.classifier(x)
        
        return output