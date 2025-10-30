import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Dict, Tuple
import logging
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class SequenceDataset(Dataset):
    """Dataset для последовательностных данных RNN"""
    def __init__(self, data: List[Dict]):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        features = torch.FloatTensor(item['features'])
        label = torch.FloatTensor([item['label']])
        return features, label

class CNNDataset(Dataset):
    """Dataset для CNN данных"""
    def __init__(self, data: List[Dict]):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        features = torch.FloatTensor(item['features'])
        label = torch.FloatTensor([item['label']])
        return features, label

class GraphDataset(Dataset):
    """Dataset для графовых данных (GNN)"""
    def __init__(self, data: List[Dict]):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        graph_data = item['features']
        
        if 'node_features' in graph_data and len(graph_data['node_features']) > 0:
            node_features = torch.FloatTensor(graph_data['node_features'])
            if node_features.dim() == 1:
                node_features = node_features.unsqueeze(0)
        else:
            node_features = torch.FloatTensor([[0.1] * 7])
            
        if 'edges' in graph_data and len(graph_data['edges']) > 0:
            edge_index = torch.LongTensor(graph_data['edges'])
            if edge_index.dim() == 1:
                edge_index = edge_index.unsqueeze(0)
            if edge_index.shape[0] != 2:
                edge_index = edge_index.t()
        else:
            edge_index = torch.LongTensor([[0], [0]])
            
        if 'edge_features' in graph_data and len(graph_data['edge_features']) > 0:
            edge_attr = torch.FloatTensor(graph_data['edge_features'])
        else:
            edge_attr = torch.FloatTensor([[0.1]])
            
        label = torch.FloatTensor([item['label']])
        
        return (node_features, edge_index, edge_attr), label

def graph_collate_fn(batch):
    node_features_list = []
    edge_index_list = []
    edge_attr_list = []
    labels = []
    batch_indices = []
    
    current_node_count = 0
    
    for i, (graph_data, label) in enumerate(batch):
        node_features, edge_index, edge_attr = graph_data
        
        node_features_list.append(node_features)
        
        if edge_index.numel() > 0 and edge_index.shape[1] > 0:
            edge_index = edge_index + current_node_count
            edge_index_list.append(edge_index)
            if edge_attr is not None and edge_attr.numel() > 0:
                edge_attr_list.append(edge_attr)
        
        batch_indices.extend([i] * node_features.shape[0])
        
        current_node_count += node_features.shape[0]
        labels.append(label)
    
    if node_features_list:
        batched_node_features = torch.cat(node_features_list, dim=0)
    else:
        batched_node_features = torch.zeros((1, 1))
    
    if edge_index_list:
        batched_edge_index = torch.cat(edge_index_list, dim=1)
    else:
        batched_edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    if edge_attr_list:
        batched_edge_attr = torch.cat(edge_attr_list, dim=0)
    else:
        batched_edge_attr = torch.zeros((1, 1))
    
    batched_batch = torch.tensor(batch_indices, dtype=torch.long)
    batched_labels = torch.cat(labels, dim=0) if labels else torch.tensor([], dtype=torch.float)
    
    return (batched_node_features, batched_edge_index, batched_edge_attr, batched_batch), batched_labels

def create_data_loaders(sequence_data: List[Dict], cnn_data: List[Dict], 
                       graph_data: List[Dict], cfg) -> Dict:
    """Создает DataLoader'ы с учетом дисбаланса классов"""
    loaders = {}
    
    # Получаем параметры разделения из конфига
    val_ratio = cfg.data.split.val_ratio
    test_ratio = cfg.data.split.test_ratio if hasattr(cfg.data.split, 'test_ratio') else 0.0
    random_state = cfg.data.split.random_state if hasattr(cfg.data.split, 'random_state') else 42
    
    logger.info(f"Создание DataLoader'ов с разделением: val={val_ratio}, test={test_ratio}")
    
    def create_stratified_split(data, val_ratio, test_ratio, random_state):
        """Создает стратифицированное разделение для сохранения распределения классов"""
        if len(data) == 0:
            return [], [], []
            
        # Извлекаем метки
        labels = [item['label'] for item in data]
        
        if test_ratio > 0:
            # Разделяем на train+val и test
            train_val_data, test_data = train_test_split(
                data, 
                test_size=test_ratio, 
                random_state=random_state,
                stratify=labels
            )
            
            # Разделяем train_val на train и val
            train_val_labels = [item['label'] for item in train_val_data]
            train_data, val_data = train_test_split(
                train_val_data,
                test_size=val_ratio/(1-test_ratio),  # Корректируем ratio
                random_state=random_state,
                stratify=train_val_labels
            )
            
            return train_data, val_data, test_data
        else:
            # Только train и val
            train_data, val_data = train_test_split(
                data,
                test_size=val_ratio,
                random_state=random_state,
                stratify=labels
            )
            return train_data, val_data, []
    
    # Обрабатываем sequence данные
    if len(sequence_data) > 0:
        logger.info(f"Создание DataLoader'ов для sequence данных ({len(sequence_data)} samples)")
        
        train_data, val_data, test_data = create_stratified_split(
            sequence_data, val_ratio, test_ratio, random_state
        )
        
        train_dataset = SequenceDataset(train_data)
        val_dataset = SequenceDataset(val_data)
        
        # Логируем распределение классов
        train_labels = [item['label'] for item in train_data]
        val_labels = [item['label'] for item in val_data]
        
        logger.info(f"Sequence Train: {len(train_data)} samples, "
                   f"положительных: {sum(train_labels)} ({sum(train_labels)/len(train_labels)*100:.1f}%)")
        logger.info(f"Sequence Val: {len(val_data)} samples, "
                   f"положительных: {sum(val_labels)} ({sum(val_labels)/len(val_labels)*100:.1f}%)")
        
        loaders['sequence'] = (
            DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True),
            DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False)
        )
    
    # Обрабатываем CNN данные
    if len(cnn_data) > 0:
        logger.info(f"Создание DataLoader'ов для CNN данных ({len(cnn_data)} samples)")
        
        train_data, val_data, test_data = create_stratified_split(
            cnn_data, val_ratio, test_ratio, random_state
        )
        
        train_dataset = CNNDataset(train_data)
        val_dataset = CNNDataset(val_data)
        
        loaders['cnn'] = (
            DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True),
            DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False)
        )
    
    # Обрабатываем graph данные
    if len(graph_data) > 0:
        logger.info(f"Создание DataLoader'ов для graph данных ({len(graph_data)} samples)")
        
        train_data, val_data, test_data = create_stratified_split(
            graph_data, val_ratio, test_ratio, random_state
        )
        
        train_dataset = GraphDataset(train_data)
        val_dataset = GraphDataset(val_data)
        
        loaders['graph'] = (
            DataLoader(
                train_dataset, 
                batch_size=min(cfg.training.batch_size, 8), 
                shuffle=True,
                collate_fn=graph_collate_fn
            ),
            DataLoader(
                val_dataset, 
                batch_size=min(cfg.training.batch_size, 8), 
                shuffle=False,
                collate_fn=graph_collate_fn
            )
        )
    
    logger.info(f"Создано DataLoader'ов: {list(loaders.keys())}")
    return loaders