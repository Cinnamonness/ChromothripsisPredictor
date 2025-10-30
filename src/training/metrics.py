import torch
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, 
    average_precision_score, confusion_matrix, roc_curve
)
import logging

logger = logging.getLogger(__name__)

def check_invalid_data(tensor):
    """Проверяет тензор на наличие некорректных значений"""
    if isinstance(tensor, tuple):
        return any(check_invalid_data(t) for t in tensor if t is not None and t.numel() > 0)
    else:
        return (torch.isnan(tensor).any() or 
                torch.isinf(tensor).any() or
                (tensor.abs() > 1e6).any())

def compute_enhanced_metrics(preds, targets, threshold=0.5):
    """Вычисление метрик с гарантированным наличием specificity"""
    try:
        preds_proba = torch.sigmoid(preds).cpu().numpy().flatten()
        targets_np = targets.cpu().numpy().flatten().astype(int)
        
        if len(np.unique(targets_np)) < 2:
            return {
                'f1': 0.0, 'auc': 0.5, 'precision': 0.0, 'recall': 0.0, 
                'avg_precision': 0.0, 'specificity': 1.0, 'balanced_accuracy': 0.5,
                'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0
            }
        
        preds_class = (preds_proba > threshold).astype(int)
        
        precision = precision_score(targets_np, preds_class, zero_division=0)
        recall = recall_score(targets_np, preds_class, zero_division=0)
        f1 = f1_score(targets_np, preds_class, zero_division=0)
        
        try:
            auc_score = roc_auc_score(targets_np, preds_proba)
        except:
            auc_score = 0.5
        
        try:
            avg_precision = average_precision_score(targets_np, preds_proba)
        except:
            avg_precision = 0.0
        
        tn, fp, fn, tp = confusion_matrix(targets_np, preds_class).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0
        balanced_accuracy = (recall + specificity) / 2
        
        return {
            'f1': f1, 'auc': auc_score, 'precision': precision, 'recall': recall,
            'avg_precision': avg_precision, 'specificity': specificity, 
            'balanced_accuracy': balanced_accuracy, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
        }
    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        return {
            'f1': 0.0, 'auc': 0.5, 'precision': 0.0, 'recall': 0.0, 
            'avg_precision': 0.0, 'specificity': 1.0, 'balanced_accuracy': 0.5,
            'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0
        }