# src/visualization/plots.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List
import os
import logging

logger = logging.getLogger(__name__)

def plot_training_history(histories: List[Dict], model_names: List[str]):
    """Визуализация процесса обучения"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for idx, (history, name) in enumerate(zip(histories, model_names)):
        if history is None or not history['train_loss']:
            continue
            
        # Loss
        axes[0, 0].plot(history['train_loss'], label=f'{name} Train', alpha=0.7)
        axes[0, 0].plot(history['val_loss'], label=f'{name} Val', alpha=0.7, linestyle='--')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # F1 Score
        axes[0, 1].plot(history['train_f1'], label=f'{name} Train', alpha=0.7)
        axes[0, 1].plot(history['val_f1'], label=f'{name} Val', alpha=0.7, linestyle='--')
        axes[0, 1].set_title('F1 Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1)
        
        # Accuracy
        axes[0, 2].plot(history['train_accuracy'], label=f'{name} Train', alpha=0.7)
        axes[0, 2].plot(history['val_accuracy'], label=f'{name} Val', alpha=0.7, linestyle='--')
        axes[0, 2].set_title('Accuracy')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_ylim(0, 1)
        
        # Precision-Recall
        axes[1, 0].plot(history['train_precision'], label=f'{name} Train Prec', alpha=0.7)
        axes[1, 0].plot(history['val_precision'], label=f'{name} Val Prec', alpha=0.7, linestyle='--')
        axes[1, 0].plot(history['train_recall'], label=f'{name} Train Rec', alpha=0.7)
        axes[1, 0].plot(history['val_recall'], label=f'{name} Val Rec', alpha=0.7, linestyle='--')
        axes[1, 0].set_title('Precision & Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1)
        
        # Learning Rate
        if 'learning_rate' in history:
            axes[1, 1].plot(history['learning_rate'], label=name, alpha=0.7)
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('outputs/figures', exist_ok=True)
    plt.savefig('outputs/figures/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curves(results: Dict, model_names: List[str]):
    """ROC кривые для всех моделей"""
    plt.figure(figsize=(10, 8))
    
    has_data = False
    for model_type in model_names:
        if (model_type in results and 
            'final_metrics' in results[model_type] and 
            'roc_curve' in results[model_type]['final_metrics']):
            
            roc_data = results[model_type]['final_metrics']['roc_curve']
            auc_score = results[model_type]['final_metrics'].get('auc', 0)
            
            plt.plot(roc_data['fpr'], roc_data['tpr'], 
                    label=f'{model_type} (AUC = {auc_score:.4f})', 
                    linewidth=2)
            has_data = True
    
    if not has_data:
        logger.warning("No ROC curve data available")
        return
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Model Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('outputs/figures/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_metrics_comparison(results: Dict):
    """Сравнение метрик всех моделей"""
    if not results:
        logger.warning("No results to plot")
        return
        
    metrics_to_plot = ['f1', 'precision', 'recall', 'specificity', 'auc']
    model_names = []
    metric_values = {metric: [] for metric in metrics_to_plot}
    
    for model_type, result in results.items():
        model_names.append(model_type)
        for metric in metrics_to_plot:
            if 'final_metrics' in result and metric in result['final_metrics']:
                metric_values[metric].append(result['final_metrics'][metric])
            elif 'metrics' in result and metric in result['metrics']:
                metric_values[metric].append(result['metrics'][metric])
            else:
                metric_values[metric].append(0)
                logger.warning(f"Metric '{metric}' not found for model {model_type}")
    
    if not model_names:
        logger.warning("No models to compare")
        return
    
    x = np.arange(len(model_names))
    width = 0.15
    multiplier = 0
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, metric in enumerate(metrics_to_plot):
        if any(metric_values[metric]):  # Проверяем, что есть данные для этой метрики
            offset = width * multiplier
            rects = ax.bar(x + offset, metric_values[metric], width, 
                          label=metric.capitalize(), alpha=0.8, color=colors[i])
            ax.bar_label(rects, padding=3, fmt='%.3f', fontsize=8)
            multiplier += 1
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title('Model Metrics Comparison')
    ax.set_xticks(x + width * (multiplier - 1) / 2)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('outputs/figures/metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrices(results: Dict):
    """Матрицы ошибок для всех моделей"""
    if not results:
        return
        
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_type, result) in enumerate(results.items()):
        if 'final_metrics' in result:
            metrics = result['final_metrics']
        elif 'metrics' in result:
            metrics = result['metrics']
        else:
            continue
            
        if all(k in metrics for k in ['tp', 'fp', 'fn', 'tn']):
            cm = np.array([[metrics['tn'], metrics['fp']],
                          [metrics['fn'], metrics['tp']]])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Predicted 0', 'Predicted 1'],
                       yticklabels=['Actual 0', 'Actual 1'])
            axes[idx].set_title(f'{model_type} - Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig('outputs/figures/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_precision_recall_curves(results: Dict, model_names: List[str]):
    """Precision-Recall кривые для всех моделей"""
    plt.figure(figsize=(10, 8))
    
    has_data = False
    for model_type in model_names:
        if (model_type in results and 
            'final_metrics' in results[model_type] and 
            'pr_curve' in results[model_type]['final_metrics']):
            
            pr_data = results[model_type]['final_metrics']['pr_curve']
            avg_precision = results[model_type]['final_metrics'].get('average_precision', 0)
            
            plt.plot(pr_data['recall'], pr_data['precision'], 
                    label=f'{model_type} (AP = {avg_precision:.4f})', 
                    linewidth=2)
            has_data = True
    
    if not has_data:
        logger.warning("No Precision-Recall curve data available")
        return
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves - Model Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('outputs/figures/precision_recall_curves.png', dpi=300, bbox_inches='tight')
    plt.show()