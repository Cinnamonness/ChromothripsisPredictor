import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from tqdm import tqdm
import os
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)

class DataAugmentation:
    """Аугментации для последовательностей"""
    
    @staticmethod
    def temporal_shift(sequence, max_shift=10):
        shift = np.random.randint(-max_shift, max_shift)
        if shift > 0:
            return np.concatenate([sequence[shift:], sequence[:shift]])
        elif shift < 0:
            return np.concatenate([sequence[shift:], sequence[:shift]])
        return sequence
    
    @staticmethod  
    def feature_noise(sequence, noise_level=0.05):
        noise = np.random.normal(0, noise_level, sequence.shape)
        return sequence + noise
    
    @staticmethod
    def random_mask(sequence, mask_prob=0.1):
        mask = np.random.binomial(1, mask_prob, sequence.shape)
        return sequence * (1 - mask)

class EMA:
    """Exponential Moving Average"""
    def __init__(self, model, decay=0.995):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
                
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
                
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

class Trainer:
    def __init__(self, model, cfg, model_type: str):
        self.model = model
        self.cfg = cfg
        self.model_type = model_type
        self.device = self._setup_device()
        self.model.to(self.device)
        
        # Training parameters - ИСПРАВЛЕНО: значения по умолчанию
        self.epochs = getattr(cfg.training, 'epochs', 100)
        self.early_stopping_patience = getattr(cfg.training, 'early_stopping_patience', 25)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=getattr(cfg.training, 'learning_rate', 1e-4),
            weight_decay=getattr(cfg.training, 'weight_decay', 1e-4),
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.warmup_epochs = getattr(cfg.training, 'warmup_epochs', 0)
        self.scheduler = self._setup_scheduler()
        
        # EMA
        self.use_ema = getattr(cfg.training, 'use_ema', False)
        if self.use_ema:
            self.ema = EMA(model, decay=getattr(cfg.training, 'ema_decay', 0.995))
            self.ema.register()
        
        # Loss function
        self.criterion = None
        self.class_weights = None
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_f1': [], 'val_f1': [],
            'train_accuracy': [], 'val_accuracy': [],
            'train_precision': [], 'val_precision': [],
            'train_recall': [], 'val_recall': [],
            'learning_rate': []
        }
        
        # Gradient accumulation
        self.grad_accum_steps = getattr(cfg.training, 'grad_accum_steps', 1)
        
        logger.info(f"Trainer initialized for {model_type} on device: {self.device}")
    
    def _setup_scheduler(self):
        """Настраивает планировщик learning rate"""
        warmup_epochs = getattr(self.cfg.training, 'warmup_epochs', 0)
        
        if warmup_epochs > 0:
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, 
                T_0=10,
                T_mult=2,
                eta_min=getattr(self.cfg.training, 'learning_rate', 1e-4) * 0.01
            )
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=0.5, 
                patience=10, 
                verbose=True,
                min_lr=getattr(self.cfg.training, 'learning_rate', 1e-4) * 0.001
            )
        
        return scheduler
    
    def _get_learning_rate(self, epoch, batch_idx, batches_per_epoch):
        """Вычисляет learning rate с warmup"""
        warmup_epochs = getattr(self.cfg.training, 'warmup_epochs', 0)
        
        if epoch < warmup_epochs:
            # Linear warmup
            progress = (epoch * batches_per_epoch + batch_idx) / (warmup_epochs * batches_per_epoch)
            lr = self.cfg.training.learning_rate * progress
        else:
            lr = self.cfg.training.learning_rate
            
        return lr
    
    def _setup_device(self):
        """Настраивает устройство для обучения"""
        device_config = getattr(self.cfg.training, 'device', 'auto')
        
        if device_config == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                logger.info("🎯 CUDA доступна, используем GPU")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device('mps')
                logger.info("🍎 MPS доступен, используем Apple Silicon GPU")
            else:
                device = torch.device('cpu')
                logger.info("🖥️ GPU не доступен, используем CPU")
        elif device_config == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
        elif device_config == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        
        if device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"💾 GPU: {gpu_name}, Память: {gpu_memory:.1f} GB")
        
        return device
    
    def _setup_loss_function(self, train_loader):
        """Настраивает функцию потерь"""
        try:
            all_labels = []
            for _, labels in train_loader:
                all_labels.extend(labels.numpy())
            
            all_labels = np.array(all_labels).flatten()
            
            pos_count = all_labels.sum()
            neg_count = len(all_labels) - pos_count
            
            if pos_count > 0 and neg_count > 0:
                pos_weight = torch.tensor([neg_count / pos_count]).to(self.device)
            else:
                pos_weight = torch.tensor([1.0]).to(self.device)
            
            logger.info(f"Class distribution: {neg_count} negative, {pos_count} positive")
            logger.info(f"Using pos_weight: {pos_weight.item()}")
            
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
        except Exception as e:
            logger.warning(f"Error setting up loss: {e}")
            self.criterion = nn.BCEWithLogitsLoss()
    
    def mixup_data(self, x, y, alpha=0.4):
        """Mixup аугментация данных"""
        if alpha <= 0:
            return x, y, y, 1.0
        
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam

    def _train_epoch_advanced(self, train_loader: DataLoader, epoch: int):
        """Улучшенная эпоха обучения с advanced аугментациями"""
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(self.device), labels.to(self.device)
            
            # Применяем аугментации с вероятностью 50%
            if np.random.random() > 0.5 and self.training:
                augmented_data = []
                for seq in data:
                    seq_np = seq.cpu().numpy()
                    # Применяем случайную аугментацию
                    aug_method = np.random.choice(['shift', 'noise', 'mask'])
                    if aug_method == 'shift':
                        seq_np = DataAugmentation.temporal_shift(seq_np)
                    elif aug_method == 'noise':
                        seq_np = DataAugmentation.feature_noise(seq_np)
                    else:
                        seq_np = DataAugmentation.random_mask(seq_np)
                    augmented_data.append(torch.tensor(seq_np))
                data = torch.stack(augmented_data).to(self.device)
            
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs.squeeze(), labels.squeeze())
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            if self.use_ema:
                self.ema.update()
            
            total_loss += loss.item()
            
            # Сохраняем предсказания
            with torch.no_grad():
                preds_proba = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(preds_proba)
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        metrics = self.compute_enhanced_metrics(
            torch.tensor(all_preds), 
            torch.tensor(all_labels)
        )
        
        return avg_loss, metrics
    
    def compute_precision_optimized_metrics(self, preds, targets):
        """Метрики с оптимизацией для precision - ИСПРАВЛЕННАЯ ФУНКЦИЯ"""
        try:
            # preds уже должны быть probabilities
            preds_proba = preds.numpy().flatten() if isinstance(preds, torch.Tensor) else preds.flatten()
            targets_np = targets.numpy().flatten() if isinstance(targets, torch.Tensor) else targets.flatten()
            targets_np = targets_np.astype(int)
            
            if len(np.unique(targets_np)) < 2:
                return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'optimal_threshold': 0.5}
            
            best_threshold = 0.7
            best_precision = 0
            
            for thresh in np.arange(0.5, 0.95, 0.02):
                preds_class = (preds_proba > thresh).astype(int)
                if len(np.unique(preds_class)) < 2:
                    continue
                    
                precision = precision_score(targets_np, preds_class, zero_division=0)
                recall = recall_score(targets_np, preds_class, zero_division=0)
                
                if precision > best_precision and recall > 0.3:
                    best_precision = precision
                    best_threshold = thresh
            
            preds_class = (preds_proba > best_threshold).astype(int)
            
            precision = precision_score(targets_np, preds_class, zero_division=0)
            recall = recall_score(targets_np, preds_class, zero_division=0)
            f1 = f1_score(targets_np, preds_class, zero_division=0)
            
            logger.info(f"🎯 Precision-optimized threshold: {best_threshold:.3f}")
            logger.info(f"📊 Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
            
            return {
                'f1': f1, 'precision': precision, 'recall': recall,
                'optimal_threshold': best_threshold,
                'tp': np.sum((preds_class == 1) & (targets_np == 1)),
                'fp': np.sum((preds_class == 1) & (targets_np == 0)),
                'fn': np.sum((preds_class == 0) & (targets_np == 1)),
                'tn': np.sum((preds_class == 0) & (targets_np == 0))
            }
        except Exception as e:
            logger.warning(f"Error in precision-optimized metrics: {e}")
            return self._calculate_metrics(preds, targets)

    def compute_enhanced_metrics(self, preds, targets):
        """Улучшенные метрики с оптимизацией для precision"""
        try:
            preds_proba = torch.sigmoid(preds).cpu().numpy().flatten()
            targets_np = targets.cpu().numpy().flatten().astype(int)
            
            # Оптимизация порога для F1-score с учетом precision
            best_threshold = 0.5
            best_score = 0
            
            for thresh in np.arange(0.3, 0.9, 0.01):  # Более точный поиск
                preds_class = (preds_proba > thresh).astype(int)
                if len(np.unique(preds_class)) < 2:
                    continue
                    
                precision = precision_score(targets_np, preds_class, zero_division=0)
                recall = recall_score(targets_np, preds_class, zero_division=0)
                f1 = f1_score(targets_np, preds_class, zero_division=0)
                
                # Комбинированная метрика с упором на precision
                score = 0.4 * precision + 0.4 * f1 + 0.2 * recall
                
                if score > best_score:
                    best_score = score
                    best_threshold = thresh
            
            # Применяем лучший порог
            preds_class = (preds_proba > best_threshold).astype(int)
            
            precision = precision_score(targets_np, preds_class, zero_division=0)
            recall = recall_score(targets_np, preds_class, zero_division=0)
            f1 = f1_score(targets_np, preds_class, zero_division=0)
            
            logger.info(f"🎯 Optimal threshold: {best_threshold:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
            
            return {
                'f1': f1, 'precision': precision, 'recall': recall,
                'optimal_threshold': best_threshold,
                # ... остальные метрики
            }
        except Exception as e:
            logger.warning(f"Error computing metrics: {e}")
            return self._calculate_metrics(torch.sigmoid(preds).cpu().numpy(), targets.cpu().numpy())

    def setup_class_weights(self, train_loader):
        """Вычисляет веса классов для борьбы с дисбалансом"""
        if not getattr(self.cfg.training, 'use_class_weights', False):
            return
            
        try:
            # Собираем все метки
            all_labels = []
            for _, labels in train_loader:
                all_labels.extend(labels.numpy())
            
            all_labels = np.array(all_labels).flatten()
            
            # Вычисляем веса классов
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(all_labels),
                y=all_labels
            )
            
            # Конвертируем в tensor
            self.class_weights = torch.FloatTensor(class_weights).to(self.device)
            logger.info(f"Class weights computed: {class_weights}")
            
        except Exception as e:
            logger.warning(f"Could not compute class weights: {e}")
    

    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Основной цикл обучения"""
        self._setup_loss_function(train_loader)
        self.setup_class_weights(train_loader)
        
        logger.info(f"🚀 Начало обучения {self.model_type} модели")
        logger.info(f"📊 Размер тренировочной выборки: {len(train_loader.dataset)}")
        logger.info(f"📊 Размер валидационной выборки: {len(val_loader.dataset)}")
        
        best_val_loss = float('inf')
        best_val_f1 = 0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            
            # Training phase
            train_loss, train_metrics = self._train_epoch_stable(train_loader, epoch)
            
            # Validation phase - ИСПРАВЛЕНО: убрана лишняя строка с compute_precision_optimized_metrics
            val_loss, val_metrics = self._validate_epoch(val_loader, epoch)
            
            # Learning rate scheduling
            if hasattr(self.scheduler, 'step'):
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start
            
            # Логирование прогресса
            self._log_epoch_progress(epoch, train_loss, val_loss, train_metrics, val_metrics, epoch_time, current_lr)
            
            # Early stopping по комбинации метрик
            current_score = val_metrics['f1'] * 0.7 + (1 - val_loss) * 0.3
            
            if current_score > best_val_f1:
                best_val_f1 = current_score
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                
                if self.use_ema:
                    self.ema.apply_shadow()
                    best_model_state = self.model.state_dict().copy()
                    self.ema.restore()
                    
                logger.info(f"🎯 Новая лучшая модель! Score: {current_score:.4f}, Val F1: {val_metrics['f1']:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"🛑 Early stopping на эпохе {epoch + 1}")
                    break
        
        # Восстанавливаем лучшую модель
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info("✅ Загружена лучшая модель")
        
        # Финальная оценка
        final_val_loss, final_metrics = self._validate_epoch(val_loader, self.epochs)
        final_metrics = self._add_comprehensive_metrics(val_loader, final_metrics)
        
        return {
            'history': self.history,
            'best_val_loss': best_val_loss,
            'best_val_f1': best_val_f1,
            'final_metrics': final_metrics,
            'metrics': final_metrics
        }
    
    def _train_epoch_stable(self, train_loader: DataLoader, epoch: int) -> Tuple[float, Dict]:
        """Стабилизированная эпоха обучения"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        self.optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1} Training', leave=False)
        
        for batch_idx, (data, labels) in enumerate(progress_bar):
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(data)
            loss = self.criterion(outputs.squeeze(), labels.squeeze())
            
            # Gradient accumulation
            loss = loss / self.grad_accum_steps
            loss.backward()
            
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if hasattr(self.cfg.training, 'gradient_clip') and self.cfg.training.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.cfg.training.gradient_clip
                    )
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                if self.use_ema:
                    self.ema.update()
            
            total_loss += loss.item() * self.grad_accum_steps
            
            # Сохраняем предсказания и метки
            with torch.no_grad():
                preds_proba = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(preds_proba)
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        metrics = self._calculate_metrics(all_preds, all_labels)
        
        return avg_loss, metrics
    
    def _train_epoch_stable(self, train_loader: DataLoader, epoch: int) -> Tuple[float, Dict]:
        """Стабилизированная эпоха обучения с диагностикой"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_outputs = []  # Сохраняем raw outputs для диагностики
        
        self.optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1} Training', leave=False)
        
        for batch_idx, (data, labels) in enumerate(progress_bar):
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(data)
            
            # Сохраняем raw outputs для диагностики
            all_outputs.extend(outputs.detach().cpu().numpy())
            
            loss = self.criterion(outputs.squeeze(), labels.squeeze())
            
            # Gradient accumulation
            loss = loss / self.grad_accum_steps
            loss.backward()
            
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if hasattr(self.cfg.training, 'gradient_clip') and self.cfg.training.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.cfg.training.gradient_clip
                    )
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                if self.use_ema:
                    self.ema.update()
            
            total_loss += loss.item() * self.grad_accum_steps
            
            # Сохраняем предсказания и метки
            with torch.no_grad():
                preds_proba = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(preds_proba)
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        
        # ДИАГНОСТИКА: Подробный анализ предсказаний
        all_preds_array = np.array(all_preds).flatten()
        all_labels_array = np.array(all_labels).flatten()
        all_outputs_array = np.array(all_outputs).flatten()
        
        logger.info(f"🔍 ДИАГНОСТИКА TRAIN Epoch {epoch+1}:")
        logger.info(f"   Outputs range: [{all_outputs_array.min():.4f}, {all_outputs_array.max():.4f}]")
        logger.info(f"   Preds proba range: [{all_preds_array.min():.4f}, {all_preds_array.max():.4f}]")
        logger.info(f"   Labels distribution: 0={np.sum(all_labels_array==0)}, 1={np.sum(all_labels_array==1)}")
        logger.info(f"   Preds > 0.5: {np.sum(all_preds_array > 0.5)}")
        logger.info(f"   Preds <= 0.5: {np.sum(all_preds_array <= 0.5)}")
        
        # Проверяем разные пороги
        for threshold in [0.3, 0.5, 0.7]:
            preds_binary = (all_preds_array > threshold).astype(int)
            accuracy = np.mean(preds_binary == all_labels_array)
            tp = np.sum((preds_binary == 1) & (all_labels_array == 1))
            fp = np.sum((preds_binary == 1) & (all_labels_array == 0))
            fn = np.sum((preds_binary == 0) & (all_labels_array == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            logger.info(f"   Threshold {threshold}: Acc={accuracy:.4f}, F1={f1:.4f}, TP={tp}, FP={fp}, FN={fn}")
        
        metrics = self._calculate_metrics(all_preds, all_labels)
        return avg_loss, metrics
    
    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> Tuple[float, Dict]:
        """Валидация после эпохи"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in val_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(data)
                
                loss = self.criterion(outputs.squeeze(), labels.squeeze())
                total_loss += loss.item()
                
                all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        metrics = self._calculate_metrics(all_preds, all_labels)
        
        return avg_loss, metrics
    
    def _calculate_metrics(self, preds: List, labels: List) -> Dict:
        """Вычисляет метрики качества"""
        preds_array = np.array(preds).flatten()
        labels_array = np.array(labels).flatten()
        
        preds_binary = (preds_array > 0.5).astype(int)
        
        accuracy = np.mean(preds_binary == labels_array)
        
        tp = np.sum((preds_binary == 1) & (labels_array == 1))
        fp = np.sum((preds_binary == 1) & (labels_array == 0))
        fn = np.sum((preds_binary == 0) & (labels_array == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    def _calculate_metrics(self, preds: List, labels: List) -> Dict:
        """Вычисляет метрики качества"""
        preds_array = np.array(preds).flatten()
        labels_array = np.array(labels).flatten()
        
        # Бинаризуем предсказания
        preds_binary = (preds_array > 0.5).astype(int)
        
        # Базовые метрики
        accuracy = np.mean(preds_binary == labels_array)
        
        # Precision, Recall, F1
        tp = np.sum((preds_binary == 1) & (labels_array == 1))
        fp = np.sum((preds_binary == 1) & (labels_array == 0))
        fn = np.sum((preds_binary == 0) & (labels_array == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    def cutmix_data(self, x, y, alpha=1.0):
        """CutMix аугментация"""
        if alpha <= 0:
            return x, y, y, 1.0
        
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)
        
        # Generate random bounding box
        H, W = x.size(1), x.size(2)
        r_x = np.random.randint(0, W)
        r_y = np.random.randint(0, H)
        r_w = int(W * np.sqrt(1 - lam))
        r_h = int(H * np.sqrt(1 - lam))
        
        x1 = max(0, r_x - r_w // 2)
        x2 = min(W, r_x + r_w // 2)
        y1 = max(0, r_y - r_h // 2)
        y2 = min(H, r_y + r_h // 2)
        
        mixed_x = x.clone()
        mixed_x[:, y1:y2, x1:x2] = x[index, y1:y2, x1:x2]
        
        # Adjust lambda to exact pixel ratio
        lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
        
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def _add_comprehensive_metrics(self, val_loader: DataLoader, metrics: Dict) -> Dict:
        """Добавляет полные метрики включая ROC-AUC"""
        try:
            from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
            import numpy as np
            
            self.model.eval()
            all_preds = []
            all_labels = []
            all_logits = []
            
            with torch.no_grad():
                for data, labels in val_loader:
                    data = data.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(data)
                    
                    all_logits.extend(outputs.cpu().numpy())
                    all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            preds_array = np.array(all_preds).flatten()
            logits_array = np.array(all_logits).flatten()
            labels_array = np.array(all_labels).flatten()
            
            # ROC-AUC
            if len(np.unique(labels_array)) > 1:
                auc_score = roc_auc_score(labels_array, preds_array)
                metrics['auc'] = auc_score
                
                # ROC curve data
                fpr, tpr, _ = roc_curve(labels_array, preds_array)
                metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr}
                
                # Precision-Recall curve
                precision, recall, _ = precision_recall_curve(labels_array, preds_array)
                avg_precision = average_precision_score(labels_array, preds_array)
                metrics['pr_curve'] = {'precision': precision, 'recall': recall}
                metrics['average_precision'] = avg_precision
            
            # Дополнительные метрики
            tn = np.sum((preds_array <= 0.5) & (labels_array == 0))
            fp = np.sum((preds_array > 0.5) & (labels_array == 0))
            fn = np.sum((preds_array <= 0.5) & (labels_array == 1))
            tp = np.sum((preds_array > 0.5) & (labels_array == 1))
            
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            balanced_accuracy = (metrics.get('recall', 0) + specificity) / 2
            
            metrics.update({
                'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
                'specificity': specificity,
                'balanced_accuracy': balanced_accuracy,
                'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
                'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0
            })
            
        except Exception as e:
            logger.warning(f"Ошибка вычисления расширенных метрик: {e}")
        
        return metrics
    
    def _log_epoch_progress(self, epoch: int, train_loss: float, val_loss: float,
                           train_metrics: Dict, val_metrics: Dict, epoch_time: float, lr: float):
        """Логирует прогресс эпохи"""
        # Обновляем историю
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_f1'].append(train_metrics['f1'])
        self.history['val_f1'].append(val_metrics['f1'])
        self.history['train_accuracy'].append(train_metrics['accuracy'])
        self.history['val_accuracy'].append(val_metrics['accuracy'])
        self.history['train_precision'].append(train_metrics['precision'])
        self.history['val_precision'].append(val_metrics['precision'])
        self.history['train_recall'].append(train_metrics['recall'])
        self.history['val_recall'].append(val_metrics['recall'])
        self.history['learning_rate'].append(lr)
        
        # Логируем
        logger.info(f"📈 Epoch {epoch+1:3d} | Time: {epoch_time:.1f}s | LR: {lr:.6f}")
        logger.info(f"   Train: Loss {train_loss:.4f} | F1 {train_metrics['f1']:.4f} | "
                   f"Acc {train_metrics['accuracy']:.4f} | Prec {train_metrics['precision']:.4f} | "
                   f"Rec {train_metrics['recall']:.4f}")
        logger.info(f"   Val:   Loss {val_loss:.4f} | F1 {val_metrics['f1']:.4f} | "
                   f"Acc {val_metrics['accuracy']:.4f} | Prec {val_metrics['precision']:.4f} | "
                   f"Rec {val_metrics['recall']:.4f}")
        logger.info(f"   Confusion Matrix: TP={val_metrics.get('tp', 0)}, FP={val_metrics.get('fp', 0)}, FN={val_metrics.get('fn', 0)}")
        logger.info(f"   Precision/Recall: {val_metrics['precision']:.3f}/{val_metrics['recall']:.3f}")
        logger.info(f"   Optimal Threshold: {val_metrics.get('optimal_threshold', 0.5):.3f}")
    
    def save_model(self, filename: str):
        """Сохраняет модель"""
        model_path = Path('outputs/models') / filename
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': self.cfg
        }, model_path)
        
        logger.info(f"💾 Модель сохранена: {model_path}")