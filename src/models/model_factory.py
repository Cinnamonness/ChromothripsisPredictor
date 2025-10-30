# src/models/model_factory.py
import torch.nn as nn
import logging
from typing import Dict, Any
from omegaconf import DictConfig

# Импорт из отдельных файлов
from src.models.sequence_models import EnhancedBiLSTM, ChromothripsisBiLSTM, ResidualLSTM
from src.models.transformer_models import TransformerGenomicModel
from src.models.cnn_models import MultiScaleCNN
from src.models.hybrid_models import HybridCNNTransformer

logger = logging.getLogger(__name__)

class ModelFactory:
    
    @staticmethod
    def create_model(model_type: str, config: DictConfig) -> nn.Module:
        logger.info(f"Создание модели типа: {model_type}")
        
        model_config = ModelFactory._prepare_model_config(model_type, config)
        
        if model_type == 'sequence':
            return ModelFactory._create_sequence_model(model_config)
        elif model_type == 'advanced_sequence':
            return ModelFactory._create_advanced_sequence_model(model_config)
        elif model_type == 'transformer':
            return ModelFactory._create_transformer_model(model_config)
        elif model_type == 'hybrid':
            return ModelFactory._create_hybrid_model(model_config)
        elif model_type == 'residual_lstm':
            return ModelFactory._create_residual_lstm_model(model_config)
        elif model_type == 'multiscale_cnn':
            return ModelFactory._create_multiscale_cnn_model(model_config)
        else:
            raise ValueError(f"Неизвестный тип модели: {model_type}")
    
    @staticmethod
    def _prepare_model_config(model_type: str, config: DictConfig) -> Dict[str, Any]:
        """Подготавливает конфигурацию модели с учетом 22 фич"""
        model_config = {}
        
        # Базовые параметры
        model_cfg = getattr(config, 'model', {})
        sequence_cfg = getattr(model_cfg, 'sequence_model', {})
        
        # Параметры для 22 фич
        model_config.update({
            'input_dim': getattr(sequence_cfg, 'input_dim', 22),  # 22 фичи!
            'hidden_dim': getattr(sequence_cfg, 'hidden_dim', 128),
            'num_layers': getattr(sequence_cfg, 'num_layers', 2),
            'dropout_rate': getattr(sequence_cfg, 'dropout_rate', 0.3),
            'sequence_length': getattr(config.data.preprocessing, 'sequence_length', 200)
        })
        
        logger.info(f"Конфигурация модели {model_type}: input_dim={model_config['input_dim']}")
        return model_config
    
    @staticmethod
    def _create_sequence_model(config: Dict[str, Any]) -> nn.Module:
        """Создает базовую BiLSTM модель"""
        logger.info("Создание EnhancedBiLSTM модели с 22 фичами")
        return EnhancedBiLSTM(config)
    
    @staticmethod
    def _create_advanced_sequence_model(config: Dict[str, Any]) -> nn.Module:
        """Создает продвинутую LSTM модель"""
        logger.info("Создание ChromothripsisBiLSTM модели с 22 фичами")
        return ChromothripsisBiLSTM(config)
    
    @staticmethod
    def _create_transformer_model(config: Dict[str, Any]) -> nn.Module:
        """Создает Transformer модель для геномных данных"""
        logger.info("Создание TransformerGenomicModel с 22 фичами")
        return TransformerGenomicModel(
            input_dim=config['input_dim'],
            d_model=config.get('d_model', 256),
            nhead=config.get('nhead', 8),
            num_layers=config.get('num_layers', 6),
            dropout=config.get('dropout_rate', 0.2)
        )
    
    @staticmethod
    def _create_hybrid_model(config: Dict[str, Any]) -> nn.Module:
        """Создает гибридную CNN-Transformer модель"""
        logger.info("Создание HybridCNNTransformer с 22 фичами")
        return HybridCNNTransformer(
            input_dim=config['input_dim'],
            cnn_channels=config.get('cnn_channels', 128),
            d_model=config.get('d_model', 256),
            nhead=config.get('nhead', 8),
            num_layers=config.get('num_layers', 4),
            dropout=config.get('dropout_rate', 0.2)
        )
    
    @staticmethod
    def _create_residual_lstm_model(config: Dict[str, Any]) -> nn.Module:
        """Создает Residual LSTM модель"""
        logger.info("Создание ResidualLSTM с 22 фичами")
        return ResidualLSTM(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config.get('num_layers', 4),
            dropout=config.get('dropout_rate', 0.3)
        )
    
    @staticmethod
    def _create_multiscale_cnn_model(config: Dict[str, Any]) -> nn.Module:
        """Создает мультимасштабную CNN модель"""
        logger.info("Создание MultiScaleCNN с 22 фичами")
        return MultiScaleCNN(
            input_dim=config['input_dim'],
            base_channels=config.get('base_channels', 64),
            dropout=config.get('dropout_rate', 0.3)
        )


# Функции для обратной совместимости
def create_sequence_model(config: Dict[str, Any]) -> nn.Module:
    return ModelFactory._create_sequence_model(config)

def create_model_from_config(full_config: DictConfig) -> Dict[str, nn.Module]:
    models = {}
    
    for model_type in getattr(full_config.data, 'output_formats', ['sequence']):
        try:
            model = ModelFactory.create_model(model_type, full_config)
            models[model_type] = model
            logger.info(f"Модель {model_type} создана успешно")
        except Exception as e:
            logger.error(f"Ошибка создания модели {model_type}: {e}")
            continue
    
    return models