#!/usr/bin/env python3
"""
Основной скрипт для обучения моделей детекции хромотрипсиса
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import logging
import os
import sys
from typing import Dict
import time
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.preprocessor import SVDataPreprocessor
from src.data.processing import merge_and_deduplicate_datasets, binarize_labels
from src.data.datasets import create_data_loaders
from src.models.model_factory import ModelFactory
from src.training.trainer import Trainer
from src.visualization.plots import (
    plot_training_history, 
    plot_roc_curves, 
    plot_metrics_comparison,
    plot_confusion_matrices,
    plot_precision_recall_curves
)

logger = logging.getLogger(__name__)

def print_final_results(results: Dict):
    """Печать финальных результатов в консоль"""
    print("\n" + "="*60)
    print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ ОБУЧЕНИЯ")
    print("="*60)
    
    for model_type, result in results.items():
        # Получаем метрики из правильного места
        if 'final_metrics' in result:
            metrics = result['final_metrics']
        elif 'metrics' in result:
            metrics = result['metrics']
        else:
            print(f"\n⚠️  {model_type.upper()} MODEL: Метрики не найдены")
            continue
        
        print(f"\n📊 {model_type.upper()} MODEL:")
        print(f"   F1 Score:       {metrics.get('f1', 'N/A'):.4f}")
        print(f"   Accuracy:       {metrics.get('accuracy', 'N/A'):.4f}")
        print(f"   Precision:      {metrics.get('precision', 'N/A'):.4f}")
        print(f"   Recall:         {metrics.get('recall', 'N/A'):.4f}")
        
        # Дополнительные метрики если есть
        if 'auc' in metrics:
            print(f"   AUC:            {metrics['auc']:.4f}")
        if 'specificity' in metrics:
            print(f"   Specificity:    {metrics['specificity']:.4f}")
        if 'balanced_accuracy' in metrics:
            print(f"   Balanced Acc:   {metrics['balanced_accuracy']:.4f}")
        
        # Анализ качества модели
        f1_score = metrics.get('f1', 0)
        accuracy = metrics.get('accuracy', 0)
        
        if f1_score > 0.9 and accuracy > 0.95:
            print("   ✅ ОТЛИЧНОЕ КАЧЕСТВО")
        elif f1_score > 0.7 and accuracy > 0.8:
            print("   👍 ХОРОШЕЕ КАЧЕСТВО")
        elif f1_score > 0.5:
            print("   ⚠️  СРЕДНЕЕ КАЧЕСТВО")
        else:
            print("   ❗ НИЗКОЕ КАЧЕСТВО")
            
        # Проверка на переобучение
        if f1_score > 0.95 and accuracy > 0.98:
            print("   ⚠️  ВОЗМОЖНО ПЕРЕОБУЧЕНИЕ!")
        
        # Confusion Matrix если есть
        if all(k in metrics for k in ['tp', 'fp', 'fn', 'tn']):
            print(f"   Confusion Matrix: TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, TN={metrics['tn']}")
        
        # Лучшая потеря если есть
        if 'best_val_loss' in result:
            print(f"   Best Val Loss:  {result['best_val_loss']:.4f}")

def load_or_process_data(preprocessor, data_format, cfg, force_reprocess=False):
    """
    Загружает препроцессированные данные если они есть, 
    иначе обрабатывает и сохраняет их
    """
    cache_filename = f"{data_format}_22features_processed_data.pkl"
    
    # Если не форсируем переобработку и файл существует - загружаем
    if not force_reprocess:
        cached_data = preprocessor.load_processed_data(cache_filename)
        if cached_data is not None:
            logger.info(f"Загружены препроцессированные данные с 22 фичами для формата {data_format} ({len(cached_data)} samples)")
            return cached_data
        else:
            logger.info(f"Препроцессированные данные с 22 фичами для формата {data_format} не найдены, начинаем обработку...")
    
    # Иначе обрабатываем данные с 22 фичами
    logger.info(f"Обработка {data_format} данных с 22 фичами...")
    format_start = time.time()
    
    # Обрабатываем ICGC данные (тренировочные) с 22 фичами
    icgc_data = preprocessor.process_patient_regions(
        sv_directory=cfg.data.data_paths.sv_directories.icgc,
        output_format=data_format,
        is_training=True
    )
    icgc_time = time.time()
    logger.info(f"ICGC данные с 22 фичами обработаны за {icgc_time - format_start:.2f} сек ({len(icgc_data)} samples)")
    
    # Обрабатываем TCGA данные (тестовые) с 22 фичами
    tcga_data = preprocessor.process_patient_regions(
        sv_directory=cfg.data.data_paths.sv_directories.tcga,
        output_format=data_format,
        is_training=False
    )
    tcga_time = time.time()
    logger.info(f"TCGA данные с 22 фичами обработаны за {tcga_time - icgc_time:.2f} сек ({len(tcga_data)} samples)")
    
    # Если оба датасета пустые, возвращаем пустой список
    if len(icgc_data) == 0 and len(tcga_data) == 0:
        logger.warning(f"⚠️  Нет данных для формата {data_format}!")
        return []
    
    # Объединяем и дедуплицируем
    merged_data = merge_and_deduplicate_datasets(
        icgc_data, tcga_data, data_format
    )
    merge_time = time.time()
    logger.info(f"Данные с 22 фичами объединены за {merge_time - tcga_time:.2f} сек ({len(merged_data)} samples)")
    
    # Бинаризуем метки если нужно и если есть данные
    if cfg.data.preprocessing.binarize_labels and len(merged_data) > 0:
        merged_data = binarize_labels(
            merged_data, 
            cfg.data.preprocessing.positive_classes
        )
        logger.info(f"Метки бинаризованы за {time.time() - merge_time:.2f} сек")
    
    # Сохраняем только если есть данные
    if len(merged_data) > 0:
        logger.info(f"Сохранение обработанных данных с 22 фичами для формата {data_format}...")
        preprocessor.save_processed_data(merged_data, cache_filename)
        
        # Сохраняем scalers если это тренировочные данные
        # ИСПРАВЛЕНИЕ: правильная проверка scaler
        if preprocessor.feature_extractor.scaler is not None:
            preprocessor.save_scalers()
    else:
        logger.warning(f"⚠️  Нет данных для сохранения для формата {data_format}")
    
    total_time = time.time() - format_start
    logger.info(f"Обработка формата {data_format} завершена за {total_time:.2f} сек ({len(merged_data)} samples)")
    
    return merged_data

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    """Основная функция обучения"""
    # Создаем директории
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('./data/processed_data', exist_ok=True)
    
    # Устанавливаем seed для воспроизводимости
    torch.manual_seed(42)
    np.random.seed(42)
    
    logger.info("Запуск обучения детекции хромотрипсиса")
    logger.info(f"Конфигурация:\n{OmegaConf.to_yaml(cfg)}")
    
    try:
        start_time = time.time()
        
        preprocessor = SVDataPreprocessor(cfg)
        logger.info("Начало обработки данных...")
        processed_data = {}
        
        # Загружаем scalers если есть
        preprocessor.load_scalers()
        
        # Флаг для принудительной переобработки (можно вынести в конфиг)
        force_reprocess = getattr(cfg.data, 'force_reprocess', False)
        
        for data_format in cfg.data.output_formats:
            # Загружаем или обрабатываем данные для каждого формата
            merged_data = load_or_process_data(
                preprocessor, data_format, cfg, force_reprocess
            )
            processed_data[data_format] = merged_data
        
        data_processing_time = time.time() - start_time
        logger.info(f"Обработка данных завершена за {data_processing_time:.2f} сек")
        
        # Создание DataLoader'ов
        logger.info("Создание DataLoader'ов...")
        loader_start = time.time()
        loaders = create_data_loaders(
            processed_data.get('sequence', []),
            processed_data.get('cnn', []),
            processed_data.get('graph', []),
            cfg
        )
        logger.info(f"DataLoader'ы созданы за {time.time() - loader_start:.2f} сек")
        
        results = {}
        
        for data_format in cfg.data.output_formats:
            if data_format not in loaders or len(processed_data.get(data_format, [])) == 0:
                logger.warning(f"Нет данных для формата {data_format}, пропускаем...")
                continue
                
            logger.info(f"Обучение {data_format} модели...")
            
            model_start = time.time()
            model = ModelFactory.create_model(data_format, cfg)
            trainer = Trainer(model, cfg, data_format)
            train_loader, val_loader = loaders[data_format]
            
            logger.info(f"Начало обучения {data_format} модели...")
            model_results = trainer.train(train_loader, val_loader)
            trainer.save_model(f"best_{data_format}_model.pth")
            
            results[data_format] = model_results
            logger.info(f"Обучение {data_format} модели завершено за {time.time() - model_start:.2f} сек")
        
        # Печать финальных результатов в консоль
        print_final_results(results)
        
        # В функции main, замените блок визуализации на:
        logger.info("Генерация графиков...")
        plot_start = time.time()

        histories = [results[fmt]['history'] for fmt in results.keys()]
        model_names = list(results.keys())

        plot_training_history(histories, model_names)
        plot_roc_curves(results, model_names)
        plot_metrics_comparison(results)
        plot_confusion_matrices(results)
        plot_precision_recall_curves(results, model_names)

        logger.info(f"Графики сгенерированы за {time.time() - plot_start:.2f} сек")
        total_time = time.time() - start_time
        logger.info(f"Обучение завершено успешно! Общее время: {total_time:.2f} сек ({total_time/60:.2f} мин)")
        
        return results
        
    except Exception as e:
        logger.error(f"Критическая ошибка во время обучения: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()