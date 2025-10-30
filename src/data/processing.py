import numpy as np
import pandas as pd
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def merge_and_deduplicate_datasets(icgc_data: List[Dict], tcga_data: List[Dict], 
                                 data_type: str = 'sequence') -> List[Dict]:
    """
    Объединяет и проверяет на дубликаты данные от ICGC и TCGA
    """
    logger.info(f"Объединение данных {data_type.upper()}")
    
    def create_sample_identifier(sample):
        return f"{sample['patient_id']}_{sample['chromosome']}_{sample['region_start']}_{sample['region_end']}"
    
    icgc_ids = [create_sample_identifier(sample) for sample in icgc_data]
    tcga_ids = [create_sample_identifier(sample) for sample in tcga_data]
    
    logger.info(f"Уникальных ICGC samples: {len(set(icgc_ids))} из {len(icgc_data)}")
    logger.info(f"Уникальных TCGA samples: {len(set(tcga_ids))} из {len(tcga_data)}")
    
    icgc_set = set(icgc_ids)
    tcga_set = set(tcga_ids)
    intersection = icgc_set.intersection(tcga_set)
    
    logger.info(f"Пересечение ICGC-TCGA: {len(intersection)} samples")
    
    unique_tcga_data = []
    seen_ids = icgc_set.copy()
    
    for sample, sample_id in zip(tcga_data, tcga_ids):
        if sample_id not in seen_ids:
            unique_tcga_data.append(sample)
            seen_ids.add(sample_id)
    
    merged_data = icgc_data + unique_tcga_data
    
    labels = [sample['label'] for sample in merged_data]
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    logger.info("Распределение классов в объединенном датасете:")
    for label, count in zip(unique_labels, counts):
        logger.info(f"  {label}: {count} ({count/len(merged_data)*100:.1f}%)")
    
    return merged_data

def binarize_labels(data: List[Dict], positive_classes: List[str]) -> List[Dict]:
    """
    Бинаризирует метки для бинарной классификации
    """
    if len(data) == 0:
        logger.warning("⚠️  Пустой датасет для бинаризации")
        return []
    
    binary_data = []
    
    # ДИАГНОСТИКА
    unique_labels_before = set(sample['label'] for sample in data)
    logger.info(f"Уникальные метки до бинаризации: {sorted(unique_labels_before)}")
    logger.info(f"Positive classes из конфига: {positive_classes}")
    
    # Если метки уже бинарные (0 и 1)
    if unique_labels_before == {0, 1} or unique_labels_before == {0.0, 1.0}:
        logger.info("Метки уже бинарные (0/1). Используем как есть.")
        return data  # Возвращаем данные без изменений
    
    # Для небинарных меток применяем стандартную логику
    positive_set = set(str(cls) for cls in positive_classes)
    
    for sample in data:
        original_label = sample['label']
        label_str = str(original_label)
        
        if label_str in positive_set:
            binary_label = 1
        else:
            binary_label = 0
        
        binary_sample = sample.copy()
        binary_sample['label'] = binary_label
        binary_data.append(binary_sample)
    
    # Подсчет результатов
    if len(binary_data) > 0:
        labels_after = [sample['label'] for sample in binary_data]
        positive_count = sum(labels_after)
        negative_count = len(labels_after) - positive_count
        
        logger.info(f"Бинаризация завершена: {positive_count} положительных, {negative_count} отрицательных")
        logger.info(f"Дисбаланс: {positive_count/len(binary_data)*100:.1f}% положительных")
    else:
        logger.warning("⚠️  После бинаризации получен пустой датасет")
    
    return binary_data