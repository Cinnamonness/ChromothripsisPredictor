# src/data/feature_extractor.py
import pandas as pd
import numpy as np
import json
from typing import List, Dict, Tuple
import logging
from sklearn.preprocessing import MinMaxScaler
import torch

logger = logging.getLogger(__name__)

class SVFeatureExtractor:
    """Класс для извлечения 22 фич из SV данных"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.scaler = MinMaxScaler()
        self.feature_dim = 22
        self.strand_map = {'+': 1, '-': -1}
        
    def chrom_to_int(self, ch):
        """Конвертирует хромосому в числовой формат"""
        if pd.isna(ch): 
            return -1
        chs = str(ch).lower().replace('chr','')
        if chs in ['x','y','mt','m']:
            return {'x':23,'y':24,'mt':25,'m':25}[chs]
        try:
            return int(chs)
        except ValueError:
            return -1
    
    def preprocess_sv_data(self, sv_df: pd.DataFrame) -> pd.DataFrame:
        """Предобработка SV данных и создание 22 фич"""
        logger.info("Предобработка SV данных...")
        
        # Базовые преобразования
        sv_df['strand1'] = sv_df['strand1'].map(self.strand_map).fillna(0).astype(int)
        sv_df['strand2'] = sv_df['strand2'].map(self.strand_map).fillna(0).astype(int)
        sv_df['svclass'] = sv_df['svclass'].fillna('Unknown').astype('category')
        
        # Кодирование svclass
        svclass_mapping = {cat: i for i, cat in enumerate(sv_df['svclass'].cat.categories)}
        sv_df['svclass'] = sv_df['svclass'].cat.codes
        
        # Конвертация хромосом
        sv_df['chrom1'] = sv_df['chrom1'].apply(self.chrom_to_int)
        sv_df['chrom2'] = sv_df['chrom2'].apply(self.chrom_to_int)
        
        # Создание 22 фич
        sv_df = self._create_22_features(sv_df)
        
        return sv_df
    
    def _create_22_features(self, sv_df: pd.DataFrame) -> pd.DataFrame:
        """Создает 22 фичи для каждого SV события"""
        logger.info("Создание 22 фич...")
        
        # 1-6: Базовые координаты
        sv_df['sv_length1'] = sv_df['end1'] - sv_df['start1']
        sv_df['sv_length2'] = sv_df['end2'] - sv_df['start2']
        
        # 7: Inter-chromosomal flag
        sv_df['inter_chromosomal'] = (sv_df['chrom1'] != sv_df['chrom2']).astype(int)
        
        # 8: Total SV length
        sv_df['total_sv_length'] = sv_df['sv_length1'] + sv_df['sv_length2']
        
        # 9: SV size ratio
        sv_df['sv_size_ratio'] = sv_df['sv_length1'] / (sv_df['sv_length2'] + 1e-8)
        
        # 10: Position variance
        sv_df['position_variance'] = sv_df[['start1', 'end1', 'start2', 'end2']].var(axis=1)
        
        # 11: Strand compatibility
        sv_df['strand_compatibility'] = (sv_df['strand1'] * sv_df['strand2']).abs()
        
        # 12: SV complexity
        sv_df['sv_complexity'] = sv_df['total_sv_length'] * sv_df['position_variance']
        
        # 13: Coordinate span
        sv_df['coordinate_span'] = sv_df[['start1', 'end1', 'start2', 'end2']].max(axis=1) - sv_df[['start1', 'end1', 'start2', 'end2']].min(axis=1)
        
        # 14: Strand pattern
        sv_df['strand_pattern'] = (sv_df['strand1'] + sv_df['strand2']).abs()
        
        # 15: Position entropy
        sv_df['position_entropy'] = -((sv_df['start1'] + 1e-8) * np.log(sv_df['start1'] + 1e-8)).fillna(0)
        
        # Нормализация числовых фич
        numeric_cols = [
            'pe_support', 'sv_length1', 'sv_length2', 'total_sv_length', 
            'sv_size_ratio', 'position_variance', 'sv_complexity', 
            'coordinate_span', 'position_entropy'
        ]
        
        # Заменяем бесконечности и NaN
        for col in numeric_cols:
            sv_df[col] = sv_df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Нормализация
        sv_df[numeric_cols] = self.scaler.fit_transform(sv_df[numeric_cols])
        
        logger.info(f"Создано {len(sv_df)} SV событий с 22 фичами")
        return sv_df
    
    def sv_to_feature_vector(self, row: pd.Series, region_start: int, region_end: int, 
                           total_sv_in_region: int) -> List[float]:
        """Создает вектор из 22 фич для одного SV события"""
        region_length = region_end - region_start
        
        # Относительные координаты
        rel_start1 = (row['start1'] - region_start) / region_length if region_length > 0 else 0
        rel_end1 = (row['end1'] - region_start) / region_length if region_length > 0 else 0
        rel_start2 = (row['start2'] - region_start) / region_length if region_length > 0 else 0
        rel_end2 = (row['end2'] - region_start) / region_length if region_length > 0 else 0
        
        # Плотность и сложность
        sv_density = total_sv_in_region / (region_length / 1e6) if region_length > 0 else 0
        local_complexity = row['sv_complexity'] * sv_density
        
        # Вектор из 22 фич
        feature_vector = [
            # 1-6: Базовые координаты и класс
            row['chrom1'], rel_start1, rel_end1, 
            row['chrom2'], rel_start2, rel_end2,
            
            # 7-10: Поддержка и страны
            row['pe_support'], row['strand1'], row['strand2'], row['svclass'],
            
            # 11-15: Длины и флаги
            row['sv_length1'], row['sv_length2'], row['inter_chromosomal'],
            row['sv_size_ratio'], row['position_variance'],
            
            # 16-17: Совместимость и плотность
            row['strand_compatibility'], sv_density,
            
            # 18-22: Сложность и паттерны
            local_complexity, row['coordinate_span'],
            row['strand_pattern'], row['position_entropy'], row['sv_complexity']
        ]
        
        return feature_vector
    
    def calculate_sv_clustering(self, group: pd.DataFrame) -> float:
        """Рассчитывает меру кластеризации SV событий"""
        if len(group) < 2:
            return 0
        
        positions = group['start1'].values
        distances = np.diff(np.sort(positions))
        
        if len(distances) == 0:
            return 0
        
        return np.std(distances) / (np.mean(distances) + 1e-8)
    
    def create_enhanced_sequences(self, sv_df: pd.DataFrame, metadata_df: pd.DataFrame, 
                                max_sv: int = 200) -> Tuple[np.ndarray, np.ndarray, List[int], pd.DataFrame]:
        """Создает последовательности с 22 фичами для каждого региона"""
        logger.info(f"Создание enhanced sequences с {max_sv} SV событиями и {self.feature_dim} фичами")
        
        # Группируем по регионам
        grouped = sv_df.groupby(['donor_unique_id', 'Chr', 'Start_region', 'End_region', 'chromo_binary'])
        
        X_seq, y_seq = [], []
        sequence_lengths = []
        region_metadata = []
        
        for (donor, chr, start_region, end_region, binary), group in grouped:
            # Сортировка SV по позиции
            group = group.sort_values('start1').reset_index(drop=True)
            
            # Расчет характеристик региона
            region_length = end_region - start_region
            total_sv_in_region = len(group)
            
            if len(group) > 0:
                first_sv_pos = group['start1'].min()
                last_sv_pos = max(group['end1'].max(), group['start2'].max(), group['end2'].max())
                effective_length = last_sv_pos - first_sv_pos
            else:
                effective_length = region_length
                
            sv_density = total_sv_in_region / (effective_length / 1e6) if effective_length > 0 else 0
            sv_clustering = self.calculate_sv_clustering(group)
            
            # Метаданные региона
            region_metadata.append({
                'donor': donor,
                'chromosome': chr,
                'start': start_region,
                'end': end_region,
                'length_bp': region_length,
                'total_sv': total_sv_in_region,
                'sv_density': sv_density,
                'sv_clustering': sv_clustering,
                'chromo_binary': binary
            })
            
            # Создание фич для каждого SV события
            features = np.array([
                self.sv_to_feature_vector(row, start_region, end_region, total_sv_in_region)
                for _, row in group.iterrows()
            ])
            
            # Обрезка и паддинг
            features = features[:max_sv]
            actual_length = len(features)
            sequence_lengths.append(actual_length)
            
            # Паддинг
            if len(features) < max_sv:
                padded = np.zeros((max_sv - len(features), self.feature_dim))
                features = np.vstack([features, padded])
            
            X_seq.append(features)
            y_seq.append(binary)
        
        region_df = pd.DataFrame(region_metadata)
        
        logger.info(f"Создано {len(X_seq)} последовательностей")
        logger.info(f"Размерность данных: {np.array(X_seq).shape}")
        logger.info(f"Распределение классов: {np.sum(y_seq)} положительных из {len(y_seq)}")
        
        return np.array(X_seq), np.array(y_seq), sequence_lengths, region_df