import pandas as pd
import numpy as np
from pathlib import Path
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
from typing import Dict, List, Optional, Any, Tuple
import hydra
from omegaconf import DictConfig
import pickle
import json
from tqdm import tqdm
import os

logger = logging.getLogger(__name__)

class SVDataPreprocessor:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.metadata_df = None
        self.label_encoder = LabelEncoder()
        self.scalers = {}
        
        # ИНИЦИАЛИЗАЦИЯ FEATURE_EXTRACTOR - ДОБАВЛЕНО
        from src.data.feature_extractor import SVFeatureExtractor
        self.feature_extractor = SVFeatureExtractor(cfg)
        
        current_file = Path(__file__).resolve()
        if 'src' in current_file.parts and 'data' in current_file.parts:
            self.project_root = current_file.parent.parent.parent
        else:
            self.project_root = current_file
            while self.project_root.name != 'ChromothripsisPredictor' and self.project_root.parent != self.project_root:
                self.project_root = self.project_root.parent
        
        self.processed_data_path = self.project_root / "data" / "processed_data"
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Processed data path: {self.processed_data_path}")
        
        self._check_required_files()
        
    def _check_required_files(self):
        """Проверяет существование необходимых файлов"""
        logger.info("Проверка необходимых файлов...")
        
        files_to_check = {
            'chromothripsis': self.cfg.data.data_paths.chromothripsis_file,
            'mapping': self.cfg.data.data_paths.mapping_file,
            'icgc_dir': self.cfg.data.data_paths.sv_directories.icgc,
            'tcga_dir': self.cfg.data.data_paths.sv_directories.tcga
        }
        
        for name, relative_path in files_to_check.items():
            absolute_path = self._get_absolute_path(relative_path)
            exists = absolute_path.exists()
            
            if name.endswith('_dir'):
                if exists:
                    files = list(absolute_path.glob("*.bedpe.gz"))
                    logger.info(f"{'✅' if files else '⚠️'} {name}: {absolute_path} (файлов: {len(files)})")
                else:
                    logger.error(f"❌ {name}: {absolute_path} - директория не найдена")
            else:
                if exists:
                    logger.info(f"✅ {name}: {absolute_path}")
                else:
                    logger.error(f"❌ {name}: {absolute_path} - файл не найден")
        
    def _get_absolute_path(self, relative_path: str) -> Path:
        """Преобразует относительный путь в абсолютный относительно корня проекта"""
        absolute_path = self.project_root / relative_path
        return absolute_path
        
    def load_metadata(self) -> pd.DataFrame:
        """Загружает и объединяет метаданные из конфигурации"""
        try:
            chromothripsis_path = self._get_absolute_path(self.cfg.data.data_paths.chromothripsis_file)
            mapping_path = self._get_absolute_path(self.cfg.data.data_paths.mapping_file)
            
            logger.info(f"Загрузка chromothripsis данных из: {chromothripsis_path}")
            logger.info(f"Загрузка mapping данных из: {mapping_path}")
            
            if not chromothripsis_path.exists():
                alt_path = self.project_root / "data" / "raw" / "chromothripsis_data.xlsx"
                if alt_path.exists():
                    chromothripsis_path = alt_path
                    logger.info(f"Файл найден в альтернативном месте: {chromothripsis_path}")
                else:
                    raise FileNotFoundError(f"Файл chromothripsis_data.xlsx не найден. Ожидался: {chromothripsis_path}")
            
            if not mapping_path.exists():
                alt_path = self.project_root / "data" / "raw" / "pcawg_sample_sheet.tsv"
                if alt_path.exists():
                    mapping_path = alt_path
                    logger.info(f"Файл найден в альтернативном месте: {mapping_path}")
                else:
                    raise FileNotFoundError(f"Файл pcawg_sample_sheet.tsv не найден. Ожидался: {mapping_path}")
            
            logger.info("Чтение Excel файла...")
            chromothripsis_df = pd.read_excel(chromothripsis_path)
            logger.info(f"Загружено {len(chromothripsis_df)} строк из chromothripsis")
            
            logger.info("Чтение TSV файла...")
            mapping_df = pd.read_csv(mapping_path, sep='\t')
            logger.info(f"Загружено {len(mapping_df)} строк из mapping")
            
            logger.info("Объединение данных...")
            self.metadata_df = chromothripsis_df.merge(
                mapping_df[['donor_unique_id', 'aliquot_id']], 
                on='donor_unique_id', 
                how='inner'
            )
            
            logger.info(f"Загружено {len(self.metadata_df)} регионов с разметкой")
            logger.info(f"Колонки в metadata: {list(self.metadata_df.columns)}")
            
            if 'chromo_label' in self.metadata_df.columns:
                label_counts = self.metadata_df['chromo_label'].value_counts()
                logger.info(f"Распределение меток: {dict(label_counts)}")
            
            return self.metadata_df
            
        except Exception as e:
            logger.error(f"Ошибка загрузки метаданных: {e}")
            raise
    
    def save_processed_data(self, data: Dict, filename: str):
        """Сохраняет препроцессированные данные"""
        filepath = self.processed_data_path / filename
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Данные сохранены в {filepath}")
    
    def load_processed_data(self, filename: str) -> Optional[Dict]:
        """Загружает препроцессированные данные"""
        filepath = self.processed_data_path / filename
        if filepath.exists():
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"Данные загружены из {filepath}")
                return data
            except Exception as e:
                logger.warning(f"Ошибка загрузки данных из {filepath}: {e}")
                return None
        return None
    
    def save_scalers(self):
        """Сохраняет scalers для последующего использования"""
        scalers_path = self.processed_data_path / "scalers.pkl"
        with open(scalers_path, 'wb') as f:
            pickle.dump(self.scalers, f)
        logger.info(f"Scalers сохранены в {scalers_path}")
    
    def load_scalers(self):
        """Загружает scalers"""
        scalers_path = self.processed_data_path / "scalers.pkl"
        if scalers_path.exists():
            try:
                with open(scalers_path, 'rb') as f:
                    self.scalers = pickle.load(f)
                logger.info(f"Scalers загружены из {scalers_path}")
            except Exception as e:
                logger.warning(f"Ошибка загрузки scalers: {e}")
                self.scalers = {}
    
    def normalize_features(self, features: np.ndarray, feature_type: str) -> np.ndarray:
        """Нормализует features с помощью StandardScaler"""
        if feature_type not in self.scalers:
            self.scalers[feature_type] = StandardScaler()
            
        if len(features.shape) == 1:
            features = features.reshape(-1, 1)
            
        if hasattr(self, '_is_training') and self._is_training:
            normalized = self.scalers[feature_type].fit_transform(features)
        else:
            if feature_type in self.scalers:
                normalized = self.scalers[feature_type].transform(features)
            else:
                normalized = features
                logger.warning(f"Scaler для {feature_type} не найден, используем исходные данные")
        
        return normalized

    def create_sequence_features(self, sv_events: pd.DataFrame, region_start: int, 
                               region_end: int) -> np.ndarray:
        """Создает последовательностные фичи для RNN с нормализацией"""
        sequence_length = self.cfg.data.preprocessing.sequence_length
        bin_size = max(1, (region_end - region_start) // sequence_length)
        sequence = np.zeros((sequence_length, 8))
        
        for i in range(sequence_length):
            bin_start = region_start + i * bin_size
            bin_end = bin_start + bin_size
            
            bin_events = sv_events[
                ((sv_events['chrom1'] == sv_events['chrom2']) & 
                 (sv_events['start1'].between(bin_start, bin_end))) |
                ((sv_events['chrom1'] != sv_events['chrom2']) & 
                 ((sv_events['start1'].between(bin_start, bin_end)) | 
                  (sv_events['start2'].between(bin_start, bin_end))))
            ]
            
            if len(bin_events) > 0:
                sequence[i, 0] = len(bin_events)
                sequence[i, 1] = bin_events['pe_support'].mean() if not bin_events['pe_support'].isna().all() else 0
                sequence[i, 2] = (bin_events['svclass'] == 'DEL').sum()
                sequence[i, 3] = (bin_events['svclass'] == 'DUP').sum()
                sequence[i, 4] = (bin_events['svclass'] == 'INV').sum()
                sequence[i, 5] = (bin_events['svclass'] == 'TRA').sum()
                sequence[i, 6] = bin_events['svclass'].nunique()
                sequence[i, 7] = 1 if len(bin_events[bin_events['chrom1'] != bin_events['chrom2']]) > 0 else 0
        
        if hasattr(self.cfg.data.preprocessing, 'normalize') and self.cfg.data.preprocessing.normalize:
            for j in range(sequence.shape[1]):
                sequence[:, j] = self.normalize_features(sequence[:, j], f'sequence_{j}').flatten()
        
        return sequence

    def process_patient_regions(self, sv_directory: str, output_format: str = 'sequence', 
                          is_training: bool = True) -> List[Dict]:
        """Обрабатывает все регионы пациентов для указанного формата"""
        self._is_training = is_training
        
        if self.metadata_df is None:
            self.load_metadata()
        
        absolute_sv_dir = self._get_absolute_path(sv_directory)
        logger.info(f"Обработка SV данных с 22 фичами из: {absolute_sv_dir}, формат: {output_format}")
        
        if not absolute_sv_dir.exists():
            logger.error(f"SV директория не найдена: {absolute_sv_dir}")
            return []
        
        all_data = []
        sv_path = absolute_sv_dir
        
        file_paths = list(sv_path.glob("*.bedpe.gz"))
        if not file_paths:
            logger.warning(f"Не найдено .bedpe.gz файлов в директории: {sv_path}")
            return all_data
        
        logger.info(f"Найдено {len(file_paths)} файлов для обработки")
        
        for file_path in tqdm(file_paths, desc=f"Processing {output_format} with 22 features"):
            file_patient_id = file_path.stem.split('.')[0]
            
            patient_regions = self.metadata_df[self.metadata_df['aliquot_id'] == file_patient_id]
            
            if len(patient_regions) == 0:
                logger.debug(f"Не найдено регионов для пациента: {file_patient_id}")
                continue
            
            try:
                # Загрузка и предобработка SV данных
                sv_df = pd.read_csv(file_path, sep='\t', compression='gzip')
                logger.debug(f"Загружен SV файл: {file_path} с {len(sv_df)} событиями")
                
                # Предобработка с созданием 22 фич
                sv_df_processed = self.feature_extractor.preprocess_sv_data(sv_df)
                
                # Добавляем идентификаторы для группировки
                sv_df_processed['donor_unique_id'] = file_patient_id
                
                for _, region in patient_regions.iterrows():
                    chromosome = region['Chr']
                    region_start = region['Start']
                    region_end = region['End']
                    chromo_label = region['chromo_label']
                    
                    # Фильтруем SV события для региона
                    region_events = sv_df_processed[
                        ((sv_df_processed['chrom1'] == chromosome) & 
                        (sv_df_processed['start1'].between(region_start, region_end))) |
                        ((sv_df_processed['chrom2'] == chromosome) & 
                        (sv_df_processed['start2'].between(region_start, region_end)))
                    ]
                    
                    if len(region_events) == 0:
                        continue
                    
                    # Добавляем метки для группировки
                    region_events = region_events.copy()
                    region_events['Chr'] = chromosome
                    region_events['Start_region'] = region_start
                    region_events['End_region'] = region_end
                    region_events['chromo_binary'] = 1 if chromo_label in self.cfg.data.preprocessing.positive_classes else 0
                    
                    # Для всех sequence-based форматов используем create_enhanced_sequences
                    if output_format in ['sequence', 'advanced_sequence', 'transformer', 'hybrid', 'residual_lstm', 'multiscale_cnn']:
                        X_seq, y_seq, seq_lengths, region_df = self.feature_extractor.create_enhanced_sequences(
                            region_events, self.metadata_df, 
                            max_sv=self.cfg.data.preprocessing.sequence_length
                        )
                        
                        # Для каждого региона добавляем данные
                        for i in range(len(X_seq)):
                            all_data.append({
                                'patient_id': file_patient_id,
                                'chromosome': chromosome,
                                'region_start': region_start,
                                'region_end': region_end,
                                'features': X_seq[i],
                                'label': y_seq[i],
                                'sv_count': seq_lengths[i],
                                'sv_density': region_df.iloc[i]['sv_density'] if i < len(region_df) else 0,
                                'format': output_format  # Добавляем информацию о формате
                            })
                    
                    # Для других форматов можно добавить соответствующую обработку
                    elif output_format == 'graph':
                        graph_features = self.create_graph_features(region_events, chromosome, region_start, region_end)
                        if len(graph_features) > 0:
                            all_data.append({
                                'patient_id': file_patient_id,
                                'chromosome': chromosome,
                                'region_start': region_start,
                                'region_end': region_end,
                                'features': graph_features,
                                'label': 1 if chromo_label in self.cfg.data.preprocessing.positive_classes else 0,
                                'format': output_format
                            })
                    
                    elif output_format == 'cnn':
                        cnn_features = self.create_cnn_features(region_events, region_start, region_end)
                        if len(cnn_features) > 0:
                            all_data.append({
                                'patient_id': file_patient_id,
                                'chromosome': chromosome,
                                'region_start': region_start,
                                'region_end': region_end,
                                'features': cnn_features,
                                'label': 1 if chromo_label in self.cfg.data.preprocessing.positive_classes else 0,
                                'format': output_format
                            })
                            
            except Exception as e:
                logger.warning(f"Ошибка обработки {file_patient_id}: {e}")
                continue
        
        logger.info(f"Обработано {len(all_data)} samples с 22 фичами в формате {output_format}")
        return all_data

    def create_graph_features(self, sv_events, chromosome, region_start, region_end):
        """Заглушка для graph features"""
        logger.warning("Graph features not implemented yet")
        return np.array([])
    
    def create_cnn_features(self, sv_events, region_start, region_end):
        """Заглушка для CNN features"""
        logger.warning("CNN features not implemented yet")
        return np.array([])
    
    def create_tabular_features(self, sv_events, chromosome, region_start, region_end):
        """Заглушка для tabular features"""
        logger.warning("Tabular features not implemented yet")
        return {}