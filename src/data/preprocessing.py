import re
import jieba
import numpy as np
from typing import List, Tuple, Optional, Dict
import pandas as pd
from gensim.models import KeyedVectors
import pickle
import os
from pathlib import Path
import logging
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import hashlib
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.sampler import WeightedRandomSampler
from imblearn.over_sampling import SMOTE
from collections import Counter

class TextDataset(Dataset):
    def __init__(self, vectors: np.ndarray, labels: np.ndarray):
        self.vectors = vectors
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.vectors[idx]), torch.LongTensor([self.labels[idx]])[0]

def text_to_vector(text: str, word2vec: KeyedVectors, max_length: int) -> np.ndarray:
    """将文本转换为向量序列"""
    words = text.split()
    vector = np.zeros((max_length, word2vec.vector_size), dtype=np.float32)
    
    for i, word in enumerate(words[:max_length]):
        if word in word2vec:
            vector[i] = word2vec[word]
    
    return vector

def collate_fn(batch):
    # 将batch中的数据堆叠起来
    vectors = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return vectors, labels

class TextPreprocessor:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 获取项目根目录
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        
        # 创建必要的目录
        self._create_directories()
        
        self.stopwords = self._load_stopwords()
        self._load_word2vec()
        
        # 添加交叉验证设置
        self.n_splits = self.config['preprocessing'].get('n_splits', 5)  # 默认5折交叉验证
        self.current_fold = 0
        
        # SMOTE设置
        self.use_smote = self.config['preprocessing'].get('use_smote', True)
        self.random_state = self.config['preprocessing'].get('random_seed', 42)
    
    def _create_directories(self):
        """创建必要的目录"""
        for dir_path in [
            os.path.join(self.root_dir, self.config['data']['processed_dir']),
            os.path.join(self.root_dir, self.config['data']['cache_dir'])
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _get_abs_path(self, rel_path):
        """获取相对于项目根目录的绝对路径"""
        return os.path.join(self.root_dir, rel_path)
    
    def _load_stopwords(self) -> set:
        """加载停用词"""
        try:
            stopwords_path = self._get_abs_path(self.config['data']['stopwords_file'])
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                stopwords = set([line.strip() for line in f])
            self.logger.info(f"成功加载 {len(stopwords)} 个停用词")
            return stopwords
        except Exception as e:
            self.logger.error(f"加载停用词失败: {str(e)}")
            raise
    
    def _load_word2vec(self):
        """加载词向量模型并降维"""
        try:
            model_path = self._get_abs_path(self.config['data']['word2vec_model'])
            # 加载原始词向量
            self.word2vec = KeyedVectors.load_word2vec_format(
                model_path, 
                binary=False,
                unicode_errors='ignore'
            )
            self.logger.info(f"成功加载词向量模型")
        except Exception as e:
            self.logger.error(f"加载词向量模型失败: {str(e)}")
            raise
    
    def _get_cache_path(self, data_file: str) -> str:
        """获取缓存文件路径"""
        # 使用数据文件路径和配置参数的哈希值作为缓存文件名
        cache_key = f"{data_file}_{self.config['preprocessing']['max_sequence_length']}_{self.config['preprocessing']['test_size']}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        cache_dir = self._get_abs_path(self.config['data']['cache_dir'])
        return os.path.join(cache_dir, f"processed_data_{cache_hash}.pkl")
    
    def clean_text(self, text: str) -> str:
        """清洗文本"""
        # 移除特殊字符、分词
        words = jieba.cut(re.sub("[\s+.!/_,$%^(+\"']+|[+——！，。？、~@#￥%……&（）]+", "", str(text)))
        # 移除停用词
        words = [w for w in words if w not in self.stopwords]
        return ' '.join(words)
    
    def _get_class_weights(self, labels: np.ndarray) -> torch.Tensor:
        """计算类别权重"""
        class_counts = np.bincount(labels)
        total = len(labels)
        class_weights = torch.FloatTensor(total / (len(class_counts) * class_counts))
        return class_weights

    def _get_sample_weights(self, labels: np.ndarray) -> np.ndarray:
        """计算样本权重用于平衡采样"""
        class_weights = self._get_class_weights(labels)
        sample_weights = np.array([class_weights[label] for label in labels])
        return sample_weights

    def _process_data(self, data_file: str) -> Dict[str, List]:
        """处理数据"""
        # 加载和预处理数据
        data_path = self._get_abs_path(data_file)
        df = pd.read_csv(data_path, sep='\t', header=None, names=['label', 'review'])
        
        # 清洗文本
        texts = [self.clean_text(text) for text in tqdm(df['review'], desc="清洗文本")]
        labels = df['label'].values
        
        return {
            'texts': texts,
            'labels': labels
        }

    def _apply_smote(self, vectors: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """应用SMOTE过采样"""
        # 重塑向量以适应SMOTE
        n_samples, seq_len, n_features = vectors.shape
        X_reshaped = vectors.reshape(n_samples, seq_len * n_features)
        
        # 创建SMOTE对象
        smote = SMOTE(random_state=self.random_state)
        
        # 应用SMOTE
        X_resampled, y_resampled = smote.fit_resample(X_reshaped, labels)
        
        # 将向量重塑回原始形状
        vectors_resampled = X_resampled.reshape(-1, seq_len, n_features)
        
        # 记录重采样前后的类别分布
        before_counts = Counter(labels)
        after_counts = Counter(y_resampled)
        self.logger.info(f"SMOTE重采样前的类别分布: {before_counts}")
        self.logger.info(f"SMOTE重采样后的类别分布: {after_counts}")
        
        return vectors_resampled, y_resampled

    def create_dataloaders(self, data_file: str, batch_size: Optional[int] = None) -> List[Tuple[DataLoader, DataLoader]]:
        """创建K折交叉验证的数据加载器"""
        if batch_size is None:
            batch_size = self.config['model']['batch_size']
        
        cache_path = self._get_cache_path(data_file)
        
        # 尝试从缓存加载数据
        if os.path.exists(cache_path):
            self.logger.info(f"从缓存加载预处理数据: {cache_path}")
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
        else:
            self.logger.info("处理数据并创建缓存...")
            data = self._process_data(data_file)
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            self.logger.info(f"数据处理完成并已缓存到: {cache_path}")
        
        # 创建K折交叉验证的数据集
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, 
                            random_state=self.config['preprocessing']['random_seed'])
        
        fold_dataloaders = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(data['texts'], data['labels'])):
            # 分割训练集和验证集
            train_texts = [data['texts'][i] for i in train_idx]
            train_labels = data['labels'][train_idx]
            val_texts = [data['texts'][i] for i in val_idx]
            val_labels = data['labels'][val_idx]
            
            # 将文本转换为向量
            train_vectors = np.array([
                text_to_vector(text, self.word2vec, self.config['preprocessing']['max_sequence_length'])
                for text in tqdm(train_texts, desc=f"正在处理第 {fold+1} 折训练集")
            ])
            
            val_vectors = np.array([
                text_to_vector(text, self.word2vec, self.config['preprocessing']['max_sequence_length'])
                for text in tqdm(val_texts, desc=f"正在处理第 {fold+1} 折验证集")
            ])
            
            # 对训练集应用SMOTE过采样
            if self.use_smote:
                self.logger.info(f"对第 {fold+1} 折训练集应用SMOTE过采样...")
                train_vectors, train_labels = self._apply_smote(train_vectors, train_labels)
            
            # 创建数据集
            train_dataset = TextDataset(train_vectors, train_labels)
            val_dataset = TextDataset(val_vectors, val_labels)
            
            # 创建数据加载器
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,  # 使用SMOTE后，数据已经平衡，可以直接使用随机打乱
                collate_fn=collate_fn,
                num_workers=1,
                pin_memory=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=1,
                pin_memory=True
            )
            
            fold_dataloaders.append((train_loader, val_loader))
            self.logger.info(f"第 {fold+1} 折 - 训练集大小: {len(train_dataset)}，验证集大小: {len(val_dataset)}")
            
            # 输出类别分布信息
            train_class_dist = np.bincount(train_labels)
            val_class_dist = np.bincount(val_labels)
            self.logger.info(f"训练集类别分布: {train_class_dist}")
            self.logger.info(f"验证集类别分布: {val_class_dist}")
        
        return fold_dataloaders 