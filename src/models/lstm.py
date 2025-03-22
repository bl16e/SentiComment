import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
import yaml
from typing import Tuple, Dict, Optional
from pathlib import Path

class SentimentLSTM(nn.Module):
    def __init__(self, config, input_dim):
        super(SentimentLSTM, self).__init__()
        self.config = config
        
        self.lstm = nn.LSTM(
            input_size=input_dim,  # 使用word2vec的原始向量维度
            hidden_size=config['model']['lstm_hidden_size'],
            num_layers=config['model']['lstm_layers'],
            batch_first=True,
            bidirectional=True,
            dropout=config['model']['dropout'] if config['model']['lstm_layers'] > 1 else 0,
            proj_size=config['model']['proj_size']
        )
        
        # 由于使用了双向LSTM和投影层，输入维度为 2 * proj_size
        lstm_output_dim = 2 * config['model']['proj_size']
        
        self.dropout = nn.Dropout(config['model']['dropout'])
        self.fc = nn.Linear(lstm_output_dim, 3)  # 3分类
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, seq_len, 2 * proj_size)
        
        # 使用最后一个时间步的输出
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

class SentimentAnalyzer:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化模型
        input_dim = self.preprocessor.word2vec.vector_size  # 使用word2vec的实际向量维度
        self.model = SentimentLSTM(self.config, input_dim)
        self.model.to(self.device)
        
        # 定义损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['model']['learning_rate']
        )
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """评估模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                total_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return total_loss / len(val_loader), correct / total
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
             num_epochs: int) -> Dict[str, list]:
        """训练模型"""
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            self.logger.info(f'Epoch {epoch+1}/{num_epochs}:')
            self.logger.info(f'Train Loss: {train_loss:.4f}')
            self.logger.info(f'Val Loss: {val_loss:.4f}')
            self.logger.info(f'Val Accuracy: {val_acc:.4f}')
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('best_model.pth')
        
        return history
    
    def save_model(self, filename: str):
        """保存模型"""
        save_path = Path(self.config['data']['models']) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        self.logger.info(f'模型已保存到: {save_path}')
    
    def load_model(self, filename: str):
        """加载模型"""
        model_path = Path(self.config['data']['models']) / filename
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.logger.info(f'模型已加载: {model_path}') 