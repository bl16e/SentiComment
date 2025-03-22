import torch
import yaml
import numpy as np
from data.preprocessing import TextPreprocessor
from models.lstm import SentimentLSTM
import logging
from pathlib import Path

class SentimentPredictor:
    def __init__(self, model_path: str, config_path: str):
        # 设置日志
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"使用设备: {self.device}")
        
        # 初始化预处理器
        self.preprocessor = TextPreprocessor(self.config)
        
        # 获取词向量维度并初始化模型
        input_dim = self.preprocessor.word2vec.vector_size
        self.logger.info(f"使用词向量维度: {input_dim}")
        
        # 加载模型
        self.model = SentimentLSTM(self.config, input_dim)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.logger.info(f"模型已加载: {model_path}")
    
    def text_to_vector(self, text: str) -> np.ndarray:
        """将文本转换为向量序列"""
        words = text.split()
        vectors = np.zeros((self.config['preprocessing']['max_sequence_length'], 
                          self.preprocessor.word2vec.vector_size), dtype=np.float32)
        
        for i, word in enumerate(words[:self.config['preprocessing']['max_sequence_length']]):
            if word in self.preprocessor.word2vec:
                vectors[i] = self.preprocessor.word2vec[word]
        
        return vectors
    
    def predict(self, text: str) -> dict:
        """预测文本情感"""
        # 预处理文本
        cleaned_text = self.preprocessor.clean_text(text)
        vector = self.text_to_vector(cleaned_text)
        vector = torch.FloatTensor(vector).unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            output = self.model(vector)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(output, dim=1).item()
        
        # 映射标签
        label_map = {0: '差评', 1: '中评', 2: '好评'}
        result = {
            'sentiment': label_map[prediction],
            'probabilities': {
                label_map[i]: prob.item()
                for i, prob in enumerate(probabilities[0])
            }
        }
        
        return result

def main():
    # 示例使用
    predictor = SentimentPredictor(
        model_path='models/best_model_final.pth',
        config_path='config/config.yaml'
    )
    
    # 测试文本
    test_texts = [
        "这个产品质量很好，我很满意",
        "一般般吧，没什么特别的",
        "太差了，完全不值这个价"
    ]
    
    for text in test_texts:
        result = predictor.predict(text)
        print(f"\n文本: {text}")
        print(f"情感: {result['sentiment']}")
        print("概率分布:")
        for sentiment, prob in result['probabilities'].items():
            print(f"  {sentiment}: {prob:.4f}")

if __name__ == '__main__':
    main() 