import logging
import os
import yaml
from pathlib import Path
import pandas as pd
from gensim.models import Word2Vec
import jieba
import re
from tqdm import tqdm

def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('word2vec_training.log'),
            logging.StreamHandler()
        ]
    )

def load_config():
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                              'config', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_stopwords(config):
    """加载停用词"""
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    stopwords_path = os.path.join(root_dir, config['data']['stopwords_file'])
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        return set([line.strip() for line in f])

def clean_text(text: str, stopwords: set) -> list:
    """清洗文本并分词"""
    # 移除特殊字符
    # text = re.sub("[\s+.!/_,$%^(+\"']+|[+——！，。？、~@#￥%……&（）]+", "", str(text))
    # 分词
    words = jieba.cut(re.sub("[\s+.!/_,$%^(+\"']+|[+——！，。？、~@#￥%……&（）]+", "", str(text)))
    # 移除停用词
    return [w for w in words if w not in stopwords]

def train_word2vec(config):
    """训练word2vec模型"""
    logger = logging.getLogger(__name__)
    
    # 加载停用词
    stopwords = load_stopwords(config)
    logger.info("停用词加载完成")
    
    # 获取项目根目录
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # 加载训练数据
    train_path = os.path.join(root_dir, config['data']['train_file'])
    df = pd.read_csv(train_path, sep='\t', header=None, names=['label', 'review'])
    logger.info(f"加载了 {len(df)} 条训练数据")
    
    # 数据预处理
    logger.info("开始文本预处理...")
    sentences = []
    for text in tqdm(df['review'], desc="处理文本"):
        words = clean_text(text, stopwords)
        if words:  # 只添加非空句子
            sentences.append(words)
    
    # 训练Word2Vec模型
    logger.info("开始训练Word2Vec模型...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=config['model']['embedding_dim'],  # 词向量维度
        window=5,  # 上下文窗口大小
        min_count=2,  # 最小词频
        workers=4,  # 训练的线程数
        sg=1,  # 使用Skip-gram模型
        epochs=10  # 训练轮数
    )
    
    # 创建保存目录
    save_dir = os.path.join(root_dir, os.path.dirname(config['data']['word2vec_model']))
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 保存模型
    save_path = os.path.join(root_dir, config['data']['word2vec_model'])
    model.wv.save_word2vec_format(save_path, binary=False)
    logger.info(f"模型已保存到: {save_path}")
    
    # 输出模型信息
    logger.info(f"词表大小: {len(model.wv)}")
    logger.info(f"词向量维度: {model.vector_size}")
    
    # 测试一些常见词的相似词
    test_words = ['好', '差', '贵', '便宜']
    for word in test_words:
        if word in model.wv:
            logger.info(f"\n'{word}'的最相似词:")
            similar_words = model.wv.most_similar(word)
            for similar_word, score in similar_words[:5]:
                logger.info(f"  {similar_word}: {score:.4f}")

if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        config = load_config()
        train_word2vec(config)
        logger.info("Word2Vec模型训练完成！")
    except Exception as e:
        logger.error(f"训练过程中出错: {str(e)}", exc_info=True) 