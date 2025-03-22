# SentiComment

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-black)](https://github.com/psf/black)
[![Issues](https://img.shields.io/github/issues/bl16e/SentiHotel)](https://github.com/bl16e/SentiHotel/issues)
[![Last Commit](https://img.shields.io/github/last-commit/bl16e/SentiHotel)](https://github.com/bl16e/SentiHotel/commits/main)

基于深度学习的京东商品评论情感分析系统

[English](README_EN.md) | 简体中文

## 项目简介

SentiComment 是一个专门针对京东商品评论的情感分析系统，使用深度学习技术对评论文本进行情感分类。系统采用了预训练词向量和LSTM深度学习模型，结合了K折交叉验证和类别平衡等技术，实现了高精度的情感分类。

### 主要特点

- 训练词向量进行文本表示
- 采用双向LSTM进行特征提取
- 实现K折交叉验证提高模型可靠性
- 使用类别权重平衡处理数据不平衡问题
- 支持模型训练过程可视化
- 提供详细的模型评估指标

## 项目结构

```
项目结构/
├── config/              # 配置文件目录
├── data/
│   ├── raw/            # 原始数据
│   ├── processed/      # 处理后的数据
│   ├── cache/          # 缓存文件
│   └── models/         # 模型相关数据
├── examples/           # 示例代码
├── figures/            # 可视化图表
├── models/             # 模型文件
├── src/
│   ├── data/          # 数据处理相关代码
│   ├── models/        # 模型定义
│   ├── predict.py     # 预测脚本
│   └── train_acc.py   # 训练脚本
├── requirements.txt    # 项目依赖
├── training.log       # 训练日志
├── word2vec_training.log  # 词向量训练日志
└── README.md          # 项目说明
```

## 环境要求

- Python 3.8+
- PyTorch 1.8+
- scikit-learn
- numpy
- pandas
- gensim
- jieba
- matplotlib
- pyyaml
- tqdm

## 快速开始

1. 克隆项目并安装依赖：
```bash
git clone https://github.com/bl16e/SentiComment.git
cd SentiComment
pip install -r requirements.txt
```

2. 准备数据：
- 将训练数据放在 `data/raw/train.tsv`
- 将停用词表放在 `data/raw/stopwords.txt`

1. 修改配置：
- 根据需要修改 `config/config.yaml` 中的参数

1. 训练模型：
```bash
python src/train.py
```

## 模型训练

模型训练过程包括：
1. 数据预处理：分词、去停用词、向量化
2. K折交叉验证：默认5折
3. 类别平衡：使用加权采样
4. 模型训练：使用带预热的学习率调度
5. 模型评估：准确率、精确率、召回率、F1分数

## 可视化

训练过程会生成以下可视化图表：
- 损失曲线
- 准确率曲线
- 学习率变化曲线

## 评估指标

模型评估包括：
- 每折的详细评估报告
- 交叉验证的平均性能
- 混淆矩阵和分类报告
- 标准差分析

## 注意事项

1. 数据格式：
   - 训练和测试数据必须是TSV格式
   - 格式：标签\t评论文本
   - 标签：0（差评）、1（中评）、2（好评）

2. 内存使用：
   - 预训练词向量可能占用较大内存
   - 可通过调整batch_size控制内存使用

3. GPU支持：
   - 自动检测并使用可用的GPU
   - 可通过配置文件调整相关参数

## 维护者

- [@bl16e](https://github.com/bl16e)
