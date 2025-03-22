# SentiComment

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-black)](https://github.com/psf/black)
[![Issues](https://img.shields.io/github/issues/bl16e/SentiHotel)](https://github.com/bl16e/SentiHotel/issues)
[![Last Commit](https://img.shields.io/github/last-commit/bl16e/SentiHotel)](https://github.com/bl16e/SentiHotel/commits/main)

Deep Learning Based JD Product Review Sentiment Analysis System

English | [简体中文](README.md)

## Introduction

SentiComment is a sentiment analysis system specifically designed for JD product reviews, utilizing deep learning technology for sentiment classification. The system employs pre-trained word vectors and LSTM deep learning models, combined with K-fold cross-validation and class balancing techniques to achieve high-precision sentiment classification.

### Key Features

- Text representation using pre-trained word vectors
- Feature extraction using Bidirectional LSTM
- K-fold cross-validation for improved model reliability
- Class weight balancing for imbalanced data
- Model training process visualization
- Detailed model evaluation metrics

## Project Structure

```
Project Structure/
├── config/              # Configuration directory
├── data/
│   ├── raw/            # Raw data
│   ├── processed/      # Processed data
│   ├── cache/          # Cache files
│   └── models/         # Model related data
├── examples/           # Example code
├── figures/            # Visualization plots
├── models/             # Model files
├── src/
│   ├── data/          # Data processing code
│   ├── models/        # Model definitions
│   ├── predict.py     # Prediction script
│   └── train_acc.py   # Training script
├── requirements.txt    # Project dependencies
├── training.log       # Training log
├── word2vec_training.log  # Word2Vec training log
└── README.md          # Project documentation
```

## Requirements

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

## Quick Start

1. Clone the repository and install dependencies:
```bash
git clone https://github.com/bl16e/SentiComment.git
cd SentiComment
pip install -r requirements.txt
```

2. Prepare the data:
- Place training data in `data/raw/train.tsv`
- Place stopwords file in `data/raw/stopwords.txt`

1. Modify configuration:
- Adjust parameters in `config/config.yaml` as needed

1. Train the model:
```bash
python src/train.py
```

## Model Training

The training process includes:
1. Data preprocessing: tokenization, stopword removal, vectorization
2. K-fold cross-validation: default 5 folds
3. Class balancing: using weighted sampling
4. Model training: with learning rate warmup scheduling
5. Model evaluation: accuracy, precision, recall, F1 score

## Visualization

The training process generates the following visualization plots:
- Loss curves
- Accuracy curves
- Learning rate curves

## Evaluation Metrics

Model evaluation includes:
- Detailed evaluation report for each fold
- Cross-validation average performance
- Confusion matrix and classification report
- Standard deviation analysis

## Notes

1. Data Format:
   - Training and test data must be in TSV format
   - Format: label\ttext
   - Labels: 0 (negative), 1 (neutral), 2 (positive)

2. Memory Usage:
   - Pre-trained word vectors may require significant memory
   - Memory usage can be controlled by adjusting batch_size

3. GPU Support:
   - Automatically detects and uses available GPU
   - Related parameters can be adjusted in the config file

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Maintainer

- [@bl16e](https://github.com/bl16e)

