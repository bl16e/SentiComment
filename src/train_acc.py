import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import logging
from pathlib import Path
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from accelerate import Accelerator
from torch.cuda.amp import autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data.preprocessing import TextPreprocessor
from models.lstm import SentimentLSTM

def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

def plot_training_history(history, fold=None):
    """绘制训练历史"""
    # 创建图表目录
    Path('figures').mkdir(exist_ok=True)
    
    # 添加折数到文件名
    fold_str = f'_fold_{fold}' if fold is not None else ''
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss{" - Fold " + str(fold) if fold else ""}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'figures/loss{fold_str}.png')
    plt.close()
    
    # 绘制准确率曲线
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title(f'Model Accuracy{" - Fold " + str(fold) if fold else ""}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'figures/accuracy{fold_str}.png')
    plt.close()
    
    # 绘制学习率曲线
    if 'learning_rates' in history:
        plt.figure(figsize=(10, 5))
        plt.plot(history['learning_rates'], label='Learning Rate')
        plt.title(f'Learning Rate Schedule{" - Fold " + str(fold) if fold else ""}')
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.legend()
        plt.savefig(f'figures/learning_rate{fold_str}.png')
        plt.close()

def load_config():
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def train_epoch(model, dataloader, criterion, optimizer, scheduler, accelerator):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    current_lr = []
    
    progress_bar = tqdm(dataloader, desc='Training', disable=not accelerator.is_local_main_process)
    
    for batch_vectors, batch_labels in progress_bar:
        # print("Input shape:", batch_vectors.shape)  # 打印输入维度
        with autocast():
            outputs = model(batch_vectors)
            loss = criterion(outputs, batch_labels)
        
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        
        if scheduler is not None:
            scheduler.step()
            current_lr.append(scheduler.get_last_lr()[0])
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += batch_labels.size(0)
        correct += predicted.eq(batch_labels).sum().item()
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(dataloader), correct / total, current_lr

def validate(model, dataloader, criterion, accelerator):
    """验证模型性能"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for vectors, labels in dataloader:
            with autocast():
                outputs = model(vectors)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            
            # 收集预测结果和真实标签
            if accelerator.is_local_main_process:
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
    
    # 在主进程上计算指标
    if accelerator.is_local_main_process:
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='macro', zero_division=1)
        recall = recall_score(all_targets, all_predictions, average='macro', zero_division=1)
        f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=1)
        
        return total_loss / len(dataloader), accuracy, {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_predictions,
            'targets': all_targets
        }
    return total_loss / len(dataloader), 0, {}

def main():
    """主训练流程"""
    # 初始化 Accelerator
    accelerator = Accelerator(
        mixed_precision='fp16',
        gradient_accumulation_steps=2
    )
    
    # 设置日志
    if accelerator.is_local_main_process:
        setup_logging()
    logger = logging.getLogger(__name__)
    
    # 加载配置
    config = load_config()
    logger.info(f"使用设备: {accelerator.device}")
    
    # 数据预处理和加载
    preprocessor = TextPreprocessor(config)
    fold_dataloaders = preprocessor.create_dataloaders(config['data']['train_file'])
    logger.info("数据加载完成")
    
    # 记录所有折的结果
    all_fold_results = []
    best_fold_acc = 0
    best_fold_model = None
    best_fold_metrics = None
    
    # K折交叉验证训练
    for fold, (train_loader, val_loader) in enumerate(fold_dataloaders):
        if accelerator.is_local_main_process:
            logger.info(f"\n开始训练第 {fold + 1} 折...")
        
        # 初始化模型
        input_dim = preprocessor.word2vec.vector_size
        model = SentimentLSTM(config, input_dim)
        criterion = nn.CrossEntropyLoss()
        
        # 使用AdamW优化器
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['model']['learning_rate'],
            weight_decay=config['model']['weight_decay'],
            betas=(config['model']['beta1'], config['model']['beta2'])
        )
        
        # 创建学习率调度器
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['model']['learning_rate'],
            epochs=config['model']['epochs'],
            steps_per_epoch=len(train_loader),
            pct_start=config['model']['warmup_epochs'] / config['model']['epochs'],
            div_factor=25,
            final_div_factor=1e5,
            anneal_strategy='cos'
        )
        
        # 使用 Accelerator 准备模型和数据加载器
        model, optimizer, train_loader, val_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader
        )
        
        # 初始化训练历史记录
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
        # 训练循环
        best_val_acc = 0
        best_metrics = None
        for epoch in range(config['model']['epochs']):
            # 训练
            train_loss, train_acc, epoch_lrs = train_epoch(
                model, train_loader, criterion, optimizer, scheduler, accelerator
            )
            history['learning_rates'].extend(epoch_lrs)
            
            # 验证
            val_loss, val_acc, metrics = validate(model, val_loader, criterion, accelerator)
            
            # 记录历史
            if accelerator.is_local_main_process:
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                history['val_metrics'].append(metrics)
                
                # 输出日志
                logger.info(f'Fold {fold + 1}, Epoch {epoch+1}/{config["model"]["epochs"]}:')
                logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
                logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
                logger.info(f'Val Precision: {metrics["precision"]:.4f}, Val Recall: {metrics["recall"]:.4f}, Val F1: {metrics["f1"]:.4f}')
                logger.info(f'Learning Rate: {epoch_lrs[-1]:.6f}')
                
                # 保存当前折的最佳模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_metrics = metrics
                    
                    # 如果是所有折中的最佳模型，则保存
                    if val_acc > best_fold_acc:
                        best_fold_acc = val_acc
                        best_fold_metrics = metrics
                        best_fold_model = accelerator.unwrap_model(model).state_dict()
                        
                        model_dir = Path(config['data']['model_dir'])
                        model_dir.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            best_fold_model,
                            model_dir / f'best_model_fold_{fold + 1}.pth'
                        )
                        logger.info(f'保存第 {fold + 1} 折的最佳模型，验证准确率: {val_acc:.4f}')
        
        if accelerator.is_local_main_process:
            # 保存当前折的训练历史
            all_fold_results.append({
                'fold': fold + 1,
                'history': history,
                'best_val_acc': best_val_acc,
                'best_metrics': best_metrics
            })
            
            # 绘制当前折的训练历史
            plot_training_history(history, fold + 1)
            
            # 输出当前折的评估报告
            logger.info(f"\n第 {fold + 1} 折评估报告:")
            logger.info(f'Best Accuracy: {best_metrics["accuracy"]:.4f}')
            logger.info(f'Best Precision: {best_metrics["precision"]:.4f}')
            logger.info(f'Best Recall: {best_metrics["recall"]:.4f}')
            logger.info(f'Best F1 Score: {best_metrics["f1"]:.4f}')
            
            logger.info("\nClassification Report:")
            logger.info("\n" + classification_report(
                best_metrics["targets"],
                best_metrics["predictions"],
                target_names=['负面', '中性', '正面'],
                digits=4
            ))
    
    if accelerator.is_local_main_process:
        # 输出所有折的平均结果
        avg_metrics = {
            'accuracy': np.mean([fold['best_metrics']['accuracy'] for fold in all_fold_results]),
            'precision': np.mean([fold['best_metrics']['precision'] for fold in all_fold_results]),
            'recall': np.mean([fold['best_metrics']['recall'] for fold in all_fold_results]),
            'f1': np.mean([fold['best_metrics']['f1'] for fold in all_fold_results])
        }
        
        logger.info("\n交叉验证总结:")
        logger.info(f"平均准确率: {avg_metrics['accuracy']:.4f} ± {np.std([fold['best_metrics']['accuracy'] for fold in all_fold_results]):.4f}")
        logger.info(f"平均精确率: {avg_metrics['precision']:.4f} ± {np.std([fold['best_metrics']['precision'] for fold in all_fold_results]):.4f}")
        logger.info(f"平均召回率: {avg_metrics['recall']:.4f} ± {np.std([fold['best_metrics']['recall'] for fold in all_fold_results]):.4f}")
        logger.info(f"平均F1分数: {avg_metrics['f1']:.4f} ± {np.std([fold['best_metrics']['f1'] for fold in all_fold_results]):.4f}")
        
        # 保存最终的最佳模型
        if best_fold_model is not None:
            model_dir = Path(config['data']['model_dir'])
            torch.save(best_fold_model, model_dir / 'best_model_final.pth')
            logger.info(f"\n最佳模型已保存，验证准确率: {best_fold_acc:.4f}")
        
        logger.info("训练完成！")

if __name__ == "__main__":
    main() 