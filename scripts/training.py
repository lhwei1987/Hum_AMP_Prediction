import hashlib
import os
import pickle
import random
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
import math
from scipy.spatial import cKDTree  # 导入cKDTree用于高效最近邻搜索
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter

from models.multimodal_cnn import AMPPredictionModel
from src.data_loader import create_data_loader
from src.preprocessing import load_dataset
from src.utils import Tokenizer, analyze_sequence_position_impact, calculate_property_importance, visualize_shap_values, \
    visualize_property_importance, compare_amp_nonamp_properties, analyze_position_interactions, \
    analyze_peptide_contribution
from src.utils import calculate_metrics, plot_roc_curve, plot_pr_curve, plot_confusion_matrix, PROPERTY_NAMES, \
    generate_shap_explanation
from src.utils import progress_bar, time_estimator


class EarlyStopping:
    """早停机制，支持多种指标和模式"""

    def __init__(self, patience=10, delta=0, mode='min', metric='loss'):
        """
        参数:
        patience: 在停止前等待的epoch数
        delta: 被视为改进的最小变化量
        mode: 'min' 或 'max'，表示指标是越小越好还是越大越好
        metric: 要监控的指标名称
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode
        self.metric = metric

        if self.mode not in ['min', 'max']:
            raise ValueError(f"无效的mode参数: {mode}。必须是'min'或'max'")

        print(f"早停机制配置: 模式={mode}, 指标={metric}, 耐心={patience}, 最小变化={delta}")

    def __call__(self, current_metric):
        if self.best_score is None:
            self.best_score = current_metric
            print(f"初始化最佳{self.metric}: {self.best_score:.4f}")
            return

        if self.mode == 'min':
            # 指标越小越好（如损失）
            if current_metric < self.best_score - self.delta:
                improvement = self.best_score - current_metric
                self.best_score = current_metric
                self.counter = 0
                print(f"{self.metric}改善: +{improvement:.4f} (新最佳: {self.best_score:.4f})")
            else:
                self.counter += 1
                print(
                    f"{self.metric}未改善: 当前 {current_metric:.4f}, 最佳 {self.best_score:.4f}, 计数器 {self.counter}/{self.patience}")
        else:
            # 指标越大越好（如特异性）
            if current_metric > self.best_score + self.delta:
                improvement = current_metric - self.best_score
                self.best_score = current_metric
                self.counter = 0
                print(f"{self.metric}改善: +{improvement:.4f} (新最佳: {self.best_score:.4f})")
            else:
                self.counter += 1
                print(
                    f"{self.metric}未改善: 当前 {current_metric:.4f}, 最佳 {self.best_score:.4f}, 计数器 {self.counter}/{self.patience}")

        if self.counter >= self.patience:
            self.early_stop = True
            print(f"早停触发: 指标 '{self.metric}' 在 {self.patience} 个epoch内未改善")


# 新增函数：检查序列数据是否变化
def sequences_changed(new_sequences, cached_sequences):
    """检查序列数据是否发生变化"""
    if len(new_sequences) != len(cached_sequences):
        return True

    for i, seq in enumerate(new_sequences):
        if seq != cached_sequences[i]:
            return True

    return False
#调试
def check_feature_usage(model, props, seq, evo_features):
    # 前向传播
    output = model(seq, props, evo_features)

def diagnose_model_sensitivity(model, tokenizer, data_loader, device, config, save_dir):
    """诊断模型对特征变化的敏感性"""
    print("\n[模型诊断]")

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 1. 检查属性特征是否归一化
    sample_props = data_loader.dataset.properties[0]
    print("属性统计 - 最小值:", sample_props.min(axis=0))
    print("属性统计 - 最大值:", sample_props.max(axis=0))

    # 保存属性统计信息
    with open(os.path.join(save_dir, "property_stats.txt"), 'w') as f:
        f.write(f"属性最小值: {sample_props.min(axis=0)}\n")
        f.write(f"属性最大值: {sample_props.max(axis=0)}\n")

    # 2. 检查输入梯度
    model.eval()
    batch_data = next(iter(data_loader))

    # 正确提取数据元素
    sequence = batch_data['seq_ids'].to(device)
    labels = batch_data['labels'].to(device)
    properties = batch_data['properties'].to(device)
    orig_sequences = batch_data['sequence_strs']  # 原始序列列表（字符串）

    # 确保属性需要梯度
    properties.requires_grad = True

    # 模型前向传播
    outputs = model(sequence, properties, orig_sequences)

    # 反向传播计算梯度
    outputs.sum().backward()

    # 计算梯度统计
    grad_mean = properties.grad.abs().mean().item()
    grad_max = properties.grad.abs().max().item()
    print(f"输入属性梯度 - 均值: {grad_mean:.6f}, 最大值: {grad_max:.6f}")

    # 保存梯度信息
    with open(os.path.join(save_dir, "gradient_stats.txt"), 'w') as f:
        f.write(f"梯度均值: {grad_mean:.6f}\n")
        f.write(f"梯度最大值: {grad_max:.6f}\n")

    # 3. 检查属性通道重要性
    print("\n属性通道灵敏度测试:")
    base_outputs = model(sequence, properties, orig_sequences)
    base_prob = torch.sigmoid(base_outputs).mean().item()  # 使用平均值
    sensitivity_results = []

    for i in range(properties.shape[2]):  # 对每个属性特征
        mod_properties = properties.clone()
        mod_properties[:, :, i] = 0  # 归零特定特征

        mod_outputs = model(sequence, mod_properties, orig_sequences)
        mod_prob = torch.sigmoid(mod_outputs).mean().item()  # 使用平均值

        delta = abs(base_prob - mod_prob)
        result_line = f"特征 {PROPERTY_NAMES[i]}: Δ={delta:.4f} {'*' if delta > 0.01 else ''}"
        print(result_line)
        sensitivity_results.append(result_line)

        # 保存灵敏度测试结果
    with open(os.path.join(save_dir, "feature_sensitivity.txt"), 'w') as f:
        f.write("\n".join(sensitivity_results))

    # 4. 给出具体建议
    print("\n[修复建议]")
    recommendations = []

    if grad_mean < 1e-4:
        rec1 = "1. 建议添加特征重要性正则化: 在损失函数中加入梯度惩罚项\n   loss = main_loss + 0.1 * |d(output)/d(input)|"
        print(rec1)
        recommendations.append(rec1)

    if any(delta < 0.01 for i in range(properties.shape[2])):
        rec2 = "2. 建议调整模型结构: 增加属性特征与序列嵌入的交互"
        print(rec2)
        recommendations.append(rec2)

    rec3 = "3. 验证属性特征计算方法: 某些特征可能包含无效值"
    rec4 = "4. 尝试提升学习率或延长训练周期"
    print(rec3)
    print(rec4)
    recommendations.extend([rec3, rec4])

    # 保存建议
    with open(os.path.join(save_dir, "recommendations.txt"), 'w') as f:
        f.write("\n".join(recommendations))

    # 特征通道灵敏度对比图
    feature_changes = []
    feature_names = PROPERTY_NAMES

    for i in range(len(feature_names)):
        mod_properties = properties.clone()
        mod_properties[:, :, i] = 0
        mod_outputs = model(sequence, mod_properties, orig_sequences)
        mod_prob = torch.sigmoid(mod_outputs).mean().item()
        delta = abs(base_prob - mod_prob)
        feature_changes.append(delta)

    plt.figure(figsize=(12, 6))
    plt.bar(feature_names, feature_changes, color='skyblue')
    plt.title('Model Sensitivity to Property Removal')
    plt.ylabel('Probability Change')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_sensitivity.png"))
    plt.close()

    print("\n特征剔除灵敏度测试结果:")
    for name, change in zip(feature_names, feature_changes):
        print(f"  {name}: Δ={change:.5f}")

    # 梯度可视化
    gradient_magnitudes = properties.grad.abs().mean(dim=(0, 1)).cpu().detach().numpy()
    plt.figure(figsize=(12, 6))
    plt.bar(feature_names, gradient_magnitudes, color='salmon')
    plt.title('Average Gradient Magnitude per Feature')
    plt.ylabel('Gradient L1 Norm')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_gradients.png"))
    plt.close()

    print(f"模型诊断结果已保存至: {save_dir}")

# 新增函数：验证数据完整性
def validate_data_integrity(sequences, labels, properties, data_type, max_len):
    """验证数据完整性并打印调试信息"""
    start_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] 验证{data_type}数据完整性...")

    try:
        # 1. 检查序列和标签数量是否匹配
        if len(sequences) != len(labels):
            print(f"⚠️ 严重错误: {data_type}序列数({len(sequences)}) ≠ 标签数({len(labels)})")
            print(f"  序列样本: {types(sequences[:5])}...")
            print(f"  标签样本: {labels[:5]}...")
            raise ValueError(f"{data_type}序列数和标签数不匹配")

        # 2. 检查属性和序列数量是否匹配
        if len(properties) > 0 and len(properties) != len(sequences):
            print(f"⚠️ 严重错误: {data_type}属性数({len(properties)}) ≠ 序列数({len(sequences)})")
            raise ValueError(f"{data_type}属性数和序列数不匹配")

        # 3. 检查特征矩阵维度
        for i in range(min(10, len(sequences))):  # 只检查前10个样本避免过多输出
            # 获取原始序列（去除填充）
            orig_sequence = truncate_padded_sequence(sequences[i], max_len)
            orig_seq_len = len(orig_sequence)

            # 检查属性矩阵长度
            if len(properties) > 0:
                prop_shape = properties[i].shape

                # 属性矩阵应该与最大序列长度一致
                if prop_shape[0] != max_len:
                    print(f"⚠️ 样本 {i} 警告: 属性矩阵长度({prop_shape[0]}) ≠ 最大序列长度({max_len})")

                # 检查是否与原始序列匹配
                if orig_seq_len > max_len:
                    print(f"⚠️ 样本 {i} 警告: 序列长度({orig_seq_len}) > 最大序列长度({max_len}) - 将被截断")
                elif prop_shape[0] != max_len:
                    print(f"⚠️ 样本 {i} 警告: 属性矩阵长度({prop_shape[1]})应等于最大序列长度({max_len})")

                # 检查特征数量
                if prop_shape[1] != len(PROPERTY_NAMES):
                    print(f"⚠️ 样本 {i} 警告: 特征数量({prop_shape[1]}) ≠ 定义的特征名称数量({len(PROPERTY_NAMES)}")
    except Exception as e:
        print(f"验证过程中发生异常: {str(e)}")

    end_time = time.time()
    process_time = end_time - start_time
    print(f"[{time.strftime('%H:%M:%S')}] 验证完成: 处理 {len(sequences)} 条序列耗时 {process_time:.2f} 秒")


# 新增函数：分析特征相关性
def analyze_feature_correlations(properties, labels, max_len):
    """分析属性特征与标签的相关性，考虑序列长度差异"""
    if not isinstance(properties, (list, np.ndarray)):
        properties = list(properties)

    n_samples = len(properties)
    if n_samples == 0:
        print("⚠️ 警告: 无有效属性特征可用于相关性分析")
        return np.zeros(len(PROPERTY_NAMES))

    # 获取每个样本实际序列长度
    seq_lengths = [len(prop) for prop in properties]

    # 创建特征矩阵 - 对每个特征取平均值
    n_features = len(PROPERTY_NAMES)
    mean_features = np.zeros((n_samples, n_features))

    for i in range(n_samples):
        # 只取实际序列部分（去除填充部分）
        actual_props = properties[i][:seq_lengths[i], :]
        if len(actual_props) > 0:
            mean_features[i, :] = actual_props.mean(axis=0)
        else:
            mean_features[i, :] = properties[i][0, :]  # 保底处理

    # 计算相关系数
    labels = np.array(labels).flatten()
    correlations = []

    for j in range(n_features):
        try:
            corr = np.corrcoef(mean_features[:, j], labels)[0, 1]
        except:
            corr = 0.0
        correlations.append(corr)

    # 可视化特征相关性
    plt.figure(figsize=(12, 6))
    plt.bar(PROPERTY_NAMES, correlations, color='skyblue')
    plt.title('Feature Correlation with Labels')
    plt.xlabel('Property Features')
    plt.ylabel('Correlation Coefficient')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 标记显著相关性
    for i, corr in enumerate(correlations):
        if abs(corr) > 0.1:
            plt.text(i, corr, f'{corr:.2f}', ha='center', va='bottom' if corr > 0 else 'top')

    return np.array(correlations)


# 新增：序列填充助手函数
def truncate_padded_sequence(sequence, max_len):
    """去除填充部分，恢复原始序列"""
    return sequence.replace('-', '')


def calculate_property_importance(model, val_loader, device):
    """更可靠的特征重要性计算方法"""
    model.eval()
    importance_scores = np.zeros(len(PROPERTY_NAMES))

    # 获取基线预测
    with torch.no_grad():
        total_diff = 0.0
        count = 0

        # 遍历验证集中的所有样本
        for batch_idx, batch_data in enumerate(val_loader):
            # 从字典中获取数据
            seq = batch_data['seq_ids'].to(device)
            labels = batch_data['labels'].to(device)
            props = batch_data['properties'].to(device)
            orig_sequences = batch_data['sequence_strs']  # 原始序列列表

            # 原始预测
            outputs = model(seq, props, orig_sequences)

            # 计算原始概率
            orig_probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            # 对每个属性特征计算变化
            for prop_idx in range(len(PROPERTY_NAMES)):
                # 创建修改后的属性矩阵（该特征全部置零）
                mod_props = props.clone()
                mod_props[:, :, prop_idx] = 0

                # 预测修改后的属性
                mod_outputs = model(seq, mod_props, orig_sequences)

                mod_probs = torch.sigmoid(mod_outputs).cpu().numpy().flatten()

                # 计算变化差异
                diffs = np.abs(orig_probs - mod_probs)
                total_diff += diffs.sum()
                count += len(seq)

                # 保存平均差异
                importance_scores[prop_idx] += diffs.sum()

        # 计算平均重要性分数
        if count > 0:
            importance_scores /= count

    # 标准化重要性分数 (0-1范围)
    max_score = importance_scores.max()
    if max_score > 0:
        importance_scores /= max_score
    else:
        # 如果所有分数都是0，则返回均匀分布的重要性分数
        importance_scores = np.ones(len(PROPERTY_NAMES)) / len(PROPERTY_NAMES)

    return importance_scores


def create_sensitivity_dataset(config):
    """创建用于检测模型敏感性的合成数据集"""
    # 创建含有明显模式的可解释合成数据
    patterns = {
        "AMP": [
            "GLFDIVKKVVGA"  # 阳离子模式
            "WWKWWWRRR"  # 疏水性增强
        ],
        "Non-AMP": [
            "GAGAGAAPAPAP"  # 中性模式
            "TTTTSSSSQQQQ"  # 低电荷模式
        ]
    }

    sequences = []
    labels = []
    properties = []

    # 生成带明确特征的合成肽
    for label, group in patterns.items():
        for pattern in group:
            # 复制序列并添加变化
            for i in range(10):
                seq = pattern + random.choice(['A', 'R', 'K', 'L', 'W'][:i % 5])
                sequences.append(seq)
                labels.append(1 if label == "AMP" else 0)
                # 创建有规律的物化特性矩阵
                prop_matrix = np.zeros((len(seq), len(PROPERTY_NAMES)))
                for j, prop in enumerate(PROPERTY_NAMES):
                    # 根据不同特征设置规律性值
                    if 'Charge' in prop:
                        val = np.abs(np.sin(j / 2)) if label == "AMP" else 0.1
                    elif 'Hydrophobicity' in prop:
                        val = 0.8 if 'W' in seq or 'F' in seq else 0.3
                    else:  # 其他特征
                        val = np.random.uniform(0.2, 0.8)
                    prop_matrix[:, j] = val
                properties.append(prop_matrix)

    return sequences, np.array(labels), properties


def manual_get_shap_explanation(model, shap_loader, device, config):
    """手动计算SHAP值的方法"""
    model.eval()
    baseline_value = 0.0  # 基准值
    # 收集样本和模型输出
    all_sequences = []
    all_properties = []
    all_outputs = []
    # 计算基准值
    for batch_idx, batch_data in enumerate(shap_loader):
        seq = batch_data['seq_ids'].to(device)
        labels = batch_data['labels'].to(device)
        props = batch_data['properties'].to(device)
        orig_sequences = batch_data['sequence_strs']  # 获取原始序列字符串
        with torch.no_grad():
            # 添加 orig_sequences 参数
            outputs = model(seq, props, orig_sequences)  # 修改这里
            prob = torch.sigmoid(outputs).squeeze().cpu().numpy()
            all_outputs.append(prob)

    baseline_value = np.mean(all_outputs)

    # 计算样本重要性
    shap_values = []
    for batch_idx, batch_data in enumerate(shap_loader):
        seq = batch_data['seq_ids'].to(device)
        labels = batch_data['labels'].to(device)
        props = batch_data['properties'].to(device)
        orig_sequences = batch_data['sequence_strs']  # 获取原始序列字符串
        # 原始输出
        with torch.no_grad():
            # 添加 orig_sequences 参数
            outputs = model(seq, props, orig_sequences)  # 修改这里
            full_prob = torch.sigmoid(outputs).item()

        # 创建空SHAP矩阵
        batch_shap = np.zeros((seq.size(0), seq.size(1), props.size(2)))

        # 为每个位置计算SHAP值
        for sample_idx in range(seq.size(0)):
            for pos_idx in range(seq.size(1)):
                for prop_idx in range(props.size(2)):
                    # 创建副本
                    mutated_props = props.clone()
                    mutated_props[sample_idx, pos_idx, prop_idx] = 0  # 消除该特征

                    # 计算输出差异
                    with torch.no_grad():
                        mutated_outputs = model(seq, mutated_props, orig_sequences)
                        mutated_prob = torch.sigmoid(mutated_outputs[sample_idx]).item()

                    # SHAP值 = 原始输出 - 特征缺失时的输出
                    batch_shap[sample_idx, pos_idx, prop_idx] = full_prob - mutated_prob

        shap_values.append(batch_shap)

    # 合并批次结果
    shap_values = np.concatenate(shap_values, axis=0)
    return shap_values
@time_estimator
def train_model(config):
    """训练模型主函数"""
    # 从配置获取保存目录
    config_save_dir = config['training'].get('save_dir', 'results')
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # 创建配置指定的保存目录
    config_save_dir_full = os.path.join(config_save_dir, f"training_{timestamp}")
    os.makedirs(config_save_dir_full, exist_ok=True)
    os.makedirs(os.path.join(config_save_dir_full, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(config_save_dir_full, "visualizations"), exist_ok=True)

    # 创建最佳模型保存目录
    best_model_dir = "checkpoints"
    os.makedirs(best_model_dir, exist_ok=True)

    # 保存配置文件到两个位置
    with open(os.path.join(config_save_dir_full, "config.yaml"), 'w') as f:
        yaml.dump(config, f)
    with open(os.path.join(best_model_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f)

    # 初始化TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(config_save_dir_full, "tensorboard"))

    # 创建 tokenizer 实例
    tokenizer = Tokenizer()

    # 将 tokenizer 添加到配置中
    config['data']['tokenizer'] = tokenizer

    #########################################################################
    # 获取假阳性率控制参数
    focus_fpr = config['training'].get('focus_fpr', True)  # 默认启用假阳性率优化
    negative_weight = config['training'].get('negative_weight', 1)  # 增加负样本权重
    fpr_target = config['training'].get('fpr_target', 0.05)  # 更严格的目标假阳性率
    print(f"假阳性率优化模式: {'启用' if focus_fpr else '禁用'}")
    print(f"负样本权重: {negative_weight}, 目标FPR: {fpr_target}")
    #########################################################################

    # 获取最大序列长度
    max_seq_len = config['data'].get('max_seq_len', 100)

    # 加载数据
    print(f"[{time.strftime('%H:%M:%S')}] Loading training data...")
    train_sequences, train_labels, train_properties = load_dataset(
        config['data']['train_path'],
        max_len=max_seq_len,
        augment=config['data']['augment'],
        show_progress=config['show_progress']
    )

    # 数据验证
    validate_data_integrity(train_sequences, train_labels, train_properties, "训练", max_seq_len)

    print(f"[{time.strftime('%H:%M:%S')}] Loading validation data...")
    val_sequences, val_labels, val_properties = load_dataset(
        config['data']['val_path'],
        max_len=max_seq_len
    )

    # 数据验证
    validate_data_integrity(val_sequences, val_labels, val_properties, "验证", max_seq_len)

    # ============================================================
    # 添加数据形状检查
    # ============================================================
    print("\n" + "=" * 60)
    print("DATA SHAPE CHECK")
    print("=" * 60)

    # 1. 检查基本形状
    max_len = config['data']['max_seq_len']
    print(f"\n[基本信息]")
    print(f"配置的最大序列长度: {max_seq_len}")
    print(f"训练集样本数: {len(train_sequences)}")
    print(f"验证集样本数: {len(val_sequences)}")

    # 2. 检查训练集形状
    print(f"\n[训练集检查]")

    # 检查标签形状
    if isinstance(train_labels, np.ndarray):
        print(f"标签形状: {train_labels.shape}")
    else:
        print(f"标签类型: {type(train_labels)}")

    # 检查特征形状
    if isinstance(train_properties, np.ndarray):
        print(f"特征数组形状: ({len(train_properties)}, {max_seq_len}, {len(PROPERTY_NAMES)})")
        if len(train_properties.shape) == 3:
            print(f"特征维度: {train_properties.shape[1]} (序列长度) x {train_properties.shape[2]} (特征数)")
        else:
            print("⚠️ 警告: 特征数组维度不是三维 (样本数, 序列长度, 特征数)")
    else:
        print(f"特征类型: {type(train_properties)}")

    # 3. 检查验证集形状
    print(f"\n[验证集检查]")

    # 检查标签形状
    if isinstance(val_labels, np.ndarray):
        print(f"标签形状: {val_labels.shape}")
    else:
        print(f"标签类型: {type(val_labels)}")

    # 检查特征形状
    if isinstance(val_properties, np.ndarray):
        print(f"特征数组形状: ({len(val_properties)}, {max_seq_len}, {len(PROPERTY_NAMES)})")
        if len(val_properties.shape) == 3:
            print(f"特征维度: {val_properties.shape[1]} (序列长度) x {val_properties.shape[2]} (特征数)")
        else:
            print("⚠️ 警告: 特征数组维度不是三维 (样本数, 序列长度, 特征数)")
    else:
        print(f"特征类型: {type(val_properties)}")

    # 4. 详细样本检查
    print(f"\n[序列长度检查]")
    train_seq_lengths = [len(seq) for seq in train_sequences]
    val_seq_lengths = [len(seq) for seq in val_sequences]

    print(f"训练集最小序列长度: {min(train_seq_lengths)}")
    print(f"训练集最大序列长度: {max(train_seq_lengths)}")
    print(f"验证集最小序列长度: {min(val_seq_lengths)}")
    print(f"验证集最大序列长度: {max(val_seq_lengths)}")

    # 检查是否有序列超过最大长度
    train_long_seqs = [i for i, l in enumerate(train_seq_lengths) if l > max_len]
    val_long_seqs = [i for i, l in enumerate(val_seq_lengths) if l > max_len]

    if train_long_seqs:
        print(f"⚠️ 训练集中有 {len(train_long_seqs)} 个序列长度超过最大长度 {max_len}")
        print(f"  示例索引: {train_long_seqs[:5]}")
        print(f"  示例序列: {train_sequences[train_long_seqs[0]]} (长度: {train_seq_lengths[train_long_seqs[0]]})")
    else:
        print("✅ 所有训练序列长度 <= 最大长度")

    if val_long_seqs:
        print(f"⚠️ 验证集中有 {len(val_long_seqs)} 个序列长度超过最大长度 {max_len}")
        print(f"  示例索引: {val_long_seqs[:5]}")
        print(f"  示例序列: {val_sequences[val_long_seqs[0]]} (长度: {val_seq_lengths[val_long_seqs[0]]})")
    else:
        print("✅ 所有验证序列长度 <= 最大长度")

    print("=" * 60 + "\n")
    # ============================================================
    # 结束数据形状检查
    # ============================================================

    # ====== 添加这里 ====== 特征相关性分析部分 ======
    print(f"[{time.strftime('%H:%M:%S')}] 分析特征相关性...")

    # 特征相关性分析 (只在有足够特征数据时)
    # 检查是否有足够的有效数据进行分析
    if len(train_properties) > 0 and all(len(p) == max_seq_len for p in train_properties):
        # 使用最大序列长度参数进行分析
        train_correlations = analyze_feature_correlations(train_properties, train_labels, max_seq_len)

        # 保存相关性图
        save_path = os.path.join(config_save_dir_full, "feature_correlations.png")
        plt.savefig(save_path)
        plt.close()
        print(f"特征相关性图保存至: {save_path}")

        # 找出重要特征索引
        important_indices = [i for i, corr in enumerate(train_correlations) if abs(corr) > 0.1]
        print(f"显著特征数量: {len(important_indices)}/{len(PROPERTY_NAMES)}")
    else:
        print("⚠️ 警告: 训练属性数据无效或缺失，跳过相关性分析")
        important_indices = list(range(len(PROPERTY_NAMES)))
    # ========================================================

    # 在创建数据加载器之前添加
    print("\n[数据加载器前检查]")
    # 检查序列长度是否一致
    train_seq_lengths = [len(seq) for seq in train_sequences]
    val_seq_lengths = [len(seq) for seq in val_sequences]
    print(f"训练集序列长度范围: {min(train_seq_lengths)}-{max(train_seq_lengths)}")
    print(f"验证集序列长度范围: {min(val_seq_lengths)}-{max(val_seq_lengths)}")
    # 检查特征矩阵形状
    train_prop_shapes = [prop.shape for prop in train_properties]
    val_prop_shapes = [prop.shape for prop in val_properties]
    print(f"训练集特征矩阵形状: {set(train_prop_shapes)}")
    print(f"验证集特征矩阵形状: {set(val_prop_shapes)}")

    # 在创建数据加载器前添加
    # 根据GPU内存调整批处理大小
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        free_memory = gpu_memory - torch.cuda.memory_allocated(0)
        print(f"GPU总内存: {gpu_memory / 1024 ** 3:.2f} GB")
        print(f"GPU可用内存: {free_memory / 1024 ** 3:.2f} GB")

        # 动态调整批处理大小
        if gpu_memory < 8 * 1024 ** 3:  # 小于8GB显存
            config['data']['batch_size'] = 32
            print(f"检测到有限GPU内存，调整批处理大小为: {config['data']['batch_size']}")
    else:
        print("使用CPU进行训练")
        config['data']['batch_size'] = 64  # CPU可以处理更大的批次

    # 创建数据加载器
    print("\n创建数据加载器...")

    # 更新数据加载器创建
    train_loader = create_data_loader(
        sequences=train_sequences,
        labels=train_labels,
        properties=train_properties,
        config=config,
        shuffle=True
    )
    val_loader = create_data_loader(
        sequences=val_sequences,
        labels=val_labels,
        properties=val_properties,
        config=config,
        shuffle=False
    )
    ####################################################################################
    #训练主循环
    ####################################################################################
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AMPPredictionModel(config['model']).to(device)
    model.config['train_sequences'] = train_sequences  # 添加这行
    ####################################################################################

    # 确保模型有kl_loss方法（如果模型类没有实现）
    if not hasattr(model, 'kl_loss'):
        # 使用设备的默认实现
        model.kl_loss = lambda: torch.tensor(0.0, device=device)
    if not hasattr(model, 'pinn_loss'):
        model.pinn_loss = lambda seq, props, pred: torch.tensor(0.0, device=device)

    if not hasattr(model, 'predict_with_uncertainty'):
        # 改进的默认实现，支持多任务输出
        def default_predict_with_uncertainty(self, seq, props, n_samples=10):
            """带不确定性的预测 - 支持多任务模型"""
            # 获取模型输出
            with torch.no_grad():
                outputs = self(seq, props)

            # 处理多任务输出
            if isinstance(outputs, dict):
                mean_pred = {}
                std_pred = {}
                for task in outputs.keys():
                    task_output = outputs[task]
                    # 使用一致的键名
                    mean_pred[f"{task}_mean"] = task_output.detach().cpu().numpy()
                    std_pred[f"{task}_std"] = torch.zeros_like(task_output).cpu().numpy()
                return mean_pred, std_pred, None
            else:
                # 单任务模型
                return (
                    outputs.detach().cpu().numpy(),
                    torch.zeros_like(outputs).cpu().numpy(),
                    None
                )

        model.predict_with_uncertainty = default_predict_with_uncertainty.__get__(model)

    # 获取KL和PINN损失的权重（从配置中读取，默认为0.01和0.1）
    kl_weight = config['training'].get('kl_weight', 0.01)
    pinn_weight = config['training'].get('pinn_weight', 0.1)
    print(f"KL损失权重: {kl_weight}, PINN损失权重: {pinn_weight}")

    ###################################################################################
    # 根据正负样本比例动态设置权重
    # 增加负样本权重（non-AMP样本）
    neg_weight = config['training'].get('negative_weight', 1)  # 更高的值增加负样本权重
    print(f"负样本权重: {neg_weight}")
    # 根据正负样本比例设置权重
    num_pos = max(1, np.sum(train_labels == 1))
    num_neg = np.sum(train_labels == 0)
    pos_weight = num_neg / num_pos  # 正样本权重
    neg_weight = neg_weight  # 负样本权重增强因子
    # 计算加权损失
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight]).to(device),
        reduction='none'  # 手动加权
    )
    ###################################################################################

    # 损失函数和优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # 监控F1（越大越好）
        factor=0.5,  # 学习率衰减因子
        patience=3,  # 3个epoch无改善则降低LR
        min_lr=1e-6,  # 最小学习率
        threshold=0.001  # 最小改进阈值
    )
    ###################################################################################
    # 早停机制 - 优先监控F1值
    early_stopping = EarlyStopping(
        patience=config['training']['patience'],
        delta=config['training']['delta'],
        mode='max',  # 最大化F1值
        metric='F1'
    )
    ###################################################################################


    ###################################################################################
    # 训练循环
    # 初始化变量
    best_f1 = 0.0
    best_threshold = 0.5  # 最佳阈值
    global_step = 0

    # 学习率预热配置
    warmup_epochs = config['training'].get('warmup_epochs', 3)
    base_lr = config['training']['learning_rate']

    # best_val_f1 = 0.0
    for epoch in range(config['training']['epochs']):
        print(f"\n[Epoch {epoch + 1}/{config['training']['epochs']}]")
        print("-" * 50)

        # 在训练初期使用较小的学习率-添加学习率预热
        if epoch < warmup_epochs:
            # 使用余弦预热更平滑
            lr_scale = 0.5 * (1 + math.cos(math.pi * (epoch) / warmup_epochs))
            current_lr = base_lr * lr_scale

            # 避免覆盖调度器的内部状态
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        else:
            # 确保使用调度器设置的学习率
            current_lr = optimizer.param_groups[0]['lr']

        # 记录学习率
        writer.add_scalar('Learning Rate', current_lr, epoch)
        print(f"Epoch {epoch + 1} - 学习率: {current_lr:.7f}")

        # 在训练循环中添加正样本预测监控
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_kl_loss = 0.0
        train_pinn_loss = 0.0
        train_pos_preds = 0.0
        train_total = 0
        train_bar = progress_bar(train_loader, desc="Training", unit="batch")

        # 定义训练损失函数 - 用于训练循环
        train_criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]).to(device),
            reduction='none'  # 返回每个样本的损失
        )

        # 训练循环
        for batch_idx, batch_data in enumerate(train_loader):
            seq = batch_data['seq_ids'].to(device)
            labels = batch_data['labels'].to(device)
            props = batch_data['properties'].to(device)
            orig_sequences = batch_data['sequence_strs']  # 原始序列列表

            labels = labels.float().unsqueeze(1)  # 从[32]变为[32, 1]

            # 前向传播
            outputs = model(seq, props, orig_sequences)

            # 计算BCE损失
            bce_loss = F.binary_cross_entropy_with_logits(
                outputs,
                labels.float(),
                reduction='mean'  # 确保返回标量
            )

            # 计算KL损失
            kl_loss = model.kl_loss() * kl_weight
            # 计算物理约束损失
            pinn_loss = model.pinn_loss(seq, props, outputs) * pinn_weight
            # 组合总损失
            loss = bce_loss + kl_loss + pinn_loss
            # 确保损失是标量
            if loss.dim() > 0:
                loss = loss.mean()

            # 计算预测值（用于统计）
            preds = (torch.sigmoid(outputs) > 0.5).float()
            accuracy = (preds == labels).float().mean().item()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # 更新统计
            train_loss += bce_loss.item()
            train_kl_loss += kl_loss.item()
            train_pinn_loss += pinn_loss.item()
            train_pos_preds += preds.sum().item()
            train_total += labels.size(0)

            # 更新进度条
            pos_rate = train_pos_preds / train_total
            train_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "bce": f"{bce_loss.item():.4f}",
                "kl": f"{kl_loss.item():.4f}",
                "pinn": f"{pinn_loss.item():.4f}",
                "acc": f"{accuracy:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
                "pos_rate": f"{pos_rate:.4f}"
            })

            # 更新进度条和全局步数
            train_bar.update(1)
            global_step += 1

        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        avg_train_kl_loss = train_kl_loss / len(train_loader)
        avg_train_pinn_loss = train_pinn_loss / len(train_loader)

        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss_KL/train', avg_train_kl_loss, epoch)
        writer.add_scalar('Loss_PINN/train', avg_train_pinn_loss, epoch)

        ###################################################################################


        # 验证循环
        ###################################################################################

        # 验证集评估
        model.eval()
        all_labels, all_probs, all_preds = [], [], []
        val_loss = 0.0  # 添加损失统计
        val_bar = progress_bar(train_loader, desc="Validation", unit="batch")

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_loader):
                seq = batch_data['seq_ids'].to(device)
                labels = batch_data['labels'].to(device)
                props = batch_data['properties'].to(device)
                orig_sequences = batch_data['sequence_strs']

                outputs = model(seq, props, orig_sequences)

                probs = torch.sigmoid(outputs).cpu()

                # 收集结果
                all_labels.append(labels.cpu())
                all_probs.append(probs)
                all_preds.append((probs > 0.5).float())

                # 计算验证损失
                labels = labels.float().unsqueeze(1)
                bce_loss = F.binary_cross_entropy_with_logits(outputs, labels)
                val_loss += bce_loss.item()

                # 计算当前批次的准确率（修复后的代码）
                batch_labels = labels.cpu()  # 将标签移到CPU
                batch_preds = (probs > 0.5).float()  # 在CPU上计算预测
                batch_correct = (batch_preds == batch_labels).sum().item()
                batch_accuracy = batch_correct / len(labels) if len(labels) > 0 else 0

                # 更新验证进度条
                val_bar.set_postfix({
                    "loss": f"{val_loss:.4f}",
                    "acc": f"{batch_accuracy:.4f}"
                })
                val_bar.update(1)

        ###################################################################################
        #验证循环结束

        # 循环结束后连接所有结果，计算指标，保存模型
        ###################################################################################
        # 循环结束后连接所有结果
        if all_probs:
            all_probs = torch.cat(all_probs).flatten().numpy()
            all_labels = torch.cat(all_labels).numpy()
            all_preds = torch.cat(all_preds).flatten().numpy()
        else:
            all_probs = np.array([])
            all_labels = np.array([])
            all_preds = np.array([])

        # 计算验证指标
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0

        # 检查是否有数据
        if len(all_labels) > 0:
            # 计算指标时会生成混淆矩阵
            metrics = calculate_metrics(all_labels, all_preds, all_probs)

            # 从指标中获取混淆矩阵
            cm = metrics['confusion_matrix']

            # 确保混淆矩阵是有效的2x2矩阵
            if cm.size == 4 and cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                cm_valid = True
            else:
                cm_valid = False
                tn, fp, fn, tp = 0, 0, 0, 0
                print(f"⚠️ 混淆矩阵格式异常: {cm.shape}")
        else:
            print("⚠️ 警告: 验证集无数据，使用默认指标")
            metrics = {
                'f1': 0,
                'roc_auc': 0.5,
                'pr_auc': 0,
                'confusion_matrix': np.array([[0, 0], [0, 0]])
            }
            tn, fp, fn, tp = 0, 0, 0, 0
            cm_valid = False
        # 计算FPR和Recall
        fpr = fp / (fp + tn) if cm_valid and (fp + tn) > 0 else 0.0
        recall = tp / (tp + fn) if cm_valid and (tp + fn) > 0 else 0.0
        # 打印验证集统计信息（只打印一次）
        print(f"混淆矩阵详细: TP={tp}, FP={fp}, TN={tn}, FN={fn} | FPR={fpr:.4f}, Recall={recall:.4f}")
        if len(all_labels) > 0:
            print(f"\n验证集统计:")
            print(f"样本数量: {len(all_labels)}")
            print(f"正样本数量: {np.sum(all_labels)}")
            print(f"负样本数量: {len(all_labels) - np.sum(all_labels)}")
            print(f"预测正样本数量: {tp + fp}")
            print(f"预测正确的正样本数量 (TP): {tp}")
            print(f"预测错误的正样本数量 (FP): {fp}")
            print(f"预测错误的负样本数量 (FN): {fn}")
            print(f"预测正确的负样本数量 (TN): {tn}")

        # 打印关键指标
        if 'f1' in metrics:
            print(
                f"验证指标 - F1: {metrics['f1']:.4f}, AUC: {metrics['roc_auc']:.4f}, PR-AUC: {metrics['pr_auc']:.4f}")
        else:
            print("⚠️ 警告: 无法获取验证指标")

        # 记录指标
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        if 'f1' in metrics:
            writer.add_scalar('F1/val', metrics['f1'], epoch)
        if 'roc_auc' in metrics:
            writer.add_scalar('AUC_ROC/val', metrics['roc_auc'], epoch)
            writer.add_scalar('AUC_PR/val', metrics['pr_auc'], epoch)

        # 保存最佳模型 - 基于F1值
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_threshold = 0.5
            # 保存模型为 best_model.pth
            best_model_path = os.path.join(best_model_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"保存最佳模型，F1: {best_f1:.4f}，阈值: {best_threshold:.4f}")

        # 打印epoch结果
        print(
            f"Epoch {epoch + 1}/{config['training']['epochs']}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Val Metrics - F1: {metrics['f1']:.4f}, AUC: {metrics['roc_auc']:.4f}, PR-AUC: {metrics['pr_auc']:.4f}")

        #更新学习率
        scheduler.step(metrics['f1'])  # 根据F1值更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1} - 学习率: {current_lr:.7f}")

        # 早停检查
        early_stopping(metrics['f1'])
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break
        ###################################################################################

    # 保存最终模型
    final_model_path = os.path.join(config_save_dir_full, "checkpoints", "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"✅ 保存最终模型: {final_model_path}")

    # 保存最佳模型的阈值
    threshold_path = os.path.join(config_save_dir_full, "best_threshold.txt")
    with open(threshold_path, 'w') as f:
        f.write(f"{best_threshold}")
    print(f"最佳阈值 {best_threshold} 已保存至 {threshold_path}")


    # 关闭TensorBoard
    writer.close()

    # 保存验证集可视化结果
    plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['roc_auc'],
                   os.path.join(config_save_dir_full, "visualizations", "roc_curve.png"))
    plot_pr_curve(metrics['precision_curve'], metrics['recall_curve'], metrics['pr_auc'],
                  os.path.join(config_save_dir_full, "visualizations", "pr_curve.png"))
    plot_confusion_matrix(metrics['confusion_matrix'],
                          os.path.join(config_save_dir_full, "visualizations", "confusion_matrix.png"))

    # 保存评估结果
    with open(os.path.join(config_save_dir_full, "val_metrics.txt"), 'w') as f:
        f.write(f"Validation Metrics:\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n")
        f.write(f"Specificity: {metrics['specificity']:.4f}\n")
        f.write(f"MCC: {metrics['mcc']:.4f}\n")
        f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n")
        f.write(f"PR AUC: {metrics['pr_auc']:.4f}\n")

    # 打印完成信息
    print(f"Training completed. Results saved to:")
    print(f"1. Training directory: {config_save_dir_full}")
    print(f"2. Best model directory: {best_model_dir}")

    # ===================================================

    # 属性分析模块 (训练时)_重新构建进化特征处理器并设置到模型中_保持训练-分析一致性
    # 工作流程总结
    # 训练阶段：
    # 构建进化处理器 → 设置到模型 → 训练模型 → 保存模型权重
    # 分析阶段：
    # 创建新模型实例 → 加载权重 → 重建进化处理器 → 设置到模型 → 进行分析
    # 这段代码是连接训练和分析阶段的关键桥梁，确保模型在分析时能正确重现训练时的行为特征。
    # ===================================================
    if config.get('analysis', {}).get('enable', False):
        print(f"[{time.strftime('%H:%M:%S')}] 开始属性分析...")
        analysis_dir = os.path.join(config_save_dir_full, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        # 确保模型是最佳状态
        model = AMPPredictionModel(config['model']).to(device)
        best_model_path = os.path.join(best_model_dir, "best_model.pth")
        try:
            model.load_state_dict(torch.load(best_model_path))
            print(f"成功加载最佳模型: {best_model_path}")
        except:
            # 如果加载失败，尝试兼容性加载
            model.load_state_dict(torch.load(best_model_path), strict=False)
            print(f"兼容性模式加载最佳模型: {best_model_path}")


        model.to(device)
        model.eval()

        # 创建验证集数据加载器
        val_loader = create_data_loader(
            sequences=val_sequences,
            labels=val_labels,
            properties=val_properties,
            config=config,
            shuffle=False
        )

        # 计算属性重要性 - 直接使用验证集数据
        print(f"[{time.strftime('%H:%M:%S')}] 计算属性特征重要性...")
        importance_scores = calculate_property_importance(model, val_loader, device)

        # 保存和可视化重要性结果
        importance_csv = os.path.join(analysis_dir, "property_importance.csv")
        pd.DataFrame({
            "Property": PROPERTY_NAMES,
            "Importance": importance_scores
        }).to_csv(importance_csv, index=False)

        importance_img = os.path.join(analysis_dir, "property_importance.png")
        visualize_property_importance(importance_scores, PROPERTY_NAMES, importance_img)
        print(f"属性重要性结果保存至: {importance_csv} 和 {importance_img}")

        # 计算AMP/非AMP属性差异
        amp_indices = np.where(val_labels == 1)[0]
        nonamp_indices = np.where(val_labels == 0)[0]

        amp_properties = np.array([val_properties[i] for i in amp_indices])
        nonamp_properties = np.array([val_properties[i] for i in nonamp_indices])

        # 计算平均属性值
        amp_means = np.mean(amp_properties, axis=(0, 1))
        nonamp_means = np.mean(nonamp_properties, axis=(0, 1))

        diff_img, diff_csv = compare_amp_nonamp_properties(
            amp_means, nonamp_means, PROPERTY_NAMES, analysis_dir
        )
        print(f"属性差异分析结果保存至: {diff_csv} 和 {diff_img}")

        # ===================================================
        # 新增：序列位置影响评估
        # ===================================================
        if config.get('analysis', {}).get('positional_impact', False):
            print(f"[{time.strftime('%H:%M:%S')}] 开始序列位置影响分析...")
            positional_dir = os.path.join(analysis_dir, "positional_impact")
            os.makedirs(positional_dir, exist_ok=True)

            # 分析序列位置对属性的影响
            positional_results = analyze_sequence_position_impact(
                model,
                val_loader,
                device,
                max_seq_len=config['data']['max_seq_len'],
                save_dir=positional_dir
            )

            print(f"序列位置影响分析完成，结果保存至: {positional_dir}")


        # ===================================================
        # 新增：多肽序列贡献分析
        # ===================================================
        if config.get('analysis', {}).get('peptide_contribution', False):
            print(f"[{time.strftime('%H:%M:%S')}] 开始多肽序列贡献分析...")
            peptide_dir = os.path.join(analysis_dir, "peptide_contribution")
            os.makedirs(peptide_dir, exist_ok=True)

            # 分析多肽序列整体贡献
            peptide_results = analyze_peptide_contribution(
                model,
                val_loader,
                device,
                tokenizer=tokenizer,
                save_dir=peptide_dir,
                num_samples=30
            )

            print(f"多肽序列贡献分析完成，分析了 {peptide_results['num_analyzed']} 个样本，结果保存至: {peptide_dir}")

        # ===================================================
        # 新增：位置协同效应分析
        # ===================================================
        if config.get('analysis', {}).get('position_interaction', False):
            print(f"[{time.strftime('%H:%M:%S')}] 开始位置协同效应分析...")
            interaction_dir = os.path.join(analysis_dir, "position_interaction")
            os.makedirs(interaction_dir, exist_ok=True)

            # 分析位置协同效应
            interaction_results = analyze_position_interactions(
                model,
                val_loader,
                device,
                tokenizer=tokenizer,
                save_dir=interaction_dir,
                num_samples=5
            )

            print(
                f"位置协同效应分析完成，分析了 {interaction_results['num_analyzed']} 个样本，结果保存至: {interaction_dir}")

        # ===================================================
        # SHAP 分析 (修复版本)
        # ===================================================
        shap_dir = os.path.join(analysis_dir, "shap_analysis")
        os.makedirs(shap_dir, exist_ok=True)
        # 创建默认的SHAP数据加载器（即使跳过分析也需要）
        shap_loader = create_data_loader(
            sequences=val_sequences,
            labels=val_labels,
            properties=val_properties,
            config=config,
            shuffle=False
        )
        # 确保background_loader有默认值
        background_loader = None


        if config.get('analysis', {}).get('shap_analysis', False):
            print(f"[{time.strftime('%H:%M:%S')}] 开始 SHAP 分析")
            tokenizer = config['data']['tokenizer']
            try:
                # 获取样本数据
                test_seq_str = val_sequences[0]  # 序列字符串
                test_prop = val_properties[0]  # 属性矩阵
                # 使用tokenizer将序列转换为ID
                test_seq_ids = tokenizer(test_seq_str)
                # 处理序列长度(填充或截断)
                max_len = config['data']['max_seq_len']
                if len(test_seq_ids) < max_len:
                    pad_id = tokenizer.vocab['-']
                    test_seq_ids += [pad_id] * (max_len - len(test_seq_ids))
                else:
                    test_seq_ids = test_seq_ids[:max_len]
                # 创建输入张量
                test_seq_tensor = torch.tensor([test_seq_ids], dtype=torch.long).to(device)
                # 确保属性矩阵维度正确 [batch_size=1, seq_len, num_features]
                if len(test_prop.shape) == 2:  # (seq_len, num_features)
                    test_prop = test_prop.reshape(1, max_len, -1)
                elif len(test_prop.shape) == 3 and test_prop.shape[0] != 1:
                    test_prop = test_prop[:1]  # 取第一个样本
                test_prop_tensor = torch.tensor(test_prop, dtype=torch.float32).to(device)
                # 原始输入预测
                model.eval()
                with torch.no_grad():
                    orig_output = model(
                        test_seq_tensor,
                        test_prop_tensor,
                        [test_seq_str]  # 原始序列字符串
                    )

                orig_prob = torch.sigmoid(orig_output).item()
                print(f"原始预测概率: {orig_prob:.4f}")
                # 创建全零属性矩阵 (扰动输入)
                mutated_prop = np.zeros_like(test_prop)
                mutated_prop_tensor = torch.tensor(mutated_prop, dtype=torch.float32).to(device)
                # 扰动输入预测 - 添加原始序列字符串
                with torch.no_grad():
                    mutated_output = model(
                        test_seq_tensor,
                        mutated_prop_tensor,
                        [test_seq_str]  # 添加原始序列字符串
                    )

                    mutated_prob = torch.sigmoid(mutated_output).item()
                print(f"扰动后预测概率: {mutated_prob:.4f}")
                print(f"概率差异: {abs(orig_prob - mutated_prob):.4f}")

                # 创建SHAP数据加载器
                shap_loader = create_data_loader(
                    sequences=val_sequences,
                    labels=val_labels,
                    properties=val_properties,
                    config=config,
                    shuffle=False
                )

                # 根据敏感性决定是否执行SHAP分析
                if abs(orig_prob - mutated_prob) >= 0.01:

                    print(f"使用 {len(val_sequences)} 个验证样本进行SHAP分析")

                    # 创建背景数据集(随机采样)
                    background_size = min(50, len(val_sequences))
                    background_indices = np.random.choice(len(val_sequences), background_size, replace=False)
                    background_loader = create_data_loader(
                        sequences=[val_sequences[i] for i in background_indices],
                        labels=[val_labels[i] for i in background_indices],
                        properties=[val_properties[i] for i in background_indices],
                        config=config,
                        shuffle=False
                    )
                    print(f"创建了 {background_size} 个样本的背景数据集")

                try:
                    # 使用SHAP库生成解释
                    print(f"使用SHAP库生成解释...")
                    shap_values, base_value, data_df, feature_names, instance_names, labels = generate_shap_explanation(
                        model,
                        shap_loader,
                        device,
                        config,
                        background_loader=background_loader,
                        n_samples=500
                    )
                    # 验证SHAP值质量
                    print(f"SHAP值形状: {shap_values.shape}")
                    shap_mean = np.abs(shap_values).mean()
                    shap_max = np.abs(shap_values).max()
                    print(f"平均|SHAP|值: {shap_mean:.5f}, 最大|SHAP|值: {shap_max:.5f}")

                    # 可视化SHAP摘要
                    print("生成SHAP摘要图...")
                    plt.figure(figsize=(12, 8))
                    shap.summary_plot(
                        shap_values,
                        data_df,  # 修复：使用data_df
                        feature_names=feature_names,
                        show=False
                    )
                    plt.savefig(os.path.join(shap_dir, "summary_plot.png"))
                    plt.close()
                    # 为示例生成详细解释
                    print(f"生成详细解释图...")
                    for i in range(min(11, len(shap_values))):
                        plt.figure(figsize=(12, 6))
                        shap.waterfall_plot(
                            shap.Explanation(
                                values=shap_values[i],
                                base_values=base_value,
                                data=data_df.iloc[i],  # 修复：使用data_df
                                feature_names=feature_names
                            ),
                            show=False
                        )
                        plt.tight_layout()
                        plt.savefig(os.path.join(shap_dir, f"waterfall_{i}_{instance_names[i]}.png"))
                        plt.close()

                    print(f"SHAP分析完成，结果保存至: {shap_dir}")

                except Exception as e:
                    print(f"生成SHAP值时出错: {str(e)}")
                    import traceback
                    traceback.print_exc()

                else:
                    print("⚠️ 警告: 模型对特征变化不敏感，跳过SHAP分析")
                    print("建议执行以下改进:")
                    print("1. 检查属性特征是否标准化/规范化")
                    print("2. 增加属性特征的权重比例")
                    print("3. 在损失函数中添加特征重要性正则化: loss = main_loss + 0.1 * |d(output)/d(features)|")

                    # 创建模型诊断目录
                    diagnostics_dir = os.path.join(analysis_dir, "model_diagnostics")
                    os.makedirs(diagnostics_dir, exist_ok=True)

                    # 模型诊断
                    print("\n运行模型诊断:")
                    try:
                        diagnose_model_sensitivity(
                            model,
                            tokenizer,
                            val_loader,
                            device,
                            config,
                            save_dir=diagnostics_dir  # 添加save_dir参数
                        )
                    except Exception as e:
                        print(f"模型诊断失败: {str(e)}")
                        import traceback
                        traceback.print_exc()

                    # 手动SHAP方法备选方案
                    if 'manual_get_shap_explanation' in globals():
                        print("尝试手动生成SHAP解释...")
                        try:
                            # 这种情况，使用之前创建的shap_loader
                            shap_values = manual_get_shap_explanation(model, shap_loader, device, config)
                            print(f"手动SHAP值范围: min={shap_values.min():.5f}, max={shap_values.max():.5f}")

                            # 可视化SHAP值
                            visualize_shap_values(
                                shap_values,
                                val_sequences[0],
                                val_properties[0],
                                val_labels[0],
                                tokenizer,
                                PROPERTY_NAMES,
                                shap_dir
                            )
                            # ====== 新增保存功能 ======
                            # 1. 保存原始SHAP值（npy格式）
                            np.save(os.path.join(shap_dir, "manual_shap_values.npy"), shap_values)

                            # 2. 保存元数据（CSV格式）
                            sample_metadata = {
                                "sequence": [val_sequences[0]],
                                "label": [val_labels[0]],
                                "mean_shap": [np.mean(shap_values)],
                                "max_shap": [np.max(shap_values)]
                            }
                            pd.DataFrame(sample_metadata).to_csv(
                                os.path.join(shap_dir, "manual_shap_metadata.csv"),
                                index=False
                            )

                            # 3. 保存属性贡献度（CSV格式）
                            property_contrib = pd.DataFrame({
                                "property": PROPERTY_NAMES,
                                "mean_contribution": np.mean(shap_values, axis=(0, 1))  # 跨序列位置的均值
                            })
                            property_contrib.to_csv(
                                os.path.join(shap_dir, "manual_property_contributions.csv"),
                                index=False
                            )
                            # ====== 保存完成 ======

                            print(f"使用手动方法完成SHAP分析")

                        except Exception as me:
                            print(f"手动计算SHAP值失败: {str(me)}")
                            # 保存错误日志
                            with open(os.path.join(shap_dir, "manual_shap_error.log"), 'w') as f:
                                f.write(f"Error: {str(me)}\n")
                                import traceback
                                traceback.print_exc(file=f)
                    else:
                        print("找不到手动SHAP计算方法")
                        # 保存状态记录
                        with open(os.path.join(shap_dir, "manual_shap_status.log"), 'w') as f:
                            f.write("manual_get_shap_explanation函数未定义\n")

            except Exception as e:
                print(f"在整个SHAP分析过程中出错: {str(e)}")
                import traceback
                traceback.print_exc()
                print("继续执行其他分析步骤...")
        # ===================================================

        # 特征剔除机制
        importance_threshold = config.get('analysis', {}).get('importance_threshold', 0.05)
        important_properties = [PROPERTY_NAMES[i] for i, score in enumerate(importance_scores)
                                if score > importance_threshold]
        print(f"\n特征剔除结果 (阈值={importance_threshold}):")
        print(f"原始属性数量: {len(PROPERTY_NAMES)}")
        print(f"保留属性数量: {len(important_properties)}")
        print("保留属性: " + ", ".join(important_properties))
        # 保存精简模型配置
        reduced_config = config.copy()
        reduced_config['model']['important_properties'] = important_properties
        with open(os.path.join(analysis_dir, "reduced_config.yaml"), 'w') as f:
            yaml.dump(reduced_config, f)
        print(f"训练完成。结果保存至: {config_save_dir_full}")
    return best_f1
if __name__ == "__main__":
    # 加载配置文件
    with open("configs/default.yaml", 'r', encoding='utf-8', errors='ignore') as f:
        config = yaml.safe_load(f)

    # 添加缺失配置键的默认值
    # 设置默认值 - 优化假阳性控制
    config.setdefault('show_progress', True)
    config['data'].setdefault('augment', True)
    config['data'].setdefault('batch_size', 64)
    config['data'].setdefault('num_workers', 4)
    config['training'].setdefault('patience', 15)  # 增加耐心值
    config['training'].setdefault('delta', 0.001)
    config['training'].setdefault('log_interval', 50)
    config['training'].setdefault('focus_fpr', True)  # 启用假阳性优化
    config['training'].setdefault('negative_weight', 1)  # 增加负样本权重
    config['training'].setdefault('fpr_target', 0.05)  # 更严格的目标假阳性率

    # 添加属性分析配置
    config.setdefault('analysis', {})
    config['analysis'].setdefault('enable', True)
    config['analysis'].setdefault('positional_impact', True)  # 启用序列位置影响分析
    config['analysis'].setdefault('importance_threshold', 0.05)

    # 确保数值类型的配置项是浮点数（不是字符串）
    config['training']['learning_rate'] = float(config['training']['learning_rate'])
    config['training']['weight_decay'] = float(config['training']['weight_decay'])
    config['training']['delta'] = float(config['training']['delta'])

    # 添加默认值（如果这些键不存在）
    config['training'].setdefault('learning_rate', 0.001)
    config['training'].setdefault('weight_decay', 0.0001)
    config['training'].setdefault('delta', 0.001)
    config['training'].setdefault('val_interval', 1)  # 默认每个epoch都验证

    # 开始训练
    train_model(config)