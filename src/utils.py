import os
import time

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ===================================================
# 工具函数和常量
# ===================================================

# 进度条工具函数
def progress_bar(iterable, desc=None, total=None, unit="it", position=0, leave=True):
    """创建自定义进度条"""
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        position=position,
        leave=leave,
        ascii=True,
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )


# 添加时间估算装饰器
def time_estimator(func):
    """估算函数执行时间的装饰器"""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Starting {func.__name__}...")
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"[{time.strftime('%H:%M:%S')}] Completed {func.__name__} in {elapsed:.2f} seconds")
        return result

    return wrapper


# 氨基酸物化性质字典
AA_PROPERTIES = {
    'A': {'index': 0, 'charge': 0, 'hydrophobicity': 1.8, 'pI': 6.02, 'mw': 89.09, 'solubility': 167.0, 'vdw_volume': 67, 'asa': 113,
          'fold_deltaG': 0.8, 'P_alpha': 142, 'P_beta': 83, 'P_turn': 66},
    'R': {'index': 1, 'charge': 1, 'hydrophobicity': -4.5, 'pI': 10.76, 'mw': 174.20, 'solubility': 150.0, 'vdw_volume': 148,
          'asa': 241, 'fold_deltaG': -0.9, 'P_alpha': 98, 'P_beta': 93, 'P_turn': 95},
    'N': {'index': 2, 'charge': 0, 'hydrophobicity': -3.5, 'pI': 5.41, 'mw': 132.12, 'solubility': 25.0, 'vdw_volume': 96,
          'asa': 135, 'fold_deltaG': -1.2, 'P_alpha': 67, 'P_beta': 89, 'P_turn': 156},
    'D': {'index': 3, 'charge': -1, 'hydrophobicity': -3.5, 'pI': 2.77, 'mw': 133.10, 'solubility': 5.0, 'vdw_volume': 91, 'asa': 150,
          'fold_deltaG': -1.6, 'P_alpha': 151, 'P_beta': 37, 'P_turn': 74},
    'C': {'index': 4, 'charge': 0, 'hydrophobicity': 2.5, 'pI': 5.07, 'mw': 121.16, 'solubility': 280.0, 'vdw_volume': 86,
          'asa': 135, 'fold_deltaG': -1.0, 'P_alpha': 70, 'P_beta': 119, 'P_turn': 119},
    'Q': {'index': 5, 'charge': 0, 'hydrophobicity': -3.5, 'pI': 5.65, 'mw': 146.15, 'solubility': 35.0, 'vdw_volume': 114,
          'asa': 160, 'fold_deltaG': -1.1, 'P_alpha': 111, 'P_beta': 110, 'P_turn': 98},
    'E': {'index': 6, 'charge': -1, 'hydrophobicity': -3.5, 'pI': 3.22, 'mw': 147.13, 'solubility': 8.5, 'vdw_volume': 109,
          'asa': 180, 'fold_deltaG': -1.8, 'P_alpha': 151, 'P_beta': 37, 'P_turn': 74},
    'G': {'index': 7, 'charge': 0, 'hydrophobicity': -0.4, 'pI': 5.97, 'mw': 75.07, 'solubility': 250.0, 'vdw_volume': 48, 'asa': 85,
          'fold_deltaG': 1.0, 'P_alpha': 57, 'P_beta': 75, 'P_turn': 156},
    'H': {'index': 8, 'charge': 0.5, 'hydrophobicity': -3.2, 'pI': 7.59, 'mw': 155.16, 'solubility': 41.0, 'vdw_volume': 118,
          'asa': 185, 'fold_deltaG': -0.5, 'P_alpha': 100, 'P_beta': 87, 'P_turn': 95},  # 组氨酸在生理pH下部分带正电
    'I': {'index': 9, 'charge': 0, 'hydrophobicity': 4.5, 'pI': 6.05, 'mw': 131.18, 'solubility': 34.0, 'vdw_volume': 124,
          'asa': 170, 'fold_deltaG': -3.1, 'P_alpha': 108, 'P_beta': 160, 'P_turn': 47},
    'L': {'index': 10, 'charge': 0, 'hydrophobicity': 3.8, 'pI': 5.95, 'mw': 131.18, 'solubility': 24.0, 'vdw_volume': 124,
          'asa': 170, 'fold_deltaG': -3.1, 'P_alpha': 121, 'P_beta': 130, 'P_turn': 59},
    'K': {'index': 11, 'charge': 1, 'hydrophobicity': -3.9, 'pI': 9.74, 'mw': 146.19, 'solubility': 550.0, 'vdw_volume': 135,
          'asa': 211, 'fold_deltaG': -2.5, 'P_alpha': 129, 'P_beta': 74, 'P_turn': 98},
    'M': {'index': 12, 'charge': 0, 'hydrophobicity': 1.9, 'pI': 5.74, 'mw': 149.21, 'solubility': 56.0, 'vdw_volume': 124,
          'asa': 185, 'fold_deltaG': -1.9, 'P_alpha': 113, 'P_beta': 104, 'P_turn': 120},
    'F': {'index': 13, 'charge': 0, 'hydrophobicity': 2.8, 'pI': 5.48, 'mw': 165.19, 'solubility': 29.0, 'vdw_volume': 135,
          'asa': 200, 'fold_deltaG': -1.9, 'P_alpha': 111, 'P_beta': 138, 'P_turn': 60},
    'P': {'index': 14, 'charge': 0, 'hydrophobicity': -1.6, 'pI': 6.30, 'mw': 115.13, 'solubility': 162.0, 'vdw_volume': 90,
          'asa': 140, 'fold_deltaG': -0.1, 'P_alpha': 57, 'P_beta': 55, 'P_turn': 152},
    'S': {'index': 15, 'charge': 0, 'hydrophobicity': -0.8, 'pI': 5.68, 'mw': 105.09, 'solubility': 360.0, 'vdw_volume': 73,
          'asa': 102, 'fold_deltaG': -0.5, 'P_alpha': 120, 'P_beta': 75, 'P_turn': 143},
    'T': {'index': 16, 'charge': 0, 'hydrophobicity': -0.7, 'pI': 5.60, 'mw': 119.12, 'solubility': 100.0, 'vdw_volume': 93,
          'asa': 138, 'fold_deltaG': -1.2, 'P_alpha': 97, 'P_beta': 75, 'P_turn': 143},
    'W': {'index': 17, 'charge': 0, 'hydrophobicity': -0.9, 'pI': 5.88, 'mw': 204.23, 'solubility': 11.4, 'vdw_volume': 163,
          'asa': 255, 'fold_deltaG': -1.2, 'P_alpha': 137, 'P_beta': 108, 'P_turn': 96},
    'Y': {'index': 18, 'charge': 0, 'hydrophobicity': -1.3, 'pI': 5.66, 'mw': 181.19, 'solubility': 0.46, 'vdw_volume': 141,
          'asa': 210, 'fold_deltaG': -1.5, 'P_alpha': 138, 'P_beta': 108, 'P_turn': 114},
    'V': {'index': 19, 'charge': 0, 'hydrophobicity': 4.2, 'pI': 5.96, 'mw': 117.15, 'solubility': 58.0, 'vdw_volume': 105,
          'asa': 155, 'fold_deltaG': -2.9, 'P_alpha': 106, 'P_beta': 170, 'P_turn': 50},
    '-': {'index': 20, 'charge': 0, 'hydrophobicity': 0.0, 'pI': 0.0, 'mw': 0.0, 'solubility': 0.0, 'vdw_volume': 0, 'asa': 0,
          'fold_deltaG': 0.0, 'P_alpha': 0, 'P_beta': 0, 'P_turn': 0}
}

# 性质归一化范围
PROPERTY_RANGES = {
    'hydrophobicity': (-4.5, 4.5),
    'charge': (-1, 1),  # 电荷范围：-1到1（因为单个氨基酸电荷在-1到1之间）
    'pI': (0.0, 11.0),
    'mw': (75.07, 204.23),
    'solubility': (0.0, 550.0),
    'vdw_volume': (0, 200),
    'asa': (0, 300),
    'fold_deltaG': (-3.1, 1.0),
    'P_alpha': (0, 200),
    'P_beta': (0, 200),
    'P_turn': (0, 200)
}


# 11维性质名称列表
PROPERTY_NAMES = list(PROPERTY_RANGES.keys())

def generate_sequence_indices(sequences, labels):
    """生成完整的序列索引数组 - 修复索引分配问题"""
    return np.arange(len(sequences))

def normalize_value(value, min_val, max_val):
    """将值归一化到[0,1]范围"""
    return (value - min_val) / (max_val - min_val) if max_val > min_val else 0.0


def denormalize_value(norm_value, min_val, max_val):
    """将归一化值还原"""
    return norm_value * (max_val - min_val) + min_val


def calculate_metrics(y_true, y_pred, y_prob):
    """计算评估指标"""
    """计算各种评估指标，包括AUC、F1等"""
    # 确保输入是numpy数组
    y_true = np.asarray(y_true)
    y_pred_bin = np.asarray(y_pred)  # 使用传入的y_pred参数
    y_pred_prob = np.asarray(y_prob)  # 使用传入的y_prob参数

    # 处理空数据情况
    if len(y_true) == 0 or len(y_pred_prob) == 0:
        return {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'specificity': 0,
            'mcc': 0,
            'roc_auc': 0,
            'pr_auc': 0,
            'confusion_matrix': np.array([[0, 0], [0, 0]]),
            'tpr': 0,
            'fpr': 0,
            'precision_curve': np.array([]),
            'recall_curve': np.array([]),
            'fpr_curve': np.array([]),
            'tpr_curve': np.array([])
        }

    # 确保y_pred_prob是1D数组
    if y_pred_prob.ndim > 1:
        y_pred_prob = y_pred_prob.flatten()

    # 确保y_pred_bin是整数类型
    y_pred_bin = y_pred_bin.astype(int)

    # 计算混淆矩阵
    try:
        cm = confusion_matrix(y_true, y_pred_bin)
        tn, fp, fn, tp = cm.ravel()
    except ValueError:
        # 处理特殊情况（如所有预测都是同一类）
        cm = np.array([[0, 0], [0, 0]])
        tn, fp, fn, tp = 0, 0, 0, 0

    # 计算基本指标
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (
                tn + fp) * (tn + fn) > 0 else 0

    # 计算ROC曲线和AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # 计算PR曲线和AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = auc(recall_curve, precision_curve)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'mcc': mcc,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm,
        'tpr': tpr,
        'fpr': fpr,
        'precision_curve': precision_curve,
        'recall_curve': recall_curve
    }


def plot_roc_curve(fpr, tpr, roc_auc, save_path):
    """绘制ROC曲线"""
    plt.figure()

    # 检查是否有有效数据
    if len(fpr) > 0 and len(tpr) > 0:
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    else:
        plt.plot([0, 1], [0, 1], color='red', lw=2, label='No ROC curve (single class)')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()


def plot_pr_curve(precision, recall, pr_auc, save_path):
    """绘制PR曲线"""
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(cm, save_path):
    """绘制混淆矩阵"""
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-AMP', 'AMP'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()


def calculate_property_importance(model, data_loader, device):
    """
    计算每个属性在预测中的平均贡献度
    """
    model.eval()
    total_importance = np.zeros(len(PROPERTY_NAMES))
    total_samples = 0

    # 添加空批次检查
    empty_batch_count = 0

    for batch_idx, (seq, labels, props) in enumerate(data_loader):
        batch_size = seq.size(0)

        # 跳过空批次
        if batch_size == 0:
            empty_batch_count += 1
            print(f"⚠️ 警告: 批次 {batch_idx} 为空，跳过")
            continue

        # 将数据移到设备
        seq, props = seq.to(device), props.to(device)
        props.requires_grad = True

        # 前向传播
        outputs = model(seq, props)

        # 创建与输出相同形状的梯度张量
        gradients = torch.zeros_like(outputs)

        # 安全设置梯度值
        if gradients.dim() == 2:
            gradients[:, 0] = 1.0  # 设置梯度为1，用于计算对输出的影响
        else:
            gradients.fill_(1.0)

        # 反向传播计算梯度
        outputs.backward(gradient=gradients)

        # 计算每个属性的平均梯度绝对值
        if props.grad is not None and not torch.all(torch.isnan(props.grad)):
            batch_importance = torch.mean(torch.abs(props.grad), dim=(0, 1)).cpu().numpy()
        else:
            batch_importance = np.zeros(len(PROPERTY_NAMES))
            print(f"⚠️ 警告: 批次 {batch_idx} 梯度无效")

        total_importance += batch_importance * batch_size
        total_samples += batch_size

        # 清除梯度
        props.grad = None

    # 计算平均重要性
    if total_samples > 0:
        avg_importance = total_importance / total_samples
    else:
        avg_importance = np.zeros(len(PROPERTY_NAMES))
        print("⚠️ 警告: 没有样本用于计算属性重要性")

    if empty_batch_count > 0:
        print(f"⚠️ 警告: 跳过了 {empty_batch_count} 个空批次")

    return avg_importance


def visualize_property_importance(importance_scores, property_names, save_path):
    """
    可视化属性重要性
    """
    plt.figure(figsize=(12, 8))
    sorted_indices = np.argsort(importance_scores)[::-1]
    sorted_scores = importance_scores[sorted_indices]
    sorted_names = [property_names[i] for i in sorted_indices]

    # 创建条形图
    bars = plt.bar(range(len(sorted_scores)), sorted_scores, color='skyblue')
    plt.xticks(range(len(sorted_scores)), sorted_names, rotation=45, ha='right')
    plt.ylabel('Importance score')
    plt.title('Importance of Attributes in AMP Prediction')
    plt.tight_layout()

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    plt.savefig(save_path)
    plt.close()
    print(f"属性重要性可视化已保存至: {save_path}")


def compare_amp_nonamp_properties(amp_means, nonamp_means, property_names, save_dir):
    """
    对比分析抗菌肽和非抗菌肽的关键属性差异
    """
    os.makedirs(save_dir, exist_ok=True)

    # 计算差异
    differences = amp_means - nonamp_means
    abs_differences = np.abs(differences)

    # 可视化差异
    plt.figure(figsize=(14, 8))
    sorted_indices = np.argsort(abs_differences)[::-1]
    sorted_diffs = differences[sorted_indices]
    sorted_names = [property_names[i] for i in sorted_indices]

    # 创建条形图 (绿色表示AMP>非AMP，红色表示相反)
    colors = ['green' if diff > 0 else 'red' for diff in sorted_diffs]
    bars = plt.bar(range(len(sorted_diffs)), sorted_diffs, color=colors)
    plt.xticks(range(len(sorted_diffs)), sorted_names, rotation=45, ha='right')
    plt.ylabel('Difference (AMP - Non_AMP)')
    plt.title('Differences in key properties between AMP and non-AMP')
    plt.tight_layout()

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    diff_path = os.path.join(save_dir, "property_differences.png")
    plt.savefig(diff_path)
    plt.close()

    # 保存统计结果
    stats_path = os.path.join(save_dir, "property_comparison.csv")
    results = []
    for i in range(len(property_names)):
        results.append({
            "Property": property_names[i],
            "AMP_Mean": amp_means[i],
            "NonAMP_Mean": nonamp_means[i],
            "Difference": differences[i],
            "Absolute_Difference": abs_differences[i]
        })

    df = pd.DataFrame(results)
    df.to_csv(stats_path, index=False)

    print(f"属性差异分析结果保存至: {diff_path} 和 {stats_path}")
    return diff_path, stats_path


# ===================================================
# 新增：序列位置影响评估功能
# ===================================================
def calculate_positional_importance(model, data_loader, device, max_seq_len):
    """计算序列中每个位置的重要性"""
    model.eval()
    positional_importance = np.zeros((max_seq_len, len(PROPERTY_NAMES)))
    total_samples = 0
    empty_batch_count = 0
    for batch_idx, batch_data in enumerate(data_loader):
        # 从批次数据中提取
        seq = batch_data['seq_ids'].to(device)
        labels = batch_data['labels'].to(device)
        props = batch_data['properties'].to(device)
        orig_sequences = batch_data['sequence_strs']  # 添加原始序列字符串
        batch_size = seq.size(0)
        if batch_size == 0:
            empty_batch_count += 1
            print(f"⚠️ 警告: 批次 {batch_idx} 为空，跳过")
            continue
        props.requires_grad = True
        # 添加 orig_sequences 参数
        outputs = model(seq, props, orig_sequences)

        # 创建与输出相同形状的梯度张量
        gradients = torch.zeros_like(outputs)
        # 安全设置梯度值
        if gradients.dim() == 2:
            gradients[:, 0] = 1.0
        else:
            gradients.fill_(1.0)
        # 反向传播计算梯度
        outputs.backward(gradient=gradients)
        # 计算每个位置、每个属性的平均梯度绝对值
        if props.grad is not None and not torch.all(torch.isnan(props.grad)):
            batch_importance = torch.mean(torch.abs(props.grad), dim=0).cpu().numpy()
        else:
            batch_importance = np.zeros((max_seq_len, len(PROPERTY_NAMES)))
            print(f"⚠️ 警告: 批次 {batch_idx} 梯度无效")
        # 只取有效序列长度部分
        valid_length = min(max_seq_len, batch_importance.shape[0])
        positional_importance[:valid_length] += batch_importance[:valid_length] * batch_size
        total_samples += batch_size
        # 清除梯度
        props.grad = None
    # 计算平均重要性
    if total_samples > 0:
        positional_importance /= total_samples
    else:
        print("⚠️ 警告: 没有样本用于计算位置重要性")
    if empty_batch_count > 0:
        print(f"⚠️ 警告: 跳过了 {empty_batch_count} 个空批次")
    return positional_importance

def visualize_positional_importance(positional_importance, property_names, save_path):
    """
    可视化序列位置重要性
    """
    plt.figure(figsize=(16, 10))

    # 使用热力图展示位置重要性
    sns.heatmap(
        positional_importance.T,
        cmap="viridis",
        xticklabels=10,  # 每10个位置显示一个标签
        yticklabels=property_names,
        cbar_kws={'label': 'Importance Score'}
    )

    plt.xlabel("Position in Sequence")
    plt.ylabel("Property")
    plt.title("Positional Property Importance in Sequence")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"序列位置重要性热力图已保存至: {save_path}")


def analyze_sequence_position_impact(model, data_loader, device, max_seq_len, save_dir):
    """
    分析序列位置对属性的影响
    """
    os.makedirs(save_dir, exist_ok=True)

    # 计算位置重要性
    positional_importance = calculate_positional_importance(model, data_loader, device, max_seq_len)

    # 可视化位置重要性
    heatmap_path = os.path.join(save_dir, "positional_importance_heatmap.png")
    visualize_positional_importance(positional_importance, PROPERTY_NAMES, heatmap_path)

    # 保存位置重要性数据
    csv_path = os.path.join(save_dir, "positional_importance.csv")
    df = pd.DataFrame(
        positional_importance,
        columns=PROPERTY_NAMES,
        index=[f"Pos_{i + 1}" for i in range(positional_importance.shape[0])]
    )
    df.to_csv(csv_path)

    # 分析每个位置最重要的属性
    top_properties_per_position = []
    for i, row in enumerate(positional_importance):
        # 获取当前位置重要性最高的属性
        top_idx = np.argmax(row)
        top_properties_per_position.append({
            "Position": i + 1,
            "Top_Property": PROPERTY_NAMES[top_idx],
            "Importance": row[top_idx]
        })

    top_props_df = pd.DataFrame(top_properties_per_position)
    top_props_path = os.path.join(save_dir, "top_properties_per_position.csv")
    top_props_df.to_csv(top_props_path, index=False)

    # 可视化每个位置最重要的属性
    plt.figure(figsize=(16, 6))
    positions = np.arange(1, len(top_properties_per_position) + 1)
    colors = cm.viridis(np.linspace(0, 1, len(PROPERTY_NAMES)))

    # 创建颜色映射
    property_to_color = {prop: colors[i] for i, prop in enumerate(PROPERTY_NAMES)}

    # 为每个位置绘制条形
    for i, pos_data in enumerate(top_properties_per_position):
        plt.bar(i + 1, pos_data['Importance'],
                color=property_to_color[pos_data['Top_Property']])

    # 添加图例
    legend_handles = []
    for prop, color in property_to_color.items():
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color, label=prop))

    plt.legend(handles=legend_handles, title="Properties", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel("Position in Sequence")
    plt.ylabel("Importance Score")
    plt.title("Most Important Property at Each Position")
    plt.xticks(positions[::5])  # 每5个位置显示一个标签

    bar_chart_path = os.path.join(save_dir, "top_properties_bar_chart.png")
    plt.tight_layout()
    plt.savefig(bar_chart_path)
    plt.close()

    print(f"序列位置影响分析结果保存至: {save_dir}")
    return {
        "heatmap": heatmap_path,
        "csv": csv_path,
        "top_properties": top_props_path,
        "bar_chart": bar_chart_path
    }

# ===================================================
# 新增：多肽序列整体贡献分析
# ===================================================
def analyze_peptide_contribution(model, data_loader, device, tokenizer, save_dir, num_samples=10):
    """
    分析多肽序列整体对模型的贡献
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # 收集结果
    results = []

    # 获取一批数据
    for batch_idx, batch_data in enumerate(data_loader):
        seq = batch_data['seq_ids'].to(device)
        labels = batch_data['labels'].to(device)
        props = batch_data['properties'].to(device)
        orig_sequences = batch_data['sequence_strs']
        # 只分析正样本（AMP）
        amp_indices = torch.where(labels == 1)[0]
        if len(amp_indices) == 0:
            continue

        # 取前几个样本
        for i in amp_indices[:min(num_samples, len(amp_indices))]:
            single_seq = seq[i:i + 1].to(device)
            single_props = props[i:i + 1].to(device)

            # 原始预测
            with torch.no_grad():
                orig_output = model(single_seq, single_props).item()

            # 创建扰动序列
            seq_length = single_seq.size(1)
            contribution_scores = []
            amino_acids = []

            for pos in range(seq_length):
                # 创建扰动序列（将特定位置设为填充token）
                perturbed_seq = single_seq.clone()
                perturbed_seq[0, pos] = tokenizer.vocab['-']  # 使用填充token

                # 获取预测
                with torch.no_grad():
                    perturbed_output = model(perturbed_seq, single_props).item()

                # 计算贡献分数（原始输出与扰动输出的差异）
                contribution = orig_output - perturbed_output
                contribution_scores.append(contribution)

                # 获取氨基酸
                aa_idx = single_seq[0, pos].item()
                amino_acids.append(tokenizer.inverse_vocab.get(aa_idx, 'X'))

            # 保存结果
            peptide_seq = ''.join(amino_acids)
            results.append({
                "sequence": peptide_seq,
                "contribution_scores": contribution_scores,
                "original_score": orig_output,
                "label": labels[i].item()
            })

        # 只分析前num_samples个样本
        if len(results) >= num_samples:
            break

    # 可视化每个序列的贡献
    for i, result in enumerate(results):
        plt.figure(figsize=(16, 4))
        positions = np.arange(1, len(result['contribution_scores']) + 1)
        colors = ['red' if score < 0 else 'green' for score in result['contribution_scores']]

        plt.bar(positions, result['contribution_scores'], color=colors)
        plt.xlabel("Position in Sequence")
        plt.ylabel("Contribution Score")
        plt.title(
            f"Contribution of Each Position in Sequence {i + 1}\n(Original Score: {result['original_score']:.4f})")
        plt.xticks(positions)

        # 在柱子上方添加氨基酸标签
        for j, (pos, score) in enumerate(zip(positions, result['contribution_scores'])):
            plt.text(pos, score + (0.02 if score > 0 else -0.02),
                     result['sequence'][j],
                     ha='center', va='bottom' if score > 0 else 'top')

        plot_path = os.path.join(save_dir, f"peptide_contribution_{i + 1}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        # 保存数据
        data_path = os.path.join(save_dir, f"peptide_contribution_{i + 1}.csv")
        df = pd.DataFrame({
            "Position": positions,
            "Amino_Acid": list(result['sequence']),
            "Contribution": result['contribution_scores']
        })
        df.to_csv(data_path, index=False)

    print(f"多肽序列贡献分析完成，结果保存至: {save_dir}")
    return {
        "num_analyzed": len(results),
        "save_dir": save_dir
    }


def visualize_shap_values(
        shap_values,
        input_seq,
        input_props,
        labels,
        tokenizer,
        property_names,
        save_dir,
        uncertainties=None  # 添加 uncertainties 参数
):
    """
    可视化 SHAP 值

    参数:
        shap_values: 计算得到的SHAP值
        input_seq: 输入序列
        input_props: 输入属性
        labels: 真实标签
        tokenizer: 序列编码器
        property_names: 属性名称列表
        save_dir: 保存目录
        uncertainties: 不确定性值（可选）
    """
    os.makedirs(save_dir, exist_ok=True)

    # 保存 SHAP 值
    np.save(os.path.join(save_dir, "shap_values.npy"), shap_values)
    np.save(os.path.join(save_dir, "input_seq.npy"), input_seq)
    np.save(os.path.join(save_dir, "input_props.npy"), input_props)
    np.save(os.path.join(save_dir, "labels.npy"), labels)

    # 打印形状信息用于调试
    print(f"SHAP values shape: {shap_values.shape}")
    print(f"Input properties shape: {input_props.shape}")
    print(f"Input sequence shape: {input_seq.shape}")

    # 检查并调整SHAP值维度
    if shap_values.ndim == 4:
        # 压缩最后一个维度（如果是1）
        if shap_values.shape[-1] == 1:
            shap_values = shap_values.squeeze(-1)
            print(f"压缩后的SHAP值形状: {shap_values.shape}")

        # 如果SHAP值有12个特征（序列索引+11个属性）
        if shap_values.shape[-1] == 12:
            # 只取属性部分（第1到11个特征）
            shap_values = shap_values[..., 1:]
            print(f"提取属性部分后的SHAP值形状: {shap_values.shape}")

    # 确保SHAP值和输入属性的维度匹配
    if shap_values.shape != input_props.shape:
        print(f"⚠️ 警告: SHAP值形状 {shap_values.shape} 与输入属性形状 {input_props.shape} 不匹配")
        print("尝试在序列维度上平均")

        # 在序列维度上平均
        try:
            mean_shap = np.mean(shap_values, axis=1)
            mean_props = np.mean(input_props, axis=1)
            print(f"平均后的SHAP值形状: {mean_shap.shape}")
            print(f"平均后的属性形状: {mean_props.shape}")
        except Exception as e:
            print(f"平均失败: {e}")
            # 如果无法平均，则只处理前几个样本
            min_samples = min(shap_values.shape[0], input_props.shape[0])
            shap_values = shap_values[:min_samples]
            input_props = input_props[:min_samples]
            input_seq = input_seq[:min_samples]
            labels = labels[:min_samples]
            print(f"只处理前 {min_samples} 个样本")
    else:
        mean_shap = np.mean(shap_values, axis=1)
        mean_props = np.mean(input_props, axis=1)

    # 可视化 Summary Plot
    # plt.figure(figsize=(12, 8))
    try:
        # 打印调试信息
        print(f"平均SHAP值形状: {mean_shap.shape}")
        print(f"平均属性值形状: {mean_props.shape}")
        print(f"平均SHAP值范围: min={np.min(mean_shap):.4f}, max={np.max(mean_shap):.4f}")
        print(f"平均属性值范围: min={np.min(mean_props):.4f}, max={np.max(mean_props):.4f}")

        # 检查SHAP值是否全为零
        if np.all(mean_shap == 0):
            print("⚠️ 警告: 平均SHAP值全为零，无法绘制摘要图")
            raise ValueError("SHAP values are all zero")
        plt.figure(figsize=(12, 8))
        # 尝试使用shap.summary_plot
        shap.summary_plot(mean_shap, mean_props, feature_names=property_names, max_display=14, show=False)

        # 添加标题
        plt.title("SHAP Summary Plot")
        # 保存图形（必须在摘要图之后）
        plt.savefig(os.path.join(save_dir, "shap_summary_plot0.png"))
        plt.close()  # 关闭当前图形

    except Exception as e:
        print(f"无法绘制Summary Plot: {e}")

        # 尝试替代方法1: 使用条形图显示特征重要性
        try:
            plt.figure(figsize=(12, 8))
            print("尝试替代方法: 条形图显示特征重要性")

            # 计算特征重要性（平均绝对SHAP值）
            feature_importance = np.mean(np.abs(mean_shap), axis=0)

            # 保护：检查特征数量是否匹配
            if len(feature_importance) > len(property_names):
                print(f"⚠️ 警告: 特征数量({len(feature_importance)}) > 属性名称数量({len(property_names)})")
                feature_importance = feature_importance[:len(property_names)]

            # 排序特征
            sorted_idx = np.argsort(feature_importance)[::-1]
            sorted_names = [property_names[i] for i in sorted_idx]
            sorted_importance = feature_importance[sorted_idx]

            # 绘制条形图
            plt.barh(range(len(sorted_importance)), sorted_importance, align='center')
            plt.yticks(range(len(sorted_importance)), sorted_names)
            plt.xlabel('Mean |SHAP value|')
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "shap_summary_plot1.png"))
            plt.close()

        except Exception as e2:
            print(f"替代方法1失败: {e2}")

            # 尝试替代方法2: 绘制SHAP值的分布
            try:

                plt.figure(figsize=(12, 8))
                print("尝试替代方法: SHAP值分布图")

                # 展平SHAP值
                flat_shap = mean_shap.flatten()

                # 绘制直方图
                plt.hist(flat_shap, bins=50, alpha=0.7)
                plt.xlabel('SHAP Value')
                plt.ylabel('Frequency')
                plt.title('SHAP Value Distribution')
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(save_dir, "shap_summary_plot2.png"))
                plt.close()
            except Exception as e3:
                print(f"替代方法2失败: {e3}")
                plt.figure(figsize=(12, 8))
                plt.text(0.5, 0.5, "Cannot visualize SHAP values", ha='center', va='center')
                plt.title("SHAP Visualization Failed")
                # 保存图形
                plt.savefig(os.path.join(save_dir, "shap_summary_plot3.png"))
                plt.close()


    # 可视化序列 SHAP 值
    for i in range(min(10, len(labels))):  # 仅可视化前10个样本
        plt.figure(figsize=(16, 6))
        sequence = input_seq[i]

        # 确保序列长度不超过SHAP值长度
        seq_length = min(len(sequence), shap_values.shape[1])
        sequence = sequence[:seq_length]

        # 获取序列部分的 SHAP 值（对属性维度取平均）
        try:
            # 对每个位置的所有属性取平均
            shap_seq_values = np.mean(shap_values[i, :seq_length], axis=1)
        except:
            # 如果失败，只取第一个属性
            shap_seq_values = shap_values[i, :seq_length, 0]

        # 解码序列
        decoded_sequence = []
        for aa_idx in sequence:
            if aa_idx in tokenizer.inverse_vocab:
                decoded_sequence.append(tokenizer.inverse_vocab[aa_idx])
            else:
                decoded_sequence.append('X')

        # 绘制序列 SHAP 值
        positions = np.arange(1, seq_length + 1)

        # 创建基于SHAP值的颜色映射
        norm = plt.Normalize(vmin=np.min(shap_seq_values), vmax=np.max(shap_seq_values))
        colors = plt.cm.viridis(norm(shap_seq_values))

        plt.bar(positions, shap_seq_values, color=colors)  # 使用颜色列表

        plt.xlabel("Position in Sequence")
        plt.ylabel("SHAP Value (Attribute average)")
        plt.title(f"SHAP Values for Sequence {i + 1}\nSequence: {''.join(decoded_sequence)}")
        plt.xticks(positions)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"shap_sequence_plot_{i + 1}.png"))
        plt.close()

    # 如果提供了不确定性数据，创建带有不确定性的可视化
    if uncertainties is not None:
        # 1. 创建带有不确定性的特征重要性图
        plt.figure(figsize=(14, 8))

        # 计算平均绝对SHAP值
        # 使用平均后的SHAP值
        mean_abs_shap = np.mean(np.abs(mean_shap), axis=0)

        # 排序特征重要性
        sorted_idx = np.argsort(mean_abs_shap)[::-1]
        sorted_names = [property_names[i] for i in sorted_idx]
        sorted_importance = mean_abs_shap[sorted_idx]

        # 获取相应特征的不确定性
        sorted_uncertainties = uncertainties[sorted_idx] if len(uncertainties) == len(property_names) else None

        # 创建条形图
        if sorted_uncertainties is not None:
            plt.bar(
                range(len(sorted_importance)),
                sorted_importance,
                yerr=sorted_uncertainties,
                capsize=5,
                alpha=0.7,
                color='skyblue'
            )
            plt.title("Feature Importance with Uncertainty")
        else:
            plt.bar(range(len(sorted_importance)), sorted_importance, color='skyblue')
            plt.title("Feature Importance")

        plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha='right')
        plt.xlabel("Features")
        plt.ylabel("Mean |SHAP|")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "feature_importance_with_uncertainty.png"))
        plt.close()

        # 2. 创建SHAP值与预测值的关系图（带不确定性）
        plt.figure(figsize=(10, 8))

        for i, prop in enumerate(property_names[:10]):  # 只显示前10个属性
            shap_values_prop = mean_shap[:, i]

            # 计算属性值（反归一化）
            min_val, max_val = PROPERTY_RANGES.get(prop, (0, 1))
            prop_values = denormalize_value(mean_props[:, i], min_val, max_val)

            plt.scatter(
                prop_values,
                shap_values_prop,
                alpha=0.5,
                label=prop
            )

        plt.xlabel("Property Value")
        plt.ylabel("SHAP Value")
        plt.title("SHAP Value vs Property Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "shap_vs_property.png"))
        plt.close()

    print(f"SHAP 分析结果保存至: {save_dir}")


def generate_shap_explanation(model, data_loader, device, config, background_loader=None,
                              uncertainty=False, n_samples=30):
    """
    生成SHAP解释（优化版）- 使用属性级聚合和兼容API
    """
    import shap
    import numpy as np
    import torch
    from tqdm import tqdm
    import pandas as pd
    import time
    import warnings
    from joblib import Parallel, delayed

    warnings.filterwarnings("ignore", category=UserWarning)  # 忽略SHAP警告

    print(f"[SHAP] 开始生成解释 | 样本数: {n_samples}")
    start_time = time.time()

    # 确保模型在评估模式
    model.eval()

    # 获取最大序列长度和属性数量
    max_seq_len = config['data']['max_seq_len']
    num_properties = len(PROPERTY_NAMES)

    # 创建背景数据集 - 按属性聚合
    background_data = []
    background_preds = []

    # 从背景加载器中获取样本
    for batch_idx, batch in enumerate(background_loader):
        props = batch['properties'].cpu().numpy()

        # 计算每个样本的属性平均值
        batch_data = np.mean(props, axis=1)  # 沿序列维度平均
        background_data.append(batch_data)

        # 同时计算背景预测值
        with torch.no_grad():
            seq_tensor = batch['seq_ids'].to(device)
            props_tensor = batch['properties'].to(device)
            sequences = batch['sequence_strs']
            outputs = model(seq_tensor, props_tensor, sequences)

            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            background_preds.extend(probs.tolist())

        if len(background_data) >= 3:  # 只取3个批次
            break

    if background_data:
        background_data = np.concatenate(background_data, axis=0)
    else:
        # 创建虚拟背景数据
        background_data = np.zeros((10, num_properties))

    print(f"背景数据集形状: {background_data.shape}")

    # 计算基准值（背景预测的平均值）
    if background_preds:
        base_value = np.mean(background_preds)
    else:
        base_value = 0.5

    # 定义模型预测函数（批量优化版）
    def model_predict_batch(X_batch):
        """批量预测函数，显著提高效率"""
        batch_size = X_batch.shape[0]

        # 创建属性矩阵 - 使用平均值扩展为完整序列
        props_matrix = np.repeat(X_batch[:, np.newaxis, :], max_seq_len, axis=1)

        # 使用序列ID的均值作为占位符
        seq_ids = np.zeros((batch_size, max_seq_len), dtype=np.int64)

        # 转换为张量
        seq_tensor = torch.tensor(seq_ids).to(device)
        props_tensor = torch.tensor(props_matrix).float().to(device)

        # 使用占位符序列字符串
        dummy_sequences = [""] * batch_size

        # 模型预测
        with torch.no_grad():
            outputs = model(seq_tensor, props_tensor, dummy_sequences)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

        return probs

    # 创建解释器 - 使用更兼容的KernelExplainer
    explainer = shap.KernelExplainer(
        model_predict_batch,  # 使用批量预测函数
        background_data,
        link="logit"  # 适用于概率输出
    )

    # 收集所有样本的SHAP值
    all_shap_values = []
    all_data = []
    all_labels = []
    instance_names = []

    # 限制样本数量以提高性能
    max_samples = min(50, len(data_loader.dataset))
    sample_count = 0

    # 收集所有样本数据
    samples_to_process = []
    for batch_idx, batch in enumerate(data_loader):
        if sample_count >= max_samples:
            break

        # 获取当前批次数据
        props = batch['properties']
        labels = batch['labels'].cpu().numpy()
        sequences = batch['sequence_strs']

        for i in range(len(props)):
            if sample_count >= max_samples:
                break

            # 计算属性平均值
            prop_means = props[i].mean(dim=0).cpu().numpy()

            # 收集样本数据
            samples_to_process.append({
                'prop_means': prop_means,
                'label': labels[i],
                'sequence': sequences[i] if i < len(sequences) else ""
            })
            sample_count += 1

    print(f"准备处理 {len(samples_to_process)} 个样本...")

    # 定义处理单个样本的函数（优化版）
    def compute_shap_for_sample(sample, explainer, sample_idx):
        """计算单个样本的SHAP值（优化版）"""
        try:
            # 计算SHAP值 - 减少扰动样本数量
            shap_values = explainer.shap_values(
                sample['prop_means'].reshape(1, -1),
                nsamples=500,  # 大幅减少扰动样本数量
                silent=True  # 禁用SHAP内部日志
            )
            return {
                'shap_values': shap_values,
                'data': sample['prop_means'],
                'label': sample['label'],
                'sequence': sample['sequence'],
                'success': True
            }
        except Exception as e:
            print(f"计算样本 {sample_idx} 的SHAP值时出错: {e}")
            # 创建空结果
            return {
                'shap_values': np.zeros((1, len(PROPERTY_NAMES))),
                'data': sample['prop_means'],
                'label': sample['label'],
                'sequence': sample['sequence'],
                'success': False
            }

    # 使用并行计算处理所有样本
    n_jobs = min(1, os.cpu_count())  # 使用最多4个核心
    print(f"使用 {n_jobs} 个核心进行并行计算...")

    # 并行计算SHAP值（带优化进度条）
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_shap_for_sample)(sample, explainer, i)
        for i, sample in enumerate(tqdm(
            samples_to_process,
            desc="并行计算SHAP值",
            total=len(samples_to_process),
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [用时:{elapsed}<预计:{remaining}]"
        ))
    )

    # 处理结果
    success_count = 0
    for i, result in enumerate(results):
        all_shap_values.append(result['shap_values'])
        all_data.append(result['data'])
        all_labels.append(result['label'])
        instance_names.append(f"Sample_{i}")
        if result['success']:
            success_count += 1

    # 合并所有结果
    if all_shap_values:
        shap_values = np.array(all_shap_values).squeeze()
    else:
        # 创建空结果
        shap_values = np.zeros((max_samples, num_properties))

    if all_data:
        data = np.array(all_data)
    else:
        data = np.zeros((max_samples, num_properties))

    labels = np.array(all_labels) if all_labels else np.zeros(max_samples)

    print(f"SHAP值计算完成 | 成功样本: {success_count}/{len(samples_to_process)} | 形状: {shap_values.shape}")
    print(f"数据形状: {data.shape}")

    # 特征名称就是属性名称
    feature_names = PROPERTY_NAMES

    # 创建数据框
    data_df = pd.DataFrame(data, columns=feature_names)

    # 计算执行时间
    elapsed_time = time.time() - start_time
    print(f"SHAP分析完成 | 总耗时: {elapsed_time:.2f}秒 | 平均每样本: {elapsed_time / len(samples_to_process):.2f}秒")

    return shap_values, base_value, data_df, feature_names, instance_names, labels


# ===================================================
# 新增：位置协同效应分析
# ===================================================
def analyze_position_interactions(model, data_loader, device, tokenizer, save_dir, num_samples=5):
    """
    分析序列位置之间的协同效应
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    results = []

    # 获取一批数据
    for batch_idx, batch_data in enumerate(data_loader):
        seq = batch_data['seq_ids'].to(device)
        labels = batch_data['labels'].to(device)
        props = batch_data['properties'].to(device)
        orig_sequences = batch_data['sequence_strs']
        # 只分析正样本（AMP）
        amp_indices = torch.where(labels == 1)[0]
        if len(amp_indices) == 0:
            continue

        # 取前几个样本
        for i in amp_indices[:min(num_samples, len(amp_indices))]:
            single_seq = seq[i:i + 1].to(device)
            single_props = props[i:i + 1].to(device)

            # 原始预测
            with torch.no_grad():
                orig_output = model(single_seq, single_props).item()

            # 获取序列长度
            seq_length = single_seq.size(1)
            interaction_matrix = np.zeros((seq_length, seq_length))
            amino_acids = []

            # 解码氨基酸序列
            for pos in range(seq_length):
                aa_idx = single_seq[0, pos].item()
                amino_acids.append(tokenizer.inverse_vocab.get(aa_idx, 'X'))

            # 计算位置对之间的相互作用
            for i_pos in range(seq_length):
                for j_pos in range(i_pos + 1, seq_length):
                    # 创建扰动序列（同时突变两个位置）
                    perturbed_seq = single_seq.clone()
                    perturbed_seq[0, i_pos] = tokenizer.vocab['-']  # 使用填充token
                    perturbed_seq[0, j_pos] = tokenizer.vocab['-']  # 使用填充token

                    # 获取预测
                    with torch.no_grad():
                        perturbed_output = model(perturbed_seq, single_props).item()

                    # 计算单个位置突变的影响
                    perturbed_i = single_seq.clone()
                    perturbed_i[0, i_pos] = tokenizer.vocab['-']
                    with torch.no_grad():
                        output_i = model(perturbed_i, single_props).item()

                    perturbed_j = single_seq.clone()
                    perturbed_j[0, j_pos] = tokenizer.vocab['-']
                    with torch.no_grad():
                        output_j = model(perturbed_j, single_props).item()

                    # 计算协同效应：Δ(ij) - [Δ(i) + Δ(j)]
                    delta_ij = orig_output - perturbed_output
                    delta_i = orig_output - output_i
                    delta_j = orig_output - output_j

                    interaction = delta_ij - (delta_i + delta_j)
                    interaction_matrix[i_pos, j_pos] = interaction

            # 保存结果
            peptide_seq = ''.join(amino_acids)
            results.append({
                "sequence": peptide_seq,
                "interaction_matrix": interaction_matrix,
                "original_score": orig_output,
                "label": labels[i].item()
            })

        # 只分析前num_samples个样本
        if len(results) >= num_samples:
            break

    # 可视化每个序列的协同效应
    for i, result in enumerate(results):
        plt.figure(figsize=(12, 10))
        matrix = result['interaction_matrix']

        # 创建对称矩阵用于热力图
        full_matrix = np.zeros((matrix.shape[0], matrix.shape[0]))
        for i_pos in range(matrix.shape[0]):
            for j_pos in range(i_pos + 1, matrix.shape[0]):
                full_matrix[i_pos, j_pos] = matrix[i_pos, j_pos]
                full_matrix[j_pos, i_pos] = matrix[i_pos, j_pos]

        # 绘制热力图
        sns.heatmap(
            full_matrix,
            cmap="coolwarm",
            center=0,
            vmin=-0.5,
            vmax=0.5,
            annot=True,
            fmt=".2f",
            xticklabels=list(result['sequence']),
            yticklabels=list(result['sequence'])
        )

        plt.title(f"Position Interaction Matrix for Sequence {i + 1}\n(Original Score: {result['original_score']:.4f})")
        plot_path = os.path.join(save_dir, f"position_interaction_{i + 1}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        # 保存数据
        data_path = os.path.join(save_dir, f"position_interaction_{i + 1}.csv")
        df = pd.DataFrame(
            full_matrix,
            index=list(result['sequence']),
            columns=list(result['sequence'])
        )
        df.to_csv(data_path)

    print(f"位置协同效应分析完成，结果保存至: {save_dir}")
    return {
        "num_analyzed": len(results),
        "save_dir": save_dir
    }



class Tokenizer:
    def __init__(self):
        # 定义氨基酸到索引的映射
        self.vocab = {
            'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5,
            'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
            'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15,
            'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20,
            '-': 21,   # 填充和未知字符
        }
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        # 词汇表大小保持21（索引1-21）
        self.vocab_size = 21  # 索引范围：1-21

    def validate_sequence(self, sequence):
        """验证序列中的氨基酸是否都在词汇表中"""
        valid_chars = set(self.vocab.keys())
        invalid_chars = [aa for aa in sequence if aa not in valid_chars]

        if invalid_chars:
            print(f"警告: 发现无效氨基酸: {set(invalid_chars)}")
            print(f"在序列中: {sequence}")
            return False
        return True

    def __call__(self, sequence):
        """将氨基酸序列转换为索引列表"""
        tokens = []
        for aa in sequence:
            # 确保只使用有效氨基酸
            if aa in self.vocab:
                token_id = self.vocab.get(aa, 21)  # 默认使用21（X）
                tokens.append(token_id)
            else:
                # print(f"替换无效氨基酸 '{aa}' 为 '-'")
                tokens.append(self.vocab['-'])
        return tokens

    def validate_index(self, index):
        """验证索引是否在有效范围内 (1-21)"""
        return 1 <= index <= 21

    def decode(self, indices):
        """将整数索引转换回字符串序列"""
        return ''.join([self.inverse_vocab.get(i, '-') for i in indices])

    def encode(self, sequence, max_length):
        """编码序列并处理填充/截断"""
        # 编码为ID
        ids = []
        for aa in sequence[:max_length]:
            if aa in self.vocab:
                ids.append(self.vocab[aa])
            else:
                # 未知字符处理 - 使用填充ID或特定ID
                ids.append(self.vocab.get('X', self.vocab['-']))

        # 填充或截断
        if len(ids) < max_length:
            pad_id = self.vocab['-']
            ids += [pad_id] * (max_length - len(ids))
        elif len(ids) > max_length:
            ids = ids[:max_length]

        return ids
