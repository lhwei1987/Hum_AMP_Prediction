import pandas as pd
import numpy as np
from .utils import AA_PROPERTIES, PROPERTY_NAMES, PROPERTY_RANGES, normalize_value, progress_bar
import time
import matplotlib.pyplot as plt

def preprocess_sequence(seq, max_len=100):
    """
    预处理肽序列:
    1. 转换为大写
    2. 处理无效字符
    3. 截断或填充到固定长度
    """
    # 转换为大写
    seq = seq.upper()

    # 定义有效氨基酸集合
    valid_aa = "ACDEFGHIKLMNPQRSTVWY-"

    # 处理无效字符 - 替换为X
    seq = ''.join(aa if aa in valid_aa else 'X' for aa in seq)

    # 截断或填充
    if len(seq) > max_len:
        seq = seq[:max_len]
    else:
        seq = seq + '-' * (max_len - len(seq))

    return seq


def extract_features(sequence, important_properties=None):
    """提取序列的特征矩阵"""
    if important_properties is None:
        important_properties = PROPERTY_NAMES

    n_features = len(important_properties)
    properties = []

    for aa in sequence:
        if aa in AA_PROPERTIES:
            aa_props = []
            for prop_name in important_properties:
                raw_value = AA_PROPERTIES[aa].get(prop_name, 0.0)
                min_val, max_val = PROPERTY_RANGES[prop_name]
                norm_value = normalize_value(raw_value, min_val, max_val)
                aa_props.append(norm_value)
            properties.append(aa_props)
        else:
            properties.append([0.0] * n_features)

    return np.array(properties)


def analyze_feature_correlations(properties, labels):
    n_features = properties[0].shape[1]
    correlations = []

    for i in range(n_features):
        feature_vals = np.concatenate([p[:, i] for p in properties])
        corr_coef = np.corrcoef(feature_vals, labels)[0, 1]
        correlations.append(corr_coef)

    # 可视化
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(n_features), correlations, color='skyblue')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.7)
    plt.title('Property Feature Correlations with Target Labels')
    plt.ylabel('Correlation Coefficient')
    plt.xlabel('Feature Index')
    plt.xticks(range(n_features), PROPERTY_NAMES, rotation=90)

    # 添加相关系数值
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height,
                 f'{correlations[idx]:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    return correlations

def augment_data(sequence, properties):
    """
    肽序列数据增强策略:
    1. 随机截断
    2. 随机替换氨基酸
    3. 添加噪声
    """
    # 随机截断 (70%概率)
    if np.random.rand() > 0.7 and len(sequence) > 10:
        start = np.random.randint(0, len(sequence) - 10)
        end = np.random.randint(start + 5, len(sequence))
        sequence = sequence[start:end]
        properties = properties[start:end]

    # 随机替换 (20%概率)
    if np.random.rand() > 0.8:
        idx = np.random.randint(0, len(sequence) - 1)
        new_aa = np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"))
        sequence = sequence[:idx] + new_aa + sequence[idx + 1:]

        # 更新替换位置的属性
        for j, prop_name in enumerate(PROPERTY_NAMES):
            raw_value = AA_PROPERTIES[new_aa][prop_name]
            min_val, max_val = PROPERTY_RANGES[prop_name]
            properties[idx, j] = normalize_value(raw_value, min_val, max_val)

    # 添加噪声 (10%概率)
    if np.random.rand() > 0.9:
        noise = np.random.normal(0, 0.1, properties.shape)
        properties = np.clip(properties + noise, 0, 1)

    return sequence, properties


def load_dataset(data_path, max_len=100, augment=False, show_progress=True):
    """
    加载并预处理数据集
    """
    start_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    sequences = []
    labels = []
    properties_list = []

    # 确保使用全局PROPERTY_NAMES定义的特征数量
    n_features = len(PROPERTY_NAMES)

    # 创建进度条
    total_sequences = len(df)
    iterator = df.iterrows()
    if show_progress:
        iterator = progress_bar(iterator, total=total_sequences, desc="Processing sequences", unit="seq")

    processed_count = 0
    for idx, row in iterator:
        seq = row['sequence']
        label = row['label']

        # 预处理序列
        processed_seq = preprocess_sequence(seq, max_len)

        # 提取特征
        properties = extract_features(processed_seq)

        # 确保特征矩阵形状正确 (max_len, n_features)
        if properties.shape != (max_len, n_features):
            # 创建正确形状的空矩阵
            corrected_properties = np.zeros((max_len, n_features))

            # 计算实际可复制的行数和列数
            actual_rows = min(properties.shape[0], max_len)
            actual_cols = min(properties.shape[1], n_features)

            # 填充实际特征
            corrected_properties[:actual_rows, :actual_cols] = properties[:actual_rows, :actual_cols]

            properties = corrected_properties

        # 数据增强
        if augment:
            processed_seq, properties = augment_data(processed_seq, properties)

            # 增强后再次检查形状
            if properties.shape != (max_len, n_features):
                corrected_properties = np.zeros((max_len, n_features))
                actual_rows = min(properties.shape[0], max_len)
                actual_cols = min(properties.shape[1], n_features)
                corrected_properties[:actual_rows, :actual_cols] = properties[:actual_rows, :actual_cols]
                properties = corrected_properties

        sequences.append(processed_seq)
        labels.append(label)
        properties_list.append(properties)
        processed_count += 1

    # 验证所有特征矩阵形状一致
    shapes = set([prop.shape for prop in properties_list])
    if len(shapes) > 1:
        print(f"Warning: Found {len(shapes)} different feature matrix shapes")
        # 强制统一形状
        uniform_properties = []
        for prop in properties_list:
            if prop.shape != (max_len, n_features):
                corrected = np.zeros((max_len, n_features))
                actual_rows = min(prop.shape[0], max_len)
                actual_cols = min(prop.shape[1], n_features)
                corrected[:actual_rows, :actual_cols] = prop[:actual_rows, :actual_cols]
                uniform_properties.append(corrected)
            else:
                uniform_properties.append(prop)
        properties_list = uniform_properties

    properties_array = np.stack(properties_list)

    # 添加额外的数据完整性检查
    print(f"[{time.strftime('%H:%M:%S')}] 验证数据完整性...")
    for i in range(min(5, len(sequences))):  # 检查前5个样本
        seq_len = len(sequences[i])
        prop_shape = properties_array[i].shape
        if seq_len > max_len:
            print(f"⚠️ 警告: 样本 {i} 序列长度 {seq_len} > 最大长度 {max_len}")
        if prop_shape != (max_len, n_features):
            print(f"⚠️ 警告: 样本 {i} 特征矩阵形状 {prop_shape} != ({max_len}, {n_features})")
    # 确保所有标签都是数值型
    non_numeric_labels = [label for label in labels if not isinstance(label, (int, float))]
    if non_numeric_labels:
        print(f"⚠️ 警告: 发现 {len(non_numeric_labels)} 个非数值型标签，转换为数值型...")
        labels = [float(label) for label in labels]
    elapsed = time.time() - start_time
    print(f"[{time.strftime('%H:%M:%S')}] Processed {processed_count} sequences in {elapsed:.2f} seconds")
    return sequences, np.array(labels), properties_array