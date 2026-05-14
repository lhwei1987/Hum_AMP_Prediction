import os
import yaml
import torch
import numpy as np
import time
import pandas as pd
import pickle
from scipy.spatial import cKDTree
from tqdm import tqdm
import multiprocessing
import torch.cuda.amp as amp
import concurrent.futures
import re
import warnings
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

from models.multimodal_cnn import AMPPredictionModel
from src.preprocessing import preprocess_sequence, extract_features
from src.utils import Tokenizer, time_estimator, progress_bar, AA_PROPERTIES, PROPERTY_NAMES, PROPERTY_RANGES, \
    normalize_value
from src.data_loader import create_data_loader
from src.preprocessing import load_dataset
from src.utils import calculate_metrics, plot_roc_curve, plot_pr_curve, plot_confusion_matrix, generate_sequence_indices

# 输入和输出文件夹路径
input_folder = "prediction_input"
output_folder = "prediction_output"
checkpoints_dir = "./checkpoints"
config_path = os.path.join(checkpoints_dir, "config.yaml")


def fix_config_file(config_path):
    """修复配置文件，移除Python对象"""
    print(f"[{time.strftime('%H:%M:%S')}] 检查配置文件: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查是否存在Python对象
    if '!!python/object:src.utils.Tokenizer' in content:
        print(f"[{time.strftime('%H:%M:%S')}] 检测到Python对象，正在修复...")

        # 使用default.yaml作为模板
        default_config_path = "configs/default.yaml"
        if os.path.exists(default_config_path):
            with open(default_config_path, 'r', encoding='utf-8') as f:
                default_config = yaml.safe_load(f)

            # 读取当前配置但跳过tokenizer
            with open(config_path, 'r', encoding='utf-8') as f:
                current_config = yaml.load(f, Loader=yaml.FullLoader)

            # 合并配置，移除tokenizer对象
            if 'data' in current_config and 'tokenizer' in current_config['data']:
                del current_config['data']['tokenizer']

            # 确保必要路径存在
            if 'data' in current_config:
                # 使用default.yaml中的默认路径
                current_config['data']['train_path'] = default_config['data']['train_path']
                current_config['data']['val_path'] = default_config['data']['val_path']
                current_config['data']['test_path'] = default_config['data']['test_path']

            # 保存修复后的配置
            backup_path = config_path + '.backup'
            if not os.path.exists(backup_path):
                os.rename(config_path, backup_path)
                print(f"[{time.strftime('%H:%M:%S')}] 原始配置已备份至: {backup_path}")

            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(current_config, f, default_flow_style=False)

            print(f"[{time.strftime('%H:%M:%S')}] ✅ 配置文件修复完成: {config_path}")
            return True
        else:
            print(f"[{time.strftime('%H:%M:%S')}] ⚠️ 警告: 默认配置文件未找到: {default_config_path}")
            return False
    else:
        print(f"[{time.strftime('%H:%M:%S')}] ✅ 配置文件无需修复")
        return True


def _get_model_files(checkpoints_dir):
    """获取所有模型文件，排除配置文件"""
    model_files = []
    for file in os.listdir(checkpoints_dir):
        if file.endswith(".pth"):
            model_files.append(os.path.join(checkpoints_dir, file))

    # 按F1分数排序
    def extract_f1(filename):
        match = re.search(r'f1_([\d.]+)\.pth', filename)
        if match:
            return float(match.group(1))
        return 0.0

    model_files.sort(key=extract_f1, reverse=True)
    return model_files


def _detect_model_property_count(model_path, device):
    """从模型的state_dict中检测使用的属性数量"""
    try:
        state_dict = torch.load(model_path, map_location=device)

        # 在state_dict中查找属性相关的权重
        for key in state_dict.keys():
            if 'property_importance_module' in key and 'weight' in key:
                weight_shape = state_dict[key].shape
                # 权重的第二个维度是输入特征数量
                if len(weight_shape) == 2:
                    return weight_shape[1]
                elif len(weight_shape) == 3:
                    # 对于卷积层: [out_channels, in_channels, kernel_size]
                    return weight_shape[1]

        # 如果没找到，检查property_projection层
        for key in state_dict.keys():
            if 'property_projection' in key and 'weight' in key:
                weight_shape = state_dict[key].shape
                if len(weight_shape) == 2:
                    return weight_shape[1]

        return None
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] 检测属性数量错误: {e}")
        return None


def _get_property_names_for_count(count):
    """根据数量获取属性名称"""
    if count <= len(PROPERTY_NAMES):
        return PROPERTY_NAMES[:count]
    else:
        # 如果数量大于可用属性，使用所有并添加虚拟属性
        property_names = list(PROPERTY_NAMES)
        for i in range(count - len(PROPERTY_NAMES)):
            property_names.append(f"dummy_property_{i + 1}")
        return property_names


def load_models(config_path, checkpoints_dir):
    """加载所有训练好的模型"""
    # 修复配置文件
    fix_config_file(config_path)

    # 加载配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 添加tokenizer
    config['data']['tokenizer'] = Tokenizer()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{time.strftime('%H:%M:%S')}] 使用设备: {device}")

    # 获取所有模型文件
    model_files = _get_model_files(checkpoints_dir)
    print(f"[{time.strftime('%H:%M:%S')}] 找到 {len(model_files)} 个模型文件")

    models = []
    model_configs = []
    model_property_counts = []

    for i, model_path in enumerate(model_files, 1):
        print(f"[{time.strftime('%H:%M:%S')}] 加载模型{i}: {os.path.basename(model_path)}")

        # 检测模型使用的属性数量
        property_count = _detect_model_property_count(model_path, device)

        if property_count is None:
            print(f"[{time.strftime('%H:%M:%S')}] ⚠️ 警告: 无法检测属性数量，使用默认值(4)")
            property_count = 4

        print(f"[{time.strftime('%H:%M:%S')}] 模型 {i} 使用 {property_count} 个属性")

        # 更新配置使用检测到的属性数量
        model_config = config.copy()
        if 'model' not in model_config:
            model_config['model'] = {}

        # 使用适当的属性数量
        property_names = _get_property_names_for_count(property_count)
        model_config['model']['important_properties'] = property_names

        # 创建模型
        try:
            model = AMPPredictionModel(model_config['model']).to(device)

            # 加载权重
            state_dict = torch.load(model_path, map_location=device)

            # 过滤state_dict以移除不兼容的键
            model_state_dict = model.state_dict()
            filtered_state_dict = {}

            for key, value in state_dict.items():
                if key in model_state_dict:
                    if model_state_dict[key].shape == value.shape:
                        filtered_state_dict[key] = value
                    else:
                        print(
                            f"[{time.strftime('%H:%M:%S')}] ⚠️ 形状不匹配 {key}: {value.shape} vs {model_state_dict[key].shape}")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] ⚠️ 模型中未找到的键: {key}")

            # 加载过滤后的state_dict
            model.load_state_dict(filtered_state_dict, strict=False)
            model.eval()

            models.append(model)
            model_configs.append(model_config)
            model_property_counts.append(property_count)

            print(f"[{time.strftime('%H:%M:%S')}] ✅ 模型{i}加载成功")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] ❌ 加载模型{i}失败: {e}")

    return models, model_configs, model_property_counts, device, model_files


def preprocess_single_sequence(args):
    """处理单个序列的预处理任务，用于并行处理"""
    idx, seq, max_seq_len, vocab_size, tokenizer, property_count = args
    try:
        processed_seq = preprocess_sequence(seq, max_seq_len)
        # 获取每个氨基酸的属性特征（二维数组）
        properties = extract_features(processed_seq)

        # 确保属性数组是二维的 (seq_len, num_properties)
        if len(properties.shape) == 1:
            # 如果是一维，转换为二维 (1, num_properties)
            properties = properties.reshape(1, -1)

        # 确保属性数组长度正确，截断或填充到property_count
        if properties.shape[1] > property_count:
            # 如果多于所需，截断
            properties = properties[:, :property_count]
        elif properties.shape[1] < property_count:
            # 如果少于所需，填充零
            padded_properties = np.zeros((properties.shape[0], property_count))
            padded_properties[:, :properties.shape[1]] = properties
            properties = padded_properties

        seq_ids = tokenizer(processed_seq)

        # 确保所有索引都在有效范围内
        valid_seq_ids = []
        for token_id in seq_ids:
            if token_id < 0 or token_id >= vocab_size:
                valid_seq_ids.append(tokenizer.vocab.get('X', 21))
            else:
                valid_seq_ids.append(token_id)

        return (idx, processed_seq, valid_seq_ids, properties, None)
    except Exception as e:
        # 返回默认长度的属性数组
        return (idx, "", [], np.zeros(property_count), str(e))


def predict_model(model, seq_tensor, prop_tensor, batch_orig_sequences):
    """单个模型的预测函数，用于线程池"""
    with torch.no_grad():
        with amp.autocast():
            return model(seq_tensor, prop_tensor, batch_orig_sequences)


@time_estimator
def predict_sequences(models, model_configs, model_property_counts, device, sequences):
    """使用多个模型预测肽序列是否为抗菌肽"""
    print(f"[{time.strftime('%H:%M:%S')}] 使用 {len(models)} 个模型预测 {len(sequences)} 个序列")

    # 使用第一个模型的配置作为基准
    base_config = model_configs[0]
    vocab_size = base_config['model']['vocab_size']
    max_seq_len = base_config['data']['max_seq_len']
    batch_size = 1024

    # 创建 tokenizer 实例
    tokenizer = Tokenizer()

    # 初始化结果数组
    all_processed_seqs = [""] * len(sequences)
    all_seq_ids = [[]] * len(sequences)
    all_properties = [None] * len(sequences)
    errors = [None] * len(sequences)

    # 使用最大属性数量进行预处理
    max_property_count = max(model_property_counts)

    # =================== 并行预处理部分 ===================
    print(f"[{time.strftime('%H:%M:%S')}] 并行预处理 {len(sequences)} 条序列...")
    # 创建进程池
    num_workers = min(multiprocessing.cpu_count(), 16)
    pool = multiprocessing.Pool(processes=num_workers)

    # 准备参数列表 (添加索引)
    args_list = [(i, seq, max_seq_len, vocab_size, tokenizer, max_property_count)
                 for i, seq in enumerate(sequences)]

    # 动态计算块大小
    chunk_size = max(10, len(sequences) // (num_workers * 10))
    print(f"[{time.strftime('%H:%M:%S')}] 使用 {num_workers} 进程, 块大小: {chunk_size}")

    try:
        # 使用imap_unordered以获得更好的性能
        error_count = 0
        with tqdm(total=len(sequences), desc="并行预处理", unit="seq") as pbar:
            for result in pool.imap_unordered(preprocess_single_sequence, args_list, chunksize=chunk_size):
                idx, processed_seq, seq_ids, properties, error = result

                # 确保属性是二维数组 (seq_len, num_features)
                if properties is not None:
                    # 如果是一维数组 (num_features,)，转换为二维 (1, num_features)
                    if len(properties.shape) == 1:
                        properties = properties.reshape(1, -1)

                    # 确保属性数组长度正确
                    if properties.shape[1] != max_property_count:
                        # 如果长度不匹配，创建正确形状的零数组
                        correct_properties = np.zeros((properties.shape[0], max_property_count))
                        # 尽可能复制有效数据
                        min_cols = min(properties.shape[1], max_property_count)
                        correct_properties[:, :min_cols] = properties[:, :min_cols]
                        properties = correct_properties
                        if error is None:
                            error = f"属性长度错误: {properties.shape[1]} != {max_property_count}"
                        else:
                            error += f"; 属性长度错误: {properties.shape[1]} != {max_property_count}"
                        error_count += 1

                # 存储结果
                all_processed_seqs[idx] = processed_seq
                all_seq_ids[idx] = seq_ids
                all_properties[idx] = properties
                errors[idx] = error

                pbar.update(1)

        if error_count > 0:
            print(f"[{time.strftime('%H:%M:%S')}] ⚠️ 警告: 发现 {error_count} 个序列的属性长度错误，已修正")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ⚠️ 并行预处理错误: {str(e)}")
        # 回退到单线程处理
        print(f"[{time.strftime('%H:%M:%S')}] ⚠️ 回退到单线程预处理...")
        for i, seq in tqdm(enumerate(sequences), total=len(sequences), desc="单线程预处理", unit="seq"):
            try:
                processed_seq = preprocess_sequence(seq, max_seq_len)
                properties = extract_features(processed_seq)

                # 确保属性是二维数组 (seq_len, num_features)
                if properties is not None:
                    # 如果是一维数组 (num_features,)，转换为二维 (1, num_features)
                    if len(properties.shape) == 1:
                        properties = properties.reshape(1, -1)

                    # 确保属性数组长度正确
                    if properties.shape[1] != max_property_count:
                        # 如果长度不匹配，创建正确形状的零数组
                        correct_properties = np.zeros((properties.shape[0], max_property_count))
                        # 尽可能复制有效数据
                        min_cols = min(properties.shape[1], max_property_count)
                        correct_properties[:, :min_cols] = properties[:, :min_cols]
                        properties = correct_properties

                seq_ids = tokenizer(processed_seq)
                valid_seq_ids = []
                for token_id in seq_ids:
                    if token_id < 0 or token_id >= vocab_size:
                        valid_seq_ids.append(tokenizer.vocab.get('X', 21))
                    else:
                        valid_seq_ids.append(token_id)
                all_processed_seqs[i] = processed_seq
                all_seq_ids[i] = valid_seq_ids
                all_properties[i] = properties
                errors[i] = None
            except Exception as e:
                all_processed_seqs[i] = ""
                all_seq_ids[i] = []
                all_properties[i] = None
                errors[i] = str(e)
    finally:
        pool.close()
        pool.join()

    # 创建三维属性数组 [batch_size, max_seq_len, max_property_count]
    properties_array = np.zeros((len(sequences), max_seq_len, max_property_count))
    for i, prop_matrix in enumerate(all_properties):
        if prop_matrix is not None:
            seq_len = min(prop_matrix.shape[0], max_seq_len)
            properties_array[i, :seq_len, :] = prop_matrix[:seq_len, :]

    # 处理序列ID数组
    seq_ids_array = np.zeros((len(sequences), max_seq_len), dtype=int)
    for i, ids in enumerate(all_seq_ids):
        if len(ids) > 0:
            padded_ids = np.pad(ids, (0, max(0, max_seq_len - len(ids)))[:max_seq_len])
            seq_ids_array[i] = padded_ids

    # =================== 预测部分 ===================
    print(f"[{time.strftime('%H:%M:%S')}] 开始预测...")
    results = []
    num_batches = (len(sequences) + batch_size - 1) // batch_size

    # 创建进度条
    predict_bar = tqdm(range(num_batches), total=num_batches, desc="预测批次", unit="batch",
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    start_time = time.time()
    processed_sequences = 0
    last_log_time = start_time

    # 禁用梯度计算
    torch.set_grad_enabled(False)

    for batch_idx in predict_bar:
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(sequences))
        batch_size_current = end_idx - start_idx

        # 准备批处理数据
        batch_seq = seq_ids_array[start_idx:end_idx]
        batch_orig_sequences = all_processed_seqs[start_idx:end_idx]

        # 转换为张量
        seq_tensor = torch.tensor(batch_seq, dtype=torch.long).to(device)

        # 并行执行所有模型
        model_outputs = []
        model_probs_list = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(models), 4)) as executor:
            futures = []
            for model_idx, model in enumerate(models):
                # 为每个模型准备相应的属性张量
                model_prop_count = model_property_counts[model_idx]
                batch_prop = properties_array[start_idx:end_idx, :, :model_prop_count]
                prop_tensor = torch.tensor(batch_prop, dtype=torch.float32).to(device)

                futures.append(executor.submit(
                    predict_model,
                    model, seq_tensor, prop_tensor, batch_orig_sequences
                ))

            for future in concurrent.futures.as_completed(futures):
                output = future.result()
                model_outputs.append(output)
                batch_probs = torch.sigmoid(output).squeeze().cpu().numpy()
                model_probs_list.append(batch_probs)

        # 转换为数组以便计算
        model_probs_array = np.array(model_probs_list)  # Shape: (n_models, batch_size_current)

        # 计算各种集成概率
        avg_probs = np.mean(model_probs_array, axis=0)  # 简单平均

        # 加权平均（按模型索引简单加权，也可以按F1分数加权）
        weights = np.array([1.0 / (i + 1) for i in range(len(models))])  # 越新的模型权重越高
        weights = weights / weights.sum()
        weighted_avg_probs = np.average(model_probs_array, axis=0, weights=weights)

        # 硬投票（多数表决）
        model_preds = (model_probs_array > 0.5).astype(int)
        hard_voting_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=model_preds)

        # 处理批处理结果
        for i in range(batch_size_current):
            orig_idx = start_idx + i
            seq = sequences[orig_idx]

            # 如果有错误，直接记录错误
            if errors[orig_idx]:
                results.append({
                    'sequence': seq,
                    'processed_sequence': "PREPROCESS_ERROR",
                    'error': errors[orig_idx]
                })
                continue

            # 收集每个模型的预测结果
            model_results = {}
            for model_idx in range(len(models)):
                model_name = f"Model{model_idx + 1:02d}"
                model_prob = model_probs_array[model_idx, i]
                model_pred = 1 if model_prob > 0.5 else 0
                model_results[f"{model_name}_Probability"] = model_prob
                model_results[f"{model_name}_Prediction"] = model_pred

            # 创建结果记录
            result_record = {
                'sequence': seq,
                'processed_sequence': all_processed_seqs[orig_idx],
                'average_probability': avg_probs[i],
                'average_prediction': 1 if avg_probs[i] > 0.5 else 0,
                'weighted_probability': weighted_avg_probs[i],
                'weighted_prediction': 1 if weighted_avg_probs[i] > 0.5 else 0,
                'hard_voting_prediction': hard_voting_preds[i],
                'error': None
            }

            # 添加每个模型的预测结果
            result_record.update(model_results)
            results.append(result_record)

        # 更新进度和显示信息
        processed_sequences += batch_size_current
        current_time = time.time()

        # 每30秒或每10000个批次显示一次详细信息
        if current_time - last_log_time > 30 or batch_idx % 10000 == 0:
            elapsed = current_time - start_time
            sequences_per_sec = processed_sequences / elapsed if elapsed > 0 else 0
            remaining_sequences = len(sequences) - processed_sequences
            remaining_time = remaining_sequences / sequences_per_sec if sequences_per_sec > 0 else 0

            # 添加GPU内存使用信息
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # MB
                gpu_info = f" | GPU Mem: {allocated:.1f}/{reserved:.1f} MB"
            else:
                gpu_info = ""

            print(f"[{time.strftime('%H:%M:%S')}] Batch {batch_idx + 1}/{num_batches} "
                  f"| Speed: {sequences_per_sec:.1f} seq/s "
                  f"| Remaining: {remaining_time / 60:.1f} min{gpu_info}")
            last_log_time = current_time

    predict_bar.close()
    return results


def generate_prediction_statistics(results, output_folder, input_file):
    """生成预测统计信息和可视化"""
    if not results:
        return

    # 提取序列和预测结果
    sequences = [r['sequence'] for r in results if 'error' not in r or not r['error']]
    avg_probs = [r['average_probability'] for r in results if 'error' not in r or not r['error']]
    avg_preds = [r['average_prediction'] for r in results if 'error' not in r or not r['error']]
    weighted_probs = [r['weighted_probability'] for r in results if 'error' not in r or not r['error']]
    weighted_preds = [r['weighted_prediction'] for r in results if 'error' not in r or not r['error']]
    hard_voting_preds = [r['hard_voting_prediction'] for r in results if 'error' not in r or not r['error']]

    if not sequences:
        return

    # 创建统计文件夹
    stats_dir = os.path.join(output_folder, "prediction_statistics")
    os.makedirs(stats_dir, exist_ok=True)

    # 基本统计
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    stats_file = os.path.join(stats_dir, f"prediction_stats_{os.path.splitext(input_file)[0]}_{timestamp}.txt")

    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("抗菌肽预测统计报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"预测时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"输入文件: {input_file}\n")
        f.write(f"总序列数: {len(results)}\n")

        # 错误统计
        error_count = sum(1 for r in results if 'error' in r and r['error'])
        f.write(f"预处理错误: {error_count}\n")
        f.write(f"有效序列: {len(sequences)}\n")
        f.write("=" * 60 + "\n\n")

        # 各种集成方法的预测结果
        for method_name, probs, preds in [
            ("简单平均", avg_probs, avg_preds),
            ("加权平均", weighted_probs, weighted_preds),
            ("硬投票", None, hard_voting_preds)
        ]:
            f.write(f"{method_name}方法:\n")
            if probs is not None:
                amp_count = sum(preds)
                non_amp_count = len(preds) - amp_count
                amp_percentage = (amp_count / len(preds)) * 100 if len(preds) > 0 else 0

                f.write(f"  抗菌肽数量: {amp_count} ({amp_percentage:.2f}%)\n")
                f.write(f"  非抗菌肽数量: {non_amp_count}\n")
                f.write(f"  平均概率: {np.mean(probs):.4f}\n")
                f.write(f"  概率标准差: {np.std(probs):.4f}\n")
                f.write(f"  概率范围: [{np.min(probs):.4f}, {np.max(probs):.4f}]\n")
            else:
                amp_count = sum(preds)
                non_amp_count = len(preds) - amp_count
                amp_percentage = (amp_count / len(preds)) * 100 if len(preds) > 0 else 0

                f.write(f"  抗菌肽数量: {amp_count} ({amp_percentage:.2f}%)\n")
                f.write(f"  非抗菌肽数量: {non_amp_count}\n")
            f.write("\n")

    print(f"[{time.strftime('%H:%M:%S')}] 📊 预测统计保存至: {stats_file}")

    # 生成可视化图表
    try:
        plt.style.use('seaborn-v0_8-whitegrid')

        # 1. 概率分布图
        plt.figure(figsize=(12, 8))
        plt.hist(avg_probs, bins=30, alpha=0.7, label='简单平均概率', density=True)
        plt.hist(weighted_probs, bins=30, alpha=0.7, label='加权平均概率', density=True)
        plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='阈值 (0.5)')
        plt.xlabel('预测概率', fontsize=12)
        plt.ylabel('密度', fontsize=12)
        plt.title('预测概率分布', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        prob_dist_path = os.path.join(stats_dir, f"probability_distribution_{timestamp}.png")
        plt.savefig(prob_dist_path, dpi=150)
        plt.close()

        # 2. 模型一致性热图
        if len(results) > 0 and 'Model01_Probability' in results[0]:
            # 提取所有模型的预测概率
            model_names = [f"Model{i + 1:02d}" for i in range(13)]  # 假设最多13个模型
            valid_model_names = []
            model_probs = []

            for model_name in model_names:
                prob_key = f"{model_name}_Probability"
                if prob_key in results[0]:
                    valid_model_names.append(model_name)
                    model_probs.append([r.get(prob_key, 0) for r in results if 'error' not in r or not r['error']])

            if model_probs:
                model_probs_array = np.array(model_probs)

                # 计算模型间的相关性
                correlation_matrix = np.corrcoef(model_probs_array)

                plt.figure(figsize=(10, 8))
                sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                            xticklabels=valid_model_names, yticklabels=valid_model_names,
                            center=0, vmin=-1, vmax=1)
                plt.title('模型预测相关性矩阵', fontsize=14)
                plt.tight_layout()
                corr_path = os.path.join(stats_dir, f"model_correlation_{timestamp}.png")
                plt.savefig(corr_path, dpi=150)
                plt.close()

        print(f"[{time.strftime('%H:%M:%S')}] 📈 可视化图表保存至: {stats_dir}")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ⚠️ 生成可视化图表时出错: {e}")


@time_estimator
def batch_predict(input_folder, output_folder, config_path, checkpoints_dir):
    """批量预测抗菌肽"""
    print(f"[{time.strftime('%H:%M:%S')}] 开始批量预测...")

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取输入文件列表
    input_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    if not input_files:
        print(f"[{time.strftime('%H:%M:%S')}] 在 {input_folder} 中没有找到CSV文件")
        return

    # 加载所有模型
    print(f"[{time.strftime('%H:%M:%S')}] 加载所有模型...")
    models, model_configs, model_property_counts, device, model_files = load_models(config_path, checkpoints_dir)

    if not models:
        print(f"[{time.strftime('%H:%M:%S')}] ❌ 没有加载到任何模型，退出预测")
        return

    # 创建统计文件
    total_stats_file = os.path.join(output_folder, f"total_prediction_stats_{time.strftime('%Y%m%d_%H%M%S')}.txt")

    # 初始化总统计
    total_stats = {
        'files_processed': 0,
        'total_sequences': 0,
        'total_errors': 0,
        'amp_counts': {'average': 0, 'weighted': 0, 'hard_voting': 0},
        'non_amp_counts': {'average': 0, 'weighted': 0, 'hard_voting': 0}
    }

    for input_file in input_files:
        input_path = os.path.join(input_folder, input_file)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_file = f"predictions_{os.path.splitext(input_file)[0]}_{timestamp}.csv"
        output_path = os.path.join(output_folder, output_file)

        # 读取输入数据
        print(f"[{time.strftime('%H:%M:%S')}] 处理文件: {input_path}")
        try:
            df = pd.read_csv(input_path)
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] ❌ 错误: 无法读取文件 {input_file}: {str(e)}")
            continue

        # 规范化列名
        df.columns = df.columns.str.strip().str.lower()

        # 检查必需的列
        if 'sequence' not in df.columns:
            sequence_cols = [col for col in df.columns if 'seq' in col or 'peptide' in col or 'aa' in col]
            if sequence_cols:
                print(
                    f"[{time.strftime('%H:%M:%S')}] ⚠️ 警告: 文件 {input_file} 缺少 'sequence' 列，但检测到类似列: {sequence_cols[0]}，将使用此列")
                df = df.rename(columns={sequence_cols[0]: 'sequence'})
            else:
                print(f"[{time.strftime('%H:%M:%S')}] ⚠️ 警告: 文件 {input_file} 缺少 'sequence' 列，跳过处理")
                continue

        sequences = df['sequence'].tolist()
        num_sequences = len(sequences)
        print(f"[{time.strftime('%H:%M:%S')}] 在 {input_file} 中找到 {num_sequences} 条序列")
        total_stats['total_sequences'] += num_sequences
        total_stats['files_processed'] += 1

        # ================ 添加超大规模数据集处理逻辑 ================
        MAX_SEQUENCES_PER_BATCH = 1000000  # 每个批次最多处理100万序列

        if num_sequences > MAX_SEQUENCES_PER_BATCH:
            print(f"[{time.strftime('%H:%M:%S')}] 检测到超大数据集，将数据分块处理...")
            chunks = [sequences[i:i + MAX_SEQUENCES_PER_BATCH] for i in
                      range(0, num_sequences, MAX_SEQUENCES_PER_BATCH)]

            all_results = []
            for chunk_idx, chunk in enumerate(chunks):
                print(f"[{time.strftime('%H:%M:%S')}] 处理分块 {chunk_idx + 1}/{len(chunks)} (大小: {len(chunk)})")
                chunk_results = predict_sequences(models, model_configs, model_property_counts, device, chunk)
                all_results.extend(chunk_results)

                # 每处理完一个分块就释放内存
                del chunk_results
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            results = all_results
        else:
            # 预测所有序列
            results = predict_sequences(models, model_configs, model_property_counts, device, sequences)
        # ================ 结束超大规模数据集处理逻辑 ================

        # 创建结果DataFrame
        result_df = pd.DataFrame(results)

        # 添加原始数据
        for col in df.columns:
            if col not in result_df.columns and col != 'sequence':
                result_df[col] = df[col]

        # 保存结果
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"[{time.strftime('%H:%M:%S')}] ✅ 预测结果保存至 {output_path}")

        # 更新总统计
        error_count = sum(1 for r in results if 'error' in r and r['error'])
        total_stats['total_errors'] += error_count

        # 统计各种方法的预测结果
        for method in ['average', 'weighted', 'hard_voting']:
            pred_col = f'{method}_prediction' if method != 'hard_voting' else 'hard_voting_prediction'
            if pred_col in result_df.columns:
                valid_results = result_df[result_df['error'].isna()]
                if not valid_results.empty:
                    amp_count = valid_results[pred_col].sum()
                    total_stats['amp_counts'][method] += amp_count
                    total_stats['non_amp_counts'][method] += (len(valid_results) - amp_count)

        # 为每个集成方法保存正样本
        for method_name, pred_col in [
            ('average', 'average_prediction'),
            ('weighted', 'weighted_prediction'),
            ('hard_voting', 'hard_voting_prediction')
        ]:
            if pred_col in result_df.columns:
                positive_df = result_df[result_df[pred_col] == 1]
                if not positive_df.empty:
                    positive_file = f"positive_{method_name}_{os.path.splitext(input_file)[0]}_{timestamp}.csv"
                    positive_path = os.path.join(output_folder, positive_file)
                    positive_df.to_csv(positive_path, index=False, encoding='utf-8-sig')
                    print(f"[{time.strftime('%H:%M:%S')}] ✅ {method_name}方法抗菌肽预测结果保存至 {positive_path}")
                    print(f"   发现 {len(positive_df)} 条抗菌肽序列")

        # 生成当前文件的统计信息
        generate_prediction_statistics(results, output_folder, input_file)

        # 释放内存
        del results, result_df
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 保存总统计
    print(f"\n[{'=' * 50}]")
    print(f"总体预测统计")
    print(f"[{'=' * 50}]")

    with open(total_stats_file, 'w', encoding='utf-8') as f:
        f.write("总体预测统计报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"统计时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"处理文件总数: {total_stats['files_processed']}\n")
        f.write(f"预测序列总数: {total_stats['total_sequences']}\n")
        f.write(f"预处理错误总数: {total_stats['total_errors']}\n")
        f.write("=" * 60 + "\n\n")

        for method in ['average', 'weighted', 'hard_voting']:
            total_preds = total_stats['amp_counts'][method] + total_stats['non_amp_counts'][method]
            if total_preds > 0:
                amp_percentage = (total_stats['amp_counts'][method] / total_preds) * 100
                f.write(f"{method}方法总体统计:\n")
                f.write(f"  抗菌肽总数: {total_stats['amp_counts'][method]} ({amp_percentage:.2f}%)\n")
                f.write(f"  非抗菌肽总数: {total_stats['non_amp_counts'][method]}\n")
                f.write(f"  有效预测总数: {total_preds}\n\n")

    print(f"[{time.strftime('%H:%M:%S')}] 处理文件总数: {total_stats['files_processed']}")
    print(f"[{time.strftime('%H:%M:%S')}] 预测序列总数: {total_stats['total_sequences']}")
    print(f"[{time.strftime('%H:%M:%S')}] 预处理错误总数: {total_stats['total_errors']}")
    print(f"[{time.strftime('%H:%M:%S')}] ✅ 总体统计保存至 {total_stats_file}")
    print(f"[{time.strftime('%H:%M:%S')}] 🎉 批量预测完成!")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("抗菌肽预测系统 - 多模型集成版本")
    print("=" * 70)

    # 检查checkpoints目录
    if not os.path.exists(checkpoints_dir):
        print(f"[{time.strftime('%H:%M:%S')}] ❌ 错误: checkpoints目录不存在: {checkpoints_dir}")
        print("请确保checkpoints目录包含训练好的模型文件")
        exit(1)

    # 检查配置文件
    if not os.path.exists(config_path):
        print(f"[{time.strftime('%H:%M:%S')}] ❌ 错误: 配置文件不存在: {config_path}")
        print("请确保checkpoints目录包含config.yaml文件")
        exit(1)

    # 检查输入文件夹
    if not os.path.exists(input_folder):
        print(f"[{time.strftime('%H:%M:%S')}] ⚠️ 警告: 输入文件夹不存在: {input_folder}")
        os.makedirs(input_folder, exist_ok=True)
        print(f"[{time.strftime('%H:%M:%S')}] ✅ 已创建输入文件夹: {input_folder}")
        print("请将要预测的CSV文件放入此文件夹，然后重新运行程序")
        exit(0)

    # 批量预测整个文件夹
    batch_predict(input_folder, output_folder, config_path, checkpoints_dir)
