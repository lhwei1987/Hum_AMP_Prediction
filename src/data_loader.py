import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from .utils import Tokenizer
from .preprocessing import preprocess_sequence, extract_features
import time
import logging


# 配置日志记录
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def peptide_collate_fn(batch):
    """
    处理批次数据
    """
    batch_size = len(batch)
    seq_ids = []
    labels = []
    properties = []
    sequence_strs = []
    indices = []  # 添加索引收集

    for item in batch:
        seq_ids.append(item['seq_ids'])
        labels.append(item['label'])
        properties.append(item['properties'])
        sequence_strs.append(item['sequence_str'])
        indices.append(item['idx'])

    return {
        'seq_ids': torch.stack(seq_ids),
        'labels': torch.cat(labels),
        'properties': torch.stack(properties),
        'sequence_strs': sequence_strs,
        'indices': torch.tensor(indices, dtype=torch.long)  # 添加索引张量
    }

class PeptideDataset(Dataset):

    def __init__(self, sequences, labels, properties, tokenizer, max_len=100):
        """
        肽序列数据集
        :param
        sequences: 肽序列列表
        :param
        labels: 标签列表
        :param
        properties: 物化性质矩阵列表
        :param
        tokenizer: Tokenizer对象用于序列编码
        :param
        max_len: 最大序列长度
        """
        self.sequences = sequences
        self.labels = labels
        self.properties = np.array(properties)  # 确保属性是numpy数组
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 确保所有特征矩阵形状一致并存储在实例变量中
        self.n_features = properties[0].shape[1]
        self.all_properties = []  # 修正后的特征矩阵列表

        # 修正特征矩阵尺寸
        for prop in properties:
            if prop.shape != (max_len, self.n_features):
                # 修正形状不一致的特征矩阵
                corrected = np.zeros((max_len, self.n_features))
                actual_len = min(prop.shape[0], max_len)
                corrected[:actual_len] = prop[:actual_len]
                self.all_properties.append(corrected)
            elif prop.shape[0] > max_len:
                # 截断到最大长度
                self.all_properties.append(prop[:max_len])
            else:
                self.all_properties.append(prop)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_str = self.sequences[idx]
        label = self.labels[idx]
        props = self.all_properties[idx]

        # 使用tokenizer编码序列
        seq_ids = self.tokenizer.encode(seq_str, self.max_len)

        # 确保序列ID长度与最大长度一致
        if len(seq_ids) < self.max_len:
            # 使用填充字符的索引进行填充 (21)
            pad_id = self.tokenizer.vocab['-']
            seq_ids += [pad_id] * (self.max_len - len(seq_ids))
        elif len(seq_ids) > self.max_len:
            # 截断序列
            seq_ids = seq_ids[:self.max_len]

        return {
            'seq_ids': torch.tensor(seq_ids, dtype=torch.long),
            'label': torch.tensor([label], dtype=torch.float32),
            'properties': torch.tensor(props, dtype=torch.float32),
            'sequence_str': seq_str,
            'idx': idx  # 添加索引
        }


def create_data_loader(sequences, labels, properties,config, shuffle=True):
    """
    创建数据加载器
    """
    tokenizer = config['data']['tokenizer']
    max_len = config['data']['max_seq_len']
    batch_size = config['data']['batch_size']
    num_workers = config['data'].get('num_workers', 0)  # Windows上设置为0以避免问题

    # 在Windows上，多进程常出现问题，将num_workers设为0
    if num_workers > 0:
        logger.warning("Windows平台不建议使用num_workers > 0，将设置为0")
        num_workers = 0

    # 创建数据集前验证所有序列
    valid_sequences = []
    valid_labels = []
    valid_properties = []


    for i, (seq, lbl, prop) in enumerate(zip(sequences, labels, properties)):
        # 验证序列中的所有字符
        valid = True
        for aa in seq:
            if aa not in tokenizer.vocab:
                print(f"过滤无效序列: 索引 {i}, 序列: {seq}, 无效字符: {aa}")
                valid = False
                break

        if valid:
            valid_sequences.append(seq)
            valid_labels.append(lbl)
            valid_properties.append(prop)


    # 修正标签维度
    valid_labels = np.array(valid_labels)
    if len(valid_labels.shape) > 1 and valid_labels.shape[1] == 1:
        valid_labels = valid_labels.squeeze(1)

    # 创建数据集
    dataset = PeptideDataset(
        sequences=valid_sequences,
        labels=valid_labels,
        properties=valid_properties,
        tokenizer=tokenizer,
        max_len=max_len
    )

    # 创建数据加载器
    batch_size = config['data'].get('batch_size', 64)
    num_workers = config['data'].get('num_workers', 4)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        # 添加原始序列引用
        collate_fn=peptide_collate_fn,
        pin_memory=True
    )

    print(
        f"已创建数据加载器 - 样本数: {len(dataset)}, 批次大小: {batch_size}, "
        f"是否打乱: {shuffle}, 工作进程: {num_workers}"
    )

    return loader
