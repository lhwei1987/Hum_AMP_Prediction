import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, config):
        """
        初始化CNN模型
        :param vocab_size: 词汇表大小
        :param embedding_dim: 嵌入维度
        :param num_classes: 类别数量
        :param config: 包含模型超参数的配置字典
        """
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 确保从配置中获取num_channels和kernel_size
        self.conv1 = nn.Conv1d(embedding_dim, config['cnn']['num_channels'], config['cnn']['kernel_size'])
        self.fc = nn.Linear(config['cnn']['num_channels'], num_classes)
        self.dropout = nn.Dropout(config['dropout_rate'])

    def forward(self, x):
        """
        前向传播
        :param x: 输入数据
        :return: 输出数据
        """
        x = self.embedding(x).permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = torch.max_pool1d(x, x.size(2)).squeeze(2)
        x = self.dropout(x)
        x = self.fc(x)
        return x