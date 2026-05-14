import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
from src.utils import PROPERTY_NAMES
import matplotlib.pyplot as plt
import seaborn as sns
from torch.distributions import Normal, kl_divergence
import math
import contextlib


# 在模型定义中添加属性特征重要性可视化网络分支
class PropertyImportance(nn.Module):
    """增强属性特征重要性的专用模块
    Args:
        n_prop_features: 输入特征维度
        hidden_dim: 隐藏层维度（默认64）
        enable_visualization: 是否允许生成可视化（默认False）
    """

    def __init__(self, n_prop_features, hidden_dim=64, enable_visualization=False):
        super().__init__()
        self.enable_visualization = enable_visualization
        print(
            f"初始化属性特征重要性模块: 输入特征={n_prop_features}, 隐藏层={hidden_dim}, 可视化={enable_visualization}")
        # 全局特征重要性分支
        self.global_importance = nn.Sequential(
            nn.Linear(n_prop_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_prop_features),
            nn.Sigmoid()  # 输出特征重要性权重（0-1）
        )
        # 位置相关特征重要性分支（可选）
        self.enable_positional_importance = True
        if self.enable_positional_importance:
            self.positional_importance = nn.Sequential(
                nn.Conv1d(n_prop_features, hidden_dim, 3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, n_prop_features, 1),
                nn.Sigmoid()
            )
            print(f"启用位置敏感特征重要性分析")
        # 可学习缩放参数增强模型容量
        self.scale_factor = nn.Parameter(torch.tensor(1.0))
        print(f"特征缩放因子初始值: {self.scale_factor.item():.2f}")

    def forward(self, props, visualize=False, batch_idx=0):
        """前向传播计算加权后的特征
        Args:
            props: 输入属性 [batch_size, seq_len, n_prop_features]
            visualize: 是否生成可视化数据
            batch_idx: 批次索引（用于可视化）
        Returns:
            包含加权特征和重要性权重的字典
        """
        # 计算全局重要性权重
        global_avg = props.mean(dim=1)  # [B, D]
        global_imp = self.global_importance(global_avg)  # [B, D]
        # 初始化位置重要性权重（如果有）
        pos_imp = torch.ones_like(props)  # 默认为1（无位置加权）
        # 计算位置相关重要性（如果启用）
        if self.enable_positional_importance:
            # 转换为卷积输入维度 [B, D, L]
            props_conv = props.permute(0, 2, 1)
            pos_conv_imp = self.positional_importance(props_conv)  # [B, D, L]
            pos_imp = pos_conv_imp.permute(0, 2, 1)  # [B, L, D]
        # 合并两种重要性
        combined_importance = global_imp.unsqueeze(1) * pos_imp  # [B, L, D]
        # 应用重要性权重
        weighted_features = self.scale_factor * props * combined_importance
        # 生成可视化数据（如果需要）
        if self.enable_visualization and visualize:
            self._generate_importance_visualization(
                props,
                global_imp,
                combined_importance,
                batch_idx
            )
        return {
            'weighted': weighted_features,
            'importance': global_imp,  # 返回全局重要性用于分析
            'positional': pos_imp if self.enable_positional_importance else None
        }

    # def _generate_importance_visualization(self, props, global_imp, combined_importance, batch_idx):
    #     """生成特征重要性可视化图表"""
    #     # 创建批次目录
    #     batch_dir = os.path.join("feature_importance_visualizations", f"batch_{batch_idx}")
    #     os.makedirs(batch_dir, exist_ok=True)
    #     # 获取特征名称
    #     feature_names = PROPERTY_NAMES  # 假设已定义
    #     importance_np = global_imp.detach().cpu().numpy()
    #     raw_properties_np = props.detach().cpu().numpy()
    #     # 1. 绘制全局特征重要性热力图
    #     plt.figure(figsize=(12, 8))
    #     plt.imshow(importance_np, cmap='viridis')
    #     plt.colorbar()
    #     plt.title(f'Batch {batch_idx} - Global Feature Importance')
    #     plt.xlabel('Feature Index')
    #     plt.ylabel('Sample Index')
    #     plt.savefig(os.path.join(batch_dir, 'global_importance.png'))
    #     plt.close()
    #     # 2. 绘制每个样本的特征重要性柱状图（选择前3个样本）
    #     for i in range(min(3, len(importance_np))):
    #         plt.figure(figsize=(10, 6))
    #         plt.bar(range(len(feature_names)), importance_np[i])
    #         plt.title(f'Sample {i} - Feature Importance')
    #         plt.xlabel('Feature')
    #         plt.ylabel('Importance')
    #         plt.xticks(range(len(feature_names)), feature_names, rotation=45)
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(batch_dir, f'sample_{i}_importance.png'))
    #         plt.close()
    #     # 3. 绘制特征值与重要性散点图
    #     for j in range(len(feature_names)):
    #         plt.figure(figsize=(8, 6))
    #         plt.scatter(raw_properties_np[:, j], importance_np[:, j], alpha=0.5)
    #         plt.title(f'Feature {feature_names[j]} - Value vs Importance')
    #         plt.xlabel('Feature Value')
    #         plt.ylabel('Importance')
    #         plt.grid(True, linestyle='--', alpha=0.3)
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(batch_dir, f'feature_{j}_scatter.png'))
    #         plt.close()



class EfficientBayesianLinear(nn.Module):
    """高效贝叶斯线性层，使用局部重参数化"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 变分参数
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        # 初始化
        nn.init.xavier_normal_(self.weight_mu)
        nn.init.constant_(self.weight_rho, -3)
        nn.init.normal_(self.bias_mu, 0, 0.1)
        nn.init.constant_(self.bias_rho, -3)
    def forward(self, x, sample=True):
        if self.training and sample:
            # 局部重参数化技巧
            gamma = F.linear(x, self.weight_mu, self.bias_mu)
            delta = F.linear(x ** 2, torch.exp(self.weight_rho) ** 2, torch.exp(self.bias_rho) ** 2)
            epsilon = torch.randn_like(gamma)
            return gamma + torch.sqrt(delta) * epsilon
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)
    def kl_divergence(self):
        weight_kl = 0.5 * (self.weight_mu ** 2 + torch.exp(self.weight_rho) ** 2 - self.weight_rho - 1).sum()
        bias_kl = 0.5 * (self.bias_mu ** 2 + torch.exp(self.bias_rho) ** 2 - self.bias_rho - 1).sum()
        return weight_kl + bias_kl  # 现在返回的是张量


class ResidualConvBlock(nn.Module):
    """残差卷积块"""

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = F.relu(out)
        return out


class AttentionLayer(nn.Module):
    """位置注意力机制模块"""

    def __init__(self, input_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.scale = input_dim ** -0.5
        self.last_attention = None  # 存储最后的注意力权重

    def forward(self, x):
        # x: [batch, seq_len, features]
        Q = self.query(x)  # [batch, seq_len, features]
        K = self.key(x)  # [batch, seq_len, features]
        V = self.value(x)  # [batch, seq_len, features]
        # 计算注意力权重
        attention = torch.matmul(Q, K.transpose(1, 2)) * self.scale
        attention = F.softmax(attention, dim=-1)
        # 存储注意力权重用于可视化
        self.last_attention = attention.detach().cpu().numpy()
        # 应用注意力权重
        attended = torch.matmul(attention, V)
        # 残差连接
        return attended + x

    def visualize(self, sequence):
        """生成注意力热力图"""
        if self.last_attention is None:
            return None
        plt.figure(figsize=(12, 4))
        sns.heatmap(self.last_attention[0], cmap="viridis")
        plt.xticks(np.arange(len(sequence)), list(sequence))
        plt.yticks(np.arange(len(sequence)), list(sequence))
        plt.title("Attention Map")
        return plt


class MultimodalFeatureFusion(nn.Module):
    """改进的多模态特征融合模块（包含属性特征重要性分析）"""

    def __init__(self, config):
        super().__init__()
        self.config = config  # 保存配置

        # 使用安全的方式获取配置值，设置默认值
        self.dropout_rate = config.get('dropout_rate', 0.2)
        self.embedding_dim = config.get('embedding_dim', 64)
        self.property_embed_dim = config.get('property_embed_dim', 48)
        self.fusion_channels = config.get('fusion_channels', 128)
        self.vocab_size = config.get('vocab_size', 128)
        self.enable_feature_visualization = config.get('enable_feature_visualization', False)
        self.important_properties = config.get('important_properties', [])
        # 创建属性索引字典
        self.prop_idx = {name: i for i, name in enumerate(PROPERTY_NAMES)}
        # 检查是否使用精简属性
        if self.important_properties:
            # 只选择存在的属性
            self.important_indices = [
                self.prop_idx[prop]
                for prop in self.important_properties
                if prop in self.prop_idx
            ]
            print(f"使用精简属性集: {self.important_properties}")
        else:
            self.important_indices = list(range(len(PROPERTY_NAMES)))
            print(f"使用全部属性: {PROPERTY_NAMES}")
        self.n_selected = len(self.important_indices)

        # 添加属性特征重要性分析模块
        self.property_importance_module = PropertyImportance(
            n_prop_features=self.n_selected,
            hidden_dim=64,
            enable_visualization=self.enable_feature_visualization
        )

        # 性质投影层 (只投影选择的属性)
        self.property_projection = nn.Sequential(
            nn.Linear(self.n_selected, self.property_embed_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )

        # 序列嵌入层
        self.seq_embedding = nn.Embedding(
            self.vocab_size,
            self.embedding_dim,
            padding_idx=0  # 添加填充索引
        )

        # 计算融合层的输入通道数
        self.fusion_in_channels = self.embedding_dim + self.property_embed_dim

        # 特征融合层-显式参数传递
        self.fusion_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=int(self.fusion_in_channels),  # 确保转换为整数
                out_channels=int(self.fusion_channels),
                kernel_size=3,
                padding=1,
                groups=1  # 明确设置groups=1
            ),
            nn.ReLU(),
            nn.BatchNorm1d(self.fusion_channels),
            nn.Dropout(self.dropout_rate)
        )

        # 增强属性特征交互
        self.property_interaction = nn.Sequential(
            nn.Linear(self.n_selected, 64),
            nn.ReLU(),
            nn.Linear(64, self.property_embed_dim)
        )

        # 特征门控机制
        self.feature_gate = nn.Sequential(
            nn.Linear(
                self.embedding_dim +
                self.property_embed_dim,
                1
            ),
            nn.Sigmoid()
        )

        # # 可视化目录设置
        # self.visualization_dir = "feature_importance_visualizations"
        # os.makedirs(self.visualization_dir, exist_ok=True)

        # 打印模型结构信息
        self.print_model_info()

        # 添加维度断言（确保各模块输出维度匹配）
        assert isinstance(self.fusion_in_channels, int), f"输入通道应为整数，实际为{type(self.fusion_in_channels)}"
        assert isinstance(self.fusion_channels, int), f"输出通道应为整数，实际为{type(self.fusion_channels)}"
        print(f"✅ 融合层维度验证通过: {self.fusion_in_channels}->{self.fusion_channels}")


    def print_model_info(self):
        """打印模型结构信息"""
        print("-" * 60)
        print("特征融合模块配置")
        print("-" * 60)
        print(f"序列嵌入维度: {self.embedding_dim}")
        print(f"属性嵌入维度: {self.property_embed_dim}")
        print(f"输入属性特征数量: {len(PROPERTY_NAMES)} | 精选属性特征: {self.n_selected}")
        print(f"融合输入通道: {self.fusion_in_channels} | 融合输出通道: {self.fusion_channels}")
        print("-" * 60)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"总参数: {total_params:,}")
        print("-" * 60)

    def forward(self, seq, props, sequence_strings):
        # 选择重要属性
        selected_props = props[:, :, self.important_indices]
        # 序列嵌入 [B, L] -> [B, L, E]
        seq_emb = self.seq_embedding(seq)
        # 应用属性特征重要性分析
        importance_output = self.property_importance_module(selected_props)
        properties_improved = importance_output['weighted']
        # 属性特征强化
        prop_interact = self.property_interaction(properties_improved)  # [B, L, E']
        prop_emb = self.property_projection(properties_improved) + prop_interact  # [B, L, E_p]

        # 扩展进化特征以匹配序列长度维度
        seq_len = seq_emb.size(1)  # 获取序列长度L

        # 现在所有张量都是3维: [B, L, *]
        combined = torch.cat([seq_emb, prop_emb], dim=2)
        # 调整维度用于卷积 [B, E+P+Evo, L]
        combined = combined.permute(0, 2, 1)
        # 特征融合 [B, F, L]
        fused_features = self.fusion_layer(combined)
        return fused_features

    def visualize_property_importance(self, importance_weights, raw_properties, sequence_ids, batch_idx):
        """可视化属性特征重要性"""
        # 创建批次目录
        batch_dir = os.path.join(self.visualization_dir, f"batch_{batch_idx}")
        os.makedirs(batch_dir, exist_ok=True)
        # 获取特征名称
        feature_names = [PROPERTY_NAMES[i] for i in self.important_indices]
        importance_np = importance_weights.detach().cpu().numpy()
        raw_properties_np = raw_properties.detach().cpu().numpy()
        # 1. 整体特征重要性热力图
        plt.figure(figsize=(12, 8))
        plt.imshow(importance_np, aspect='auto', cmap='viridis')
        plt.title(f'Batch {batch_idx} - Overall Feature Importance')
        plt.xlabel('Property Features')
        plt.ylabel('Samples')
        plt.xticks(np.arange(len(feature_names)), feature_names, rotation=45)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(batch_dir, 'feature_importance_heatmap.png'))
        plt.close()
        # 2. 按样本的特征重要性柱状图（只展示前5个样本）
        for i in range(min(5, len(sequence_ids))):
            plt.figure(figsize=(10, 6))
            plt.bar(feature_names, importance_np[i], color='skyblue')
            plt.title(f'Sample {sequence_ids[i]} - Feature Importance')
            plt.xticks(rotation=45)
            plt.ylabel('Weight')
            plt.ylim(0, 1.0)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(batch_dir, f'sample_{sequence_ids[i]}_importance.png'))
            plt.close()
        # 3. 特征值与重要性关联散点图（按特征）
        for j, prop_name in enumerate(feature_names):
            plt.figure(figsize=(8, 6))
            # 提取当前特征的值和重要性
            prop_values = raw_properties_np[:, :, j].flatten()
            importances = importance_np[:, j].repeat(raw_properties_np.shape[1])
            # 创建散点图
            plt.scatter(prop_values, importances, alpha=0.5)
            plt.title(f'Property: {prop_name}\nValue vs. Importance')
            plt.xlabel('Property Value')
            plt.ylabel('Feature Importance')
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(batch_dir, f'{prop_name}_scatter.png'))
            plt.close()
        # 4. 保存原始数据用于进一步分析
        importance_csv = os.path.join(batch_dir, 'feature_importance.csv')
        with open(importance_csv, 'w') as f:
            f.write("SampleID," + ",".join(feature_names) + "\n")
            for i, sid in enumerate(sequence_ids):
                f.write(f"{sid}," + ",".join([f"{w:.6f}" for w in importance_np[i]]) + "\n")
        print(f"特征可视化数据已保存至: {batch_dir}")

class MultiScaleFeatureExtractor(nn.Module):
    """改进的多尺度特征提取模块（整合残差连接、膨胀卷积和注意力）"""
    def __init__(self, config):
        super().__init__()
        # 添加膨胀卷积捕获长距离依赖
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(config['fusion_channels'], 64, kernel_size=3,
                          dilation=2 ** i, padding=2 ** i),
                nn.ReLU(),
                nn.BatchNorm1d(64)
            ) for i in range(3)
        ])
        # 残差卷积块
        self.conv_blocks = nn.ModuleList()
        kernel_sizes = config['kernel_sizes']
        for k in kernel_sizes:
            conv_block = nn.Sequential(
                ResidualConvBlock(config['fusion_channels'], 128, k),
                nn.MaxPool1d(2)
            )
            self.conv_blocks.append(conv_block)
        # 门控注意力机制
        self.gated_attention = nn.Sequential(
            nn.Linear(config['fusion_channels'], config['fusion_channels']),
            nn.Sigmoid()
        )
        # 通道注意力机制（使用标准线性层）
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(config['fusion_channels'], config['fusion_channels'] // 4),
            nn.ReLU(),
            nn.Linear(config['fusion_channels'] // 4, config['fusion_channels']),
            nn.Sigmoid()
        )
    def forward(self, x):
        # 门控注意力
        gate = self.gated_attention(x.mean(dim=-1))  # [B, C]
        gated_x = x * gate.unsqueeze(-1)
        # 通道注意力
        channel_att = self.channel_attention(gated_x)  # [B, C]
        gated_x = gated_x * channel_att.unsqueeze(-1)
        # 原始多尺度特征（残差卷积）
        conv_features = []
        for block in self.conv_blocks:
            feat = block(gated_x)  # [B, 128, L/2]
            # 全局平均池化
            pooled = F.adaptive_avg_pool1d(feat, 1).squeeze(-1)  # [B, 128]
            conv_features.append(pooled)
        # 膨胀卷积分支
        dilated_features = []
        for conv in self.dilated_convs:
            feat = conv(gated_x)  # [B, 64, L]
            pooled = F.adaptive_max_pool1d(feat, 1).squeeze(-1)  # [B, 64]
            dilated_features.append(pooled)
        # 组合所有特征
        combined = torch.cat(conv_features + dilated_features, dim=1)  # [B, (128*len(kernel_sizes))+64*3]
        return combined



class BayesianClassifier(nn.Module):
    """贝叶斯分类器（用于抗菌肽分类）"""
    def __init__(self, input_dim, hidden_dims, dropout_rate):
        super().__init__()
        layers = []
        current_dim = input_dim
        # 构建隐藏层
        for dim in hidden_dims:
            layers.append(nn.Linear(current_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = dim
        # 最终分类层
        layers.append(nn.Linear(current_dim, 1))
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)


class AMPPredictionModel(nn.Module):
    """改进的抗菌肽预测模型（模块化初始化）"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 使用安全方式获取配置值，提供默认值
        self.embedding_dim = config.get('embedding_dim', 64)
        self.property_embed_dim = config.get('property_embed_dim', 48)
        self.evo_embed_dim = config.get('evo_embed_dim', 8)
        self.fusion_channels = config.get('fusion_channels', 128)
        self.dropout_rate = config.get('dropout_rate', 0.2)
        self.vocab_size = config.get('vocab_size', 128)
        self.kernel_sizes = config.get('kernel_sizes', [3, 5, 7])
        self.classifier_dims = config.get('classifier_dims', [128, 64])
        self.evo_feature_dim = config.get('evo_feature_dim', 1024)
        self.has_feature_interaction = False  # 根据实际模型结构设置

        # 打印模型配置
        print("\n" + "=" * 60)
        print("初始化抗菌肽预测模型 - 安全配置版本")
        print("=" * 60)
        print(f"核心参数:")
        print(f"  - 序列嵌入维度: {self.embedding_dim}")
        print(f"  - 属性嵌入维度: {self.property_embed_dim}")
        print(f"  - 进化嵌入维度: {self.evo_embed_dim} (输入维度: {self.evo_feature_dim})")
        print(f"  - 融合通道: {self.fusion_channels}")
        print(f"  - Dropout率: {self.dropout_rate}")
        print(f"  - 卷积核尺寸: {self.kernel_sizes}")
        print(f"  - 分类器结构: {self.classifier_dims}")
        print("=" * 60)

        # 创建融合配置字典
        fusion_config = {
            'embedding_dim': self.embedding_dim,
            'property_embed_dim': self.property_embed_dim,
            'evo_embed_dim': self.evo_embed_dim,
            'fusion_channels': self.fusion_channels,
            'vocab_size': self.vocab_size,
            'dropout_rate': self.dropout_rate,
            'evo_feature_dim': self.evo_feature_dim,
            'kernel_sizes': self.kernel_sizes,
            'important_properties': config.get('important_properties', []),
            'enable_feature_visualization': config.get('enable_feature_visualization', False),
        }

        # 初始化模型组件
        self.feature_fusion = MultimodalFeatureFusion(fusion_config)

        # 特征提取器配置（直接使用主配置中的核心参数）
        self.feature_extractor = MultiScaleFeatureExtractor({
            'fusion_channels': self.fusion_channels,
            'kernel_sizes': self.kernel_sizes
        })

        # 计算特征提取器的输出维度
        input_dim = 128 * len(self.kernel_sizes) + 64 * 3

        # 初始化分类器
        self.classifier = nn.Sequential(
            *self._create_classifier_layers(input_dim, self.classifier_dims, self.dropout_rate),
            nn.Linear(self.classifier_dims[-1], 1)
        )
        print(f"分类器架构: {input_dim} → {' → '.join(map(str, self.classifier_dims))} → 1")

        # 初始化权重
        self.apply(self._init_weights)

        # 设备缓存
        self._device = None

    def _create_classifier_layers(self, input_dim, hidden_dims, dropout_rate):
        """创建分类器层序列"""
        layers = []
        current_dim = input_dim

        for dim in hidden_dims:
            layers.append(nn.Linear(current_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = dim

        return layers

    def _init_weights(self, module):
        """统一的权重初始化方法"""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.1)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def _get_fusion_config(self, config):
        """创建特征融合模块的配置"""
        return {
            'embedding_dim': self.embedding_dim,
            'property_embed_dim': self.property_embed_dim,
            'evo_embed_dim': self.evo_embed_dim,
            'dropout_rate': self.dropout_rate,
            'vocab_size': config['vocab_size'],
            'fusion_channels': self.fusion_channels,
            'kernel_sizes': self.kernel_sizes,
            'evo_feature_dim': self.evo_feature_dim,
            'enable_feature_visualization': config.get('enable_feature_visualization', False),
            'important_properties': config.get('important_properties', [])
        }


    def _init_classifier(self, config, input_dim):
        """创建分类器模块"""
        # 单一分类头（抗菌肽活性预测）
        self.classifier = BayesianClassifier(
            input_dim,
            config['classifier_dims'],
            config['dropout_rate']
        )
        print(f"🧠 分类器架构: {input_dim}→{'→'.join(map(str, config['classifier_dims']))}→1")


    def device(self):
        """获取模型所在的设备"""
        if self._device is None:
            # 从第一个参数获取设备
            self._device = next(self.parameters()).device
        return self._device

    def forward(self, seq, properties, sequence_strings):
        # 确保输入在相同设备
        seq = seq.to(self.device())
        properties = properties.to(self.device())

        # 特征融合
        fused = self.feature_fusion(seq, properties, sequence_strings)

        # 特征提取
        features = self.feature_extractor(fused)

        # 分类预测
        predictions = self.classifier(features)
        return predictions

    def predict_with_uncertainty(self, seq, properties, sequence_strings, n_samples=30):
        """预测并返回不确定性估计（单任务）"""
        self.train()  # 启用dropout
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self(seq, properties, sequence_strings)
                samples.append(pred.cpu())
        samples_tensor = torch.stack(samples)  # [n_samples, batch_size, 1]
        mean_pred = samples_tensor.mean(dim=0)
        std_pred = samples_tensor.std(dim=0)
        # 返回均值和标准差（不再返回字典）
        return mean_pred, std_pred

    def kl_loss(self):
        # 初始化张量而非浮点数，确保有dim属性
        device = next(self.parameters()).device  # 获取模型设备
        kl_div = torch.tensor(0.0, device=device)  # 初始化为张量

        for module in self.modules():
            if hasattr(module, 'kl_divergence'):
                module_kl = module.kl_divergence()
                # 确保模块返回的是张量
                if not isinstance(module_kl, torch.Tensor):
                    module_kl = torch.tensor(module_kl, device=device)
                kl_div = kl_div + module_kl

        # 确保返回标量
        if kl_div.dim() > 0:
            return kl_div.mean()
        return kl_div

    def pinn_loss(self, seq, props, pred):
        """全面的物理约束实现，确保返回标量损失"""
        # 创建属性索引字典
        prop_idx = {name: i for i, name in enumerate(PROPERTY_NAMES)}
        batch_loss = torch.zeros(props.size(0), device=props.device)  # 每个样本的损失

        # 1. 电荷约束 - AMP通常带正电荷
        if 'charge' in prop_idx:
            charge = props[:, :, prop_idx['charge']].mean(dim=1)  # [batch_size]
            # 惩罚电荷小于+2的情况
            charge_loss = F.relu(2.0 - charge)
            batch_loss += 0.3 * charge_loss

        # 2. 疏水性约束 - 最佳范围在0.3-0.7
        if 'hydrophobicity' in prop_idx:
            hydrophobicity = props[:, :, prop_idx['hydrophobicity']].mean(dim=1)
            # 惩罚过高和过低的疏水性
            hydro_loss = F.relu(0.3 - hydrophobicity) + F.relu(hydrophobicity - 0.7)
            batch_loss += 0.2 * hydro_loss

        # 3. 两亲性约束 - 需要疏水性变化
        if 'hydrophobicity' in prop_idx and props.size(1) > 5:
            hydro_var = props[:, :, prop_idx['hydrophobicity']].var(dim=1)
            amphipathic_loss = F.relu(0.05 - hydro_var)
            batch_loss += 0.2 * amphipathic_loss

        # 4. 二级结构约束
        if 'P_alpha' in prop_idx and 'P_beta' in prop_idx:
            alpha_prop = props[:, :, prop_idx['P_alpha']].mean(dim=1)
            beta_prop = props[:, :, prop_idx['P_beta']].mean(dim=1)
            secondary_loss = F.relu(0.2 - torch.max(alpha_prop, beta_prop))
            batch_loss += 0.3 * secondary_loss

        # 5. 序列长度约束
        seq_lengths = (seq != 0).sum(dim=1).float()  # 实际序列长度
        length_loss = F.relu(10 - seq_lengths) + F.relu(seq_lengths - 50)
        batch_loss += 0.1 * length_loss

        # 6. 半胱氨酸约束
        if 'cysteine' in prop_idx:
            cys_content = props[:, :, prop_idx['cysteine']].mean(dim=1)
            cys_loss = F.relu(cys_content - 0.1)
            batch_loss += 0.1 * cys_loss

        # 应用预测值作为权重
        pred_probs = torch.sigmoid(pred)
        if pred_probs.dim() > 1:
            pred_probs = pred_probs.squeeze(1)  # 确保形状匹配

        weighted_loss = batch_loss * pred_probs

        # 返回批次平均损失（标量）
        return weighted_loss.mean()

    def load_state_dict(self, state_dict, strict=True):
        """兼容性加载，处理架构变化"""
        # 1. 添加默认值防止属性缺失
        if not hasattr(self, 'has_feature_interaction'):
            self.has_feature_interaction = False

        # 2. 检查保存的模型是否有特征交互层
        saved_has_fi = any(k.startswith('fusion.feature_interaction') for k in state_dict.keys())

        # 3. 如果当前架构不同，调整状态字典
        if self.has_feature_interaction != saved_has_fi:
            print(f"架构不兼容警告：当前模型{'有' if self.has_feature_interaction else '无'}特征交互层，"
                  f"保存的模型{'有' if saved_has_fi else '无'}")

            # 删除额外的键
            keys_to_remove = []
            if saved_has_fi and not self.has_feature_interaction:
                keys_to_remove = [k for k in state_dict.keys()
                                  if k.startswith('fusion.feature_interaction')]

            for key in keys_to_remove:
                state_dict.pop(key)
                print(f"移除不兼容的键: {key}")

        # 4. 调用父类加载（非严格模式）
        super().load_state_dict(state_dict, strict=False)

        # 5. 打印加载结果
        missing = set(self.state_dict().keys()) - set(state_dict.keys())
        unexpected = set(state_dict.keys()) - set(self.state_dict().keys())

        if missing:
            print(f"⚠️ 缺失参数: {', '.join(missing)}")
        if unexpected:
            print(f"⚠️ 多余参数: {', '.join(unexpected)}")

        return self
