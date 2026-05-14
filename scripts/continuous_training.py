import os
import time
import yaml
import torch
import numpy as np
import shutil
import json
from datetime import datetime
from pathlib import Path
import sys
import traceback
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
sys.path.append(project_root)

print(f"当前脚本目录: {current_dir}")
print(f"项目根目录: {project_root}")

from training import train_model as train_single_model
from src.preprocessing import load_dataset
from src.data_loader import create_data_loader
from src.utils import Tokenizer, calculate_metrics


class ContinuousModelTrainer:
    def __init__(self, config_path, data_path, target_f1=0.945, target_count=13, max_trials=500):
        self.config_path = config_path
        self.data_path = data_path
        self.target_f1 = target_f1
        self.target_count = target_count
        self.max_trials = max_trials

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # 数据路径检查
        train_path = os.path.join(data_path, "train.csv")
        val_path = os.path.join(data_path, "val.csv")

        print(f"检查训练数据路径: {train_path}")
        print(f"训练数据存在: {os.path.exists(train_path)}")
        print(f"验证数据存在: {os.path.exists(val_path)}")

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"训练数据不存在: {train_path}")
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"验证数据不存在: {val_path}")

        # 更新配置 - 修复：移除 min_len 参数
        self.config['data']['train_path'] = train_path
        self.config['data']['val_path'] = val_path
        self.config['data']['max_seq_len'] = 50

        # 调整训练参数
        self.config['training']['epochs'] = 150
        self.config['training']['learning_rate'] = 0.005
        self.config['training']['patience'] = 20
        self.config['training']['batch_size'] = 32

        # 禁用所有分析功能以避免错误
        if 'analysis' not in self.config:
            self.config['analysis'] = {}

        # 禁用所有分析功能
        self.config['analysis']['shap_analysis'] = False
        self.config['analysis']['positional_impact'] = False
        self.config['analysis']['peptide_contribution'] = False
        self.config['analysis']['position_interaction'] = False
        self.config['analysis']['property_importance'] = False  # 禁用属性重要性分析
        self.config['analysis']['property_differences'] = False  # 禁用属性差异分析
        self.config['analysis']['feature_elimination'] = False  # 禁用特征剔除

        # 确保特征选择阈值设置为0（不剔除任何特征）
        self.config['analysis']['feature_elimination_threshold'] = 0

        # 禁用属性选择功能
        if 'model' not in self.config:
            self.config['model'] = {}
        if 'property_selection' in self.config['model']:
            self.config['model']['property_selection'] = False

        print("✅ 已禁用所有分析功能以加快训练速度并保持架构一致")

        # 创建输出目录
        self.base_output_dir = os.path.join(project_root, "continuous_training_results")

        # 创建带时间戳的模型保存目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_save_dir = os.path.join(project_root, f"checkpoints_models_{timestamp}")

        # 确保所有目录都存在且有写权限
        os.makedirs(self.base_output_dir, exist_ok=True)
        os.makedirs(self.model_save_dir, exist_ok=True)

        # 特别创建项目根目录下的checkpoints目录，因为training.py会尝试保存到这里
        checkpoints_dir = os.path.join(project_root, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)

        # 检查目录权限
        import stat
        def check_dir_permissions(directory):
            if os.path.exists(directory):
                st = os.stat(directory)
                return bool(st.st_mode & stat.S_IWUSR)
            return False

        print(f"✅ 模型将保存到目录: {self.model_save_dir}")
        print(f"✅ 检查点目录已创建: {checkpoints_dir}")

        # 检查目录权限
        if not check_dir_permissions(self.model_save_dir):
            print(f"⚠️ 警告: 模型保存目录可能没有写入权限: {self.model_save_dir}")
            try:
                os.chmod(self.model_save_dir, stat.S_IRWXU)
                print(f"✅ 已修复目录权限")
            except:
                print(f"❌ 无法修复目录权限，可能需要手动设置")

        # 初始化统计信息
        self.successful_models = []
        self.failed_models = []
        self.trial_count = 0
        self.all_f1_scores = []

        print(f"连续训练器初始化完成:")
        print(f"  目标F1值: {target_f1}")
        print(f"  目标模型数: {target_count}")
        print(f"  最大训练次数: {max_trials}")
        print(f"  训练数据路径: {data_path}")
        print(f"  模型保存目录: {self.model_save_dir}")
        print(f"  调整后的训练参数:")
        print(f"    训练轮数: {self.config['training']['epochs']}")
        print(f"    学习率: {self.config['training']['learning_rate']}")
        print(f"    批大小: {self.config['training']['batch_size']}")

    def evaluate_model(self, model_path, config):
        """评估模型性能 - 修复版本，确保模型架构一致"""
        try:
            val_sequences, val_labels, val_properties = load_dataset(
                config['data']['val_path'],
                max_len=config['data']['max_seq_len']
            )

            if len(val_sequences) == 0:
                print("错误: 验证集为空")
                return 0.0

            print(f"评估模型: 验证集大小={len(val_sequences)}")

            tokenizer = Tokenizer()
            config['data']['tokenizer'] = tokenizer

            # 修复：确保属性特征数量一致
            if val_properties is not None and len(val_properties) > 0:
                print(f"验证集属性特征维度: {val_properties.shape}")

            val_loader = create_data_loader(
                sequences=val_sequences,
                labels=val_labels,
                properties=val_properties,
                config=config,
                shuffle=False
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            from models.multimodal_cnn import AMPPredictionModel

            # 强制确保模型使用完整的属性集
            model_config = config['model'].copy()
            # 禁用任何可能改变架构的功能
            model_config['property_selection'] = False
            if 'analysis' in config:
                model_config['analysis'] = config['analysis'].copy()
                model_config['analysis']['feature_elimination'] = False

            model = AMPPredictionModel(model_config).to(device)

            try:
                state_dict = torch.load(model_path, map_location=device)

                # 获取模型当前状态
                model_state_dict = model.state_dict()

                # 过滤掉不匹配的键
                filtered_state_dict = {}
                missing_keys = []
                unexpected_keys = []

                for key, value in state_dict.items():
                    if key in model_state_dict:
                        if model_state_dict[key].shape == value.shape:
                            filtered_state_dict[key] = value
                        else:
                            print(
                                f"❌ 权重形状不匹配: {key}, 模型形状: {model_state_dict[key].shape}, 文件形状: {value.shape}")
                            missing_keys.append(key)
                    else:
                        unexpected_keys.append(key)

                # 加载过滤后的状态
                if filtered_state_dict:
                    model.load_state_dict(filtered_state_dict, strict=False)
                    print(f"✅ 加载模型权重: {len(filtered_state_dict)}/{len(state_dict)} 参数匹配")

                    if missing_keys:
                        print(f"⚠️ 缺少的键: {len(missing_keys)}")
                    if unexpected_keys:
                        print(f"⚠️ 意外的键: {len(unexpected_keys)}")
                else:
                    print(f"❌ 没有匹配的权重参数")
                    return 0.0

            except Exception as e:
                print(f"加载模型权重失败: {e}")
                print("尝试宽松加载...")
                model.load_state_dict(state_dict, strict=False)

            model.eval()

            all_labels = []
            all_probs = []
            all_preds = []

            with torch.no_grad():
                for batch_data in val_loader:
                    seq = batch_data['seq_ids'].to(device)
                    props = batch_data['properties'].to(device)
                    orig_sequences = batch_data['sequence_strs']

                    labels = batch_data['labels'].cpu().numpy().flatten()

                    try:
                        outputs = model(seq, props, orig_sequences)
                        probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                        preds = (probs > 0.5).astype(int)

                        all_labels.extend(labels)
                        all_probs.extend(probs)
                        all_preds.extend(preds)
                    except Exception as e:
                        print(f"❌ 模型前向传播失败: {e}")
                        continue

            if len(all_labels) > 0:
                metrics = calculate_metrics(
                    np.array(all_labels),
                    np.array(all_preds),
                    np.array(all_probs)
                )
                f1_score = metrics.get('f1', 0.0)

                print(f"✅ 验证集评估结果:")
                print(f"  准确率: {metrics.get('accuracy', 0.0):.4f}")
                print(f"  精确率: {metrics.get('precision', 0.0):.4f}")
                print(f"  召回率: {metrics.get('recall', 0.0):.4f}")
                print(f"  F1分数: {f1_score:.4f}")
                print(f"  ROC AUC: {metrics.get('roc_auc', 0.0):.4f}")

                return f1_score
            else:
                print("警告: 验证集没有数据")
                return 0.0

        except Exception as e:
            print(f"❌ 模型评估失败: {e}")
            traceback.print_exc()
            return 0.0

    def save_model_with_timestamp_f1(self, source_model_path, f1_score, trial_num, timestamp, config=None):
        """
        保存模型并包含时间戳和F1值在文件名中
        """
        # 生成带时间戳和F1值的文件名
        model_filename = f"model_trial{trial_num}_{timestamp}_f1_{f1_score:.4f}.pth"
        model_save_path = os.path.join(self.model_save_dir, model_filename)

        # 保存模型文件
        try:
            if os.path.exists(source_model_path):
                shutil.copy2(source_model_path, model_save_path)
                print(f"✅ 模型已保存: {model_filename}")

                # 创建元数据文件 - 修复JSON序列化问题
                metadata = {
                    'model_name': model_filename,
                    'model_path': str(model_save_path),  # 转换为字符串
                    'trial_number': int(trial_num),  # 确保是int
                    'f1_score': float(f1_score),  # 转换为float
                    'timestamp': str(timestamp),  # 转换为字符串
                    'save_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'target_f1': float(self.target_f1),
                    'met_target': bool(f1_score >= self.target_f1)  # 转换为bool
                }

                # 保存元数据
                metadata_filename = f"{os.path.splitext(model_filename)[0]}_metadata.json"
                metadata_path = os.path.join(self.model_save_dir, metadata_filename)

                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

                print(f"📋 元数据已保存: {metadata_filename}")

                # 保存配置文件
                if config is not None:
                    config_filename = f"config_trial{trial_num}_f1_{f1_score:.4f}.yaml"
                    config_path = os.path.join(self.model_save_dir, config_filename)
                    with open(config_path, 'w') as f:
                        yaml.dump(config, f)
                    print(f"📝 配置文件已保存: {config_filename}")

                return model_save_path, metadata_path
            else:
                print(f"❌ 源模型文件不存在: {source_model_path}")
                return None, None

        except Exception as e:
            print(f"❌ 保存模型失败: {e}")
            traceback.print_exc()
            return None, None

    def find_model_files(self, trial_dir):
        """查找模型文件，支持多种可能的路径"""
        possible_paths = [
            os.path.join(trial_dir, "checkpoints", "best_model.pth"),
            os.path.join(trial_dir, "checkpoints", "final_model.pth"),
            os.path.join(trial_dir, "best_model.pth"),
            os.path.join(trial_dir, "final_model.pth"),
            os.path.join(trial_dir, "model.pth"),
            os.path.join(project_root, "checkpoints", "best_model.pth"),
            os.path.join(project_root, "checkpoints", "final_model.pth")
        ]

        # 添加通配符搜索
        import glob
        search_patterns = [
            os.path.join(trial_dir, "**", "*.pth"),
            os.path.join(trial_dir, "**", "checkpoints", "*.pth"),
            os.path.join(project_root, "checkpoints", "*.pth")
        ]

        for pattern in search_patterns:
            for path in glob.glob(pattern, recursive=True):
                possible_paths.append(path)

        for path in possible_paths:
            if os.path.exists(path):
                return path

        # 如果还没找到，深度搜索
        pth_files = []
        for root, dirs, files in os.walk(trial_dir):
            for file in files:
                if file.endswith('.pth'):
                    pth_files.append(os.path.join(root, file))

        if pth_files:
            # 返回最近修改的文件
            pth_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return pth_files[0]

        return None

    def train_single_trial(self, trial_num):
        """单次训练 - 修复版本，确保架构一致"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"\n{'=' * 60}")
        print(f"开始第 {trial_num} 次训练")
        print(f"时间戳: {timestamp}")
        print(f"{'=' * 60}")

        trial_dir_base = os.path.join(self.base_output_dir, f"trial_{trial_num}_{timestamp}")
        trial_dir = trial_dir_base
        os.makedirs(trial_dir, exist_ok=True)

        trial_config = self.config.copy()
        trial_config['training']['save_dir'] = trial_dir

        # 关键修复：确保禁用所有分析功能
        if 'analysis' not in trial_config:
            trial_config['analysis'] = {}

        # 禁用所有分析功能
        trial_config['analysis']['shap_analysis'] = False
        trial_config['analysis']['positional_impact'] = False
        trial_config['analysis']['peptide_contribution'] = False
        trial_config['analysis']['position_interaction'] = False
        trial_config['analysis']['property_importance'] = False  # 禁用属性重要性分析
        trial_config['analysis']['property_differences'] = False  # 禁用属性差异分析
        trial_config['analysis']['feature_elimination'] = False  # 禁用特征剔除
        trial_config['analysis']['feature_elimination_threshold'] = 0  # 设置阈值为0

        # 确保模型配置中禁用属性选择
        if 'model' not in trial_config:
            trial_config['model'] = {}
        trial_config['model']['property_selection'] = False

        # 设置一个标志来告诉training.py跳过分析阶段
        trial_config['skip_analysis'] = True

        # 设置随机种子
        import random
        seed = trial_num * 1000 + int(time.time()) % 1000
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        print(f"随机种子: {seed}")
        print(
            f"配置确认: 已禁用所有分析功能，特征剔除阈值={trial_config['analysis'].get('feature_elimination_threshold', 0)}")

        try:
            print("开始训练...")

            # 训练模型
            best_f1 = train_single_model(trial_config)

            # 查找最佳模型文件
            best_model_path = self.find_model_files(trial_dir)

            if best_model_path is None:
                print("❌ 未找到任何模型文件，跳过评估")
                # 检查项目根目录下的checkpoints
                checkpoints_dir = os.path.join(project_root, "checkpoints")
                if os.path.exists(checkpoints_dir):
                    best_model_path = self.find_model_files(checkpoints_dir)
                    if best_model_path:
                        print(f"✅ 在checkpoints目录找到模型文件: {best_model_path}")
                    else:
                        print("❌ 在checkpoints目录也未找到模型文件")
                        return None, 0.0
                else:
                    return None, 0.0

            print(f"✅ 找到模型文件: {best_model_path}")

            print("评估模型性能...")
            f1_score = self.evaluate_model(best_model_path, trial_config)
            self.all_f1_scores.append(f1_score)

            print(f"训练完成 - F1分数: {f1_score:.4f}")

            # 保存模型和元数据（总是保存，无论是否达到目标）
            model_save_path, metadata_path = self.save_model_with_timestamp_f1(
                source_model_path=best_model_path,
                f1_score=f1_score,
                trial_num=trial_num,
                timestamp=timestamp,
                config=trial_config
            )

            # 重命名目录以包含F1分数
            if f1_score > 0 and os.path.exists(trial_dir):
                try:
                    new_trial_dir = f"{trial_dir_base}_f1_{f1_score:.4f}"
                    os.rename(trial_dir, new_trial_dir)
                    trial_dir = new_trial_dir
                    print(f"✅ 已重命名目录为: {os.path.basename(trial_dir)}")
                except Exception as e:
                    print(f"❌ 重命名目录失败: {e}")

            if f1_score >= self.target_f1:
                print(f"🎉 模型达到目标F1值 ({f1_score:.4f} >= {self.target_f1})")

                # 同时保存一个简化名称的副本（方便使用）
                simple_name = f"best_model_f1_{f1_score:.4f}.pth"
                simple_path = os.path.join(self.model_save_dir, simple_name)

                if os.path.exists(best_model_path):
                    shutil.copy2(best_model_path, simple_path)
                    print(f"✅ 简化版模型已保存: {simple_name}")
                else:
                    print(f"❌ 源模型文件不存在，无法复制: {best_model_path}")

                model_info = {
                    'trial_num': trial_num,
                    'timestamp': timestamp,
                    'f1_score': f1_score,
                    'model_path': model_save_path,
                    'simple_path': simple_path,
                    'metadata_path': metadata_path,
                    'trial_dir': trial_dir,
                    'seed': seed,
                    'category': 'successful'
                }
                self.successful_models.append(model_info)

                return model_info, f1_score

            else:
                print(f"⚠️ 模型未达到目标F1值 ({f1_score:.4f} < {self.target_f1})")

                # 如果F1分数较高，也认为是有用的模型
                if f1_score > 0.8 and os.path.exists(best_model_path):
                    debug_name = f"useful_model_trial{trial_num}_f1_{f1_score:.4f}.pth"
                    debug_path = os.path.join(self.model_save_dir, debug_name)
                    shutil.copy2(best_model_path, debug_path)
                    print(f"✅ 有用模型已保存: {debug_name}")

                failure_info = {
                    'trial_num': trial_num,
                    'timestamp': timestamp,
                    'f1_score': f1_score,
                    'model_path': model_save_path,
                    'metadata_path': metadata_path,
                    'trial_dir': trial_dir,
                    'seed': seed,
                    'category': 'failed'
                }
                self.failed_models.append(failure_info)

                return None, f1_score

        except Exception as e:
            print(f"❌ 训练过程中出现错误: {e}")
            traceback.print_exc()

            failure_info = {
                'trial_num': trial_num,
                'timestamp': timestamp,
                'error': str(e),
                'trial_dir': trial_dir,
                'seed': seed,
                'category': 'error',
                'f1_score': 0.0  # 添加默认F1分数
            }
            self.failed_models.append(failure_info)

            return None, 0.0

    def run_continuous_training(self):
        """运行连续训练"""
        print("\n" + "=" * 80)
        print("开始连续训练")
        print("=" * 80)

        start_time = time.time()

        while (self.trial_count < self.max_trials and
               len(self.successful_models) < self.target_count):

            self.trial_count += 1
            print(f"\n=== 进度: 第 {self.trial_count}/{self.max_trials} 次训练 ===")
            print(f"成功模型: {len(self.successful_models)}/{self.target_count}")

            model_info, f1_score = self.train_single_trial(self.trial_count)

            self.print_statistics()

            if len(self.successful_models) >= self.target_count:
                print(f"\n🎉 已达到目标模型数量 ({len(self.successful_models)}/{self.target_count})")
                break

        total_time = time.time() - start_time
        hours = total_time // 3600
        minutes = (total_time % 3600) // 60
        seconds = total_time % 60

        self.generate_final_report(total_time)

        print("\n" + "=" * 80)
        print("连续训练完成!")
        print("=" * 80)
        print(f"总训练次数: {self.trial_count}")
        print(f"成功模型数: {len(self.successful_models)}")
        print(f"失败模型数: {len(self.failed_models)}")
        print(f"总耗时: {hours:.0f}小时 {minutes:.0f}分钟 {seconds:.0f}秒")
        print(f"模型保存目录: {self.model_save_dir}")
        print("=" * 80)

        if self.successful_models:
            self.select_best_model()
            self.generate_model_summary()
            # 生成可视化报告
            self.generate_visualization_reports()

        return self.successful_models

    def print_statistics(self):
        """打印统计信息"""
        print(f"\n📊 当前统计:")
        print(f"  训练次数: {self.trial_count}")
        print(f"  成功模型: {len(self.successful_models)}/{self.target_count}")
        print(f"  失败模型: {len(self.failed_models)}")

        if self.all_f1_scores:
            valid_scores = [score for score in self.all_f1_scores if score > 0]
            if valid_scores:
                print(f"  所有F1值范围: {min(valid_scores):.4f} - {max(valid_scores):.4f}")
                print(f"  平均F1值: {np.mean(valid_scores):.4f}")
                print(f"  中位数F1值: {np.median(valid_scores):.4f}")
                print(f"  标准差: {np.std(valid_scores):.4f}")

        if self.successful_models:
            f1_scores = [m['f1_score'] for m in self.successful_models]
            print(f"  成功模型F1值范围: {min(f1_scores):.4f} - {max(f1_scores):.4f}")
            print(f"  平均F1值: {np.mean(f1_scores):.4f}")

    def select_best_model(self):
        """选择最佳模型"""
        if not self.successful_models:
            return

        sorted_models = sorted(self.successful_models, key=lambda x: x['f1_score'], reverse=True)
        best_model = sorted_models[0]

        print(f"\n🏆 最佳模型:")
        print(f"  训练次数: {best_model['trial_num']}")
        print(f"  F1分数: {best_model['f1_score']:.4f}")
        print(f"  时间戳: {best_model['timestamp']}")
        print(f"  模型文件: {os.path.basename(best_model['model_path'])}")

        # 创建符号链接或复制最佳模型
        best_model_link = os.path.join(self.model_save_dir, "best_model.pth")
        try:
            if os.path.exists(best_model_link):
                os.remove(best_model_link)
            os.symlink(best_model['model_path'], best_model_link)
            print(f"✅ 最佳模型符号链接已创建: {best_model_link}")
        except Exception as e:
            print(f"创建符号链接失败: {e}")
            shutil.copy2(best_model['model_path'], best_model_link)
            print(f"✅ 最佳模型已复制: {best_model_link}")

    def generate_model_summary(self):
        """生成模型摘要报告"""
        summary_dir = os.path.join(self.model_save_dir, "model_summary")
        os.makedirs(summary_dir, exist_ok=True)

        # 生成CSV摘要
        summary_data = []
        for model in self.successful_models:
            summary_data.append({
                'trial_num': model['trial_num'],
                'f1_score': model['f1_score'],
                'timestamp': model['timestamp'],
                'model_file': os.path.basename(model['model_path']),
                'category': model.get('category', 'successful'),
                'seed': model.get('seed', 'unknown')
            })

        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_path = os.path.join(summary_dir, "models_summary.csv")
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"📊 模型摘要CSV已保存: {csv_path}")

            # 保存所有训练尝试的数据
            all_data = []
            for model in self.successful_models + self.failed_models:
                all_data.append({
                    'trial_num': model['trial_num'],
                    'f1_score': model.get('f1_score', 0.0),
                    'timestamp': model.get('timestamp', ''),
                    'category': model.get('category', 'unknown'),
                    'seed': model.get('seed', 'unknown')
                })

            all_df = pd.DataFrame(all_data)
            all_csv_path = os.path.join(summary_dir, "all_trials_summary.csv")
            all_df.to_csv(all_csv_path, index=False, encoding='utf-8')
            print(f"📊 所有训练尝试CSV已保存: {all_csv_path}")

            return df, all_df

        return None, None

    def generate_final_report(self, total_time):
        """生成最终报告"""
        report_dir = os.path.join(self.base_output_dir, "final_report")
        os.makedirs(report_dir, exist_ok=True)

        report_path = os.path.join(report_dir, "training_report.txt")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("连续训练最终报告\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总训练次数: {self.trial_count}\n")
            f.write(f"成功模型数: {len(self.successful_models)}\n")
            f.write(f"失败模型数: {len(self.failed_models)}\n")
            f.write(f"总耗时: {total_time:.1f}秒 ({total_time / 3600:.2f}小时)\n")
            f.write(f"目标F1值: {self.target_f1}\n")
            f.write(f"目标模型数: {self.target_count}\n\n")

            valid_scores = [score for score in self.all_f1_scores if score > 0]
            if valid_scores:
                f.write(f"所有F1值统计:\n")
                f.write(f"  最小值: {min(valid_scores):.4f}\n")
                f.write(f"  最大值: {max(valid_scores):.4f}\n")
                f.write(f"  平均值: {np.mean(valid_scores):.4f}\n")
                f.write(f"  中位数: {np.median(valid_scores):.4f}\n")
                f.write(f"  标准差: {np.std(valid_scores):.4f}\n\n")

            f.write("成功模型列表:\n")
            f.write("-" * 60 + "\n")
            if self.successful_models:
                for i, model in enumerate(self.successful_models, 1):
                    model_name = os.path.basename(model['model_path'])
                    f.write(f"{i:2d}. 训练次数: {model['trial_num']:3d}, F1值: {model['f1_score']:.4f}, "
                            f"时间: {model['timestamp']}, 模型: {model_name}\n")
            else:
                f.write("  无成功模型\n")

            f.write(f"\n模型保存位置:\n")
            f.write(f"  模型目录: {os.path.abspath(self.model_save_dir)}\n")
            f.write(f"  训练结果目录: {os.path.abspath(self.base_output_dir)}\n")

        print(f"\n📋 最终报告已保存至: {report_dir}")

        print("\n🎯 成功模型列表:")
        print("-" * 60)
        if self.successful_models:
            for i, model in enumerate(self.successful_models, 1):
                model_name = os.path.basename(model['model_path'])
                print(f"{i:2d}. {model_name} (F1: {model['f1_score']:.4f}, 训练次数: {model['trial_num']})")
        else:
            print("  无成功模型")

    def generate_visualization_reports(self):
        """生成可视化报告"""
        try:
            # 获取数据
            df_success, df_all = self.generate_model_summary()

            if df_success is not None and len(df_success) > 0:
                # 创建可视化目录
                vis_dir = os.path.join(self.model_save_dir, "visualization")
                os.makedirs(vis_dir, exist_ok=True)

                # 生成HTML报告
                self.generate_html_summary(vis_dir, df_success)

                # 生成性能图表
                self.plot_model_performance(vis_dir, df_success)

                # 生成F1分布图
                self.plot_f1_distribution(vis_dir)

                # 生成所有训练尝试的图表
                if df_all is not None and len(df_all) > 0:
                    self.plot_all_trials_performance(vis_dir, df_all)

                print(f"📊 可视化报告已生成到: {vis_dir}")

        except Exception as e:
            print(f"❌ 生成可视化报告时出错: {e}")
            traceback.print_exc()

    def generate_html_summary(self, output_dir, df):
        """生成HTML摘要报告"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Training Summary Report</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
                h2 {{ color: #555; margin-top: 30px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .stat-box {{ background: #e8f5e9; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .success {{ color: #4CAF50; font-weight: bold; }}
                .warning {{ color: #ff9800; }}
                .error {{ color: #f44336; }}
                .img-container {{ text-align: center; margin: 20px 0; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Continuous Training Model Summary Report</h1>
                <p>Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

                <div class="stat-box">
                    <h2>Overall Statistics</h2>
                    <p>Total Trials: {self.trial_count}</p>
                    <p>Successful Models: <span class="success">{len(self.successful_models)}</span></p>
                    <p>Failed Models: <span class="warning">{len(self.failed_models)}</span></p>
                    <p>Target F1 Score: {self.target_f1}</p>
                    <p>Target Model Count: {self.target_count}</p>
                </div>

                <h2>Model List (Sorted by F1 Score)</h2>
                <table>
                    <tr>
                        <th>Rank</th>
                        <th>Trial #</th>
                        <th>F1 Score</th>
                        <th>Timestamp</th>
                        <th>Model File</th>
                    </tr>
        """

        # 添加模型行
        df_sorted = df.sort_values('f1_score', ascending=False)
        for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
            html_content += f"""
                    <tr>
                        <td>{i}</td>
                        <td>{int(row['trial_num'])}</td>
                        <td><strong>{row['f1_score']:.4f}</strong></td>
                        <td>{row['timestamp']}</td>
                        <td>{row['model_file']}</td>
                    </tr>
            """

        html_content += f"""
                </table>

                <h2>F1 Score Distribution</h2>
                <div class="img-container">
                    <img src="f1_distribution.png" alt="F1 Score Distribution Chart">
                </div>

                <h2>Training History</h2>
                <div class="img-container">
                    <img src="training_history.png" alt="Training History Chart">
                </div>

                <h2>Model Performance Overview</h2>
                <div class="img-container">
                    <img src="model_performance.png" alt="Model Performance Overview">
                </div>

                <div style="margin-top: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 5px;">
                    <h3>📝 Notes</h3>
                    <p>1. All models are saved in: {self.model_save_dir}</p>
                    <p>2. Each model includes timestamp and F1 score in filename</p>
                    <p>3. Best model accessible via symbolic link 'best_model.pth'</p>
                    <p>4. Each model has corresponding metadata file (.json)</p>
                    <p>5. Total training time: {self.trial_count} trials</p>
                </div>
            </div>
        </body>
        </html>
        """

        html_path = os.path.join(output_dir, "model_summary.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"📄 HTML summary report saved: {html_path}")

    def plot_model_performance(self, output_dir, df):
        """绘制模型性能图表"""
        try:
            plt.figure(figsize=(15, 10))

            # 1. F1 Score Distribution
            plt.subplot(2, 2, 1)
            plt.hist(df['f1_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(x=self.target_f1, color='red', linestyle='--', linewidth=2,
                        label=f'Target F1: {self.target_f1}')
            plt.xlabel('F1 Score')
            plt.ylabel('Number of Models')
            plt.title('F1 Score Distribution of Models')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 2. Training History
            plt.subplot(2, 2, 2)
            df_sorted_by_trial = df.sort_values('trial_num')
            plt.plot(df_sorted_by_trial['trial_num'], df_sorted_by_trial['f1_score'],
                     'o-', markersize=4)
            plt.axhline(y=self.target_f1, color='red', linestyle='--',
                        label=f'Target F1: {self.target_f1}')
            plt.xlabel('Trial Number')
            plt.ylabel('F1 Score')
            plt.title('F1 Score vs Trial Number')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 3. Box Plot
            plt.subplot(2, 2, 3)
            plt.boxplot(df['f1_score'])
            plt.axhline(y=self.target_f1, color='red', linestyle='--', alpha=0.7)
            plt.ylabel('F1 Score')
            plt.title('F1 Score Distribution Box Plot')
            plt.grid(True, alpha=0.3)

            # 4. Scatter Plot
            plt.subplot(2, 2, 4)
            plt.scatter(df['trial_num'], df['f1_score'], alpha=0.6, s=50)
            plt.axhline(y=self.target_f1, color='red', linestyle='--',
                        label=f'Target F1: {self.target_f1}')
            plt.xlabel('Trial Number')
            plt.ylabel('F1 Score')
            plt.title('Trial Number vs F1 Score')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = os.path.join(output_dir, "model_performance.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

            # 单独保存F1分布图
            plt.figure(figsize=(10, 6))
            plt.hist(df['f1_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(x=self.target_f1, color='red', linestyle='--', linewidth=2,
                        label=f'Target F1: {self.target_f1}')
            plt.xlabel('F1 Score')
            plt.ylabel('Number of Models')
            plt.title(f'F1 Score Distribution (Total Models: {len(df)})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            f1_dist_path = os.path.join(output_dir, "f1_distribution.png")
            plt.savefig(f1_dist_path, dpi=150, bbox_inches='tight')
            plt.close()

            # 单独保存训练历史图
            plt.figure(figsize=(12, 6))
            df_sorted_by_trial = df.sort_values('trial_num')
            plt.plot(df_sorted_by_trial['trial_num'], df_sorted_by_trial['f1_score'],
                     'o-', markersize=4, linewidth=1)
            plt.fill_between(df_sorted_by_trial['trial_num'], 0, df_sorted_by_trial['f1_score'], alpha=0.2)
            plt.axhline(y=self.target_f1, color='red', linestyle='--', linewidth=2,
                        label=f'Target F1: {self.target_f1}')
            plt.xlabel('Trial Number')
            plt.ylabel('F1 Score')
            plt.title('Training History - F1 Score Progression')
            plt.legend()
            plt.grid(True, alpha=0.3)
            history_path = os.path.join(output_dir, "training_history.png")
            plt.savefig(history_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"📈 Model performance charts saved to: {output_dir}")

        except Exception as e:
            print(f"❌ Error plotting model performance: {e}")
            traceback.print_exc()

    def plot_f1_distribution(self, report_dir):
        """绘制F1分数分布图（包含成功标记）"""
        try:
            if not self.all_f1_scores:
                return

            plt.figure(figsize=(10, 6))
            plt.hist(self.all_f1_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(x=self.target_f1, color='red', linestyle='--', linewidth=2,
                        label=f'Target F1: {self.target_f1}')

            if self.successful_models:
                success_f1 = [m['f1_score'] for m in self.successful_models]
                plt.hist(success_f1, bins=20, alpha=0.7, color='green',
                         edgecolor='black', label='Successful Models')

            plt.xlabel('F1 Score')
            plt.ylabel('Frequency')
            plt.title(
                f'F1 Score Distribution (Total Trials: {self.trial_count}, Successful: {len(self.successful_models)})')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plot_path = os.path.join(report_dir, "f1_distribution_with_success.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"📈 F1 score distribution plot saved: {plot_path}")

        except Exception as e:
            print(f"❌ Failed to plot F1 distribution: {e}")
            traceback.print_exc()

    def plot_all_trials_performance(self, output_dir, df):
        """绘制所有训练尝试的性能图表"""
        try:
            plt.figure(figsize=(14, 8))

            # 按类别着色
            colors = {'successful': 'green', 'failed': 'orange', 'error': 'red'}
            point_colors = [colors.get(cat, 'gray') for cat in df['category']]

            plt.scatter(df['trial_num'], df['f1_score'], c=point_colors,
                        alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

            # 添加目标线
            plt.axhline(y=self.target_f1, color='red', linestyle='--',
                        linewidth=2, label=f'Target F1: {self.target_f1}')

            # 添加趋势线
            if len(df) > 1:
                z = np.polyfit(df['trial_num'], df['f1_score'], 1)
                p = np.poly1d(z)
                plt.plot(df['trial_num'], p(df['trial_num']), "b--",
                         alpha=0.5, label='Trend Line')

            plt.xlabel('Trial Number')
            plt.ylabel('F1 Score')
            plt.title(f'All Trials Performance (Total: {len(df)}, Successful: {len(self.successful_models)})')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 添加图例说明
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', alpha=0.6, label='Successful'),
                Patch(facecolor='orange', alpha=0.6, label='Failed'),
                Patch(facecolor='red', alpha=0.6, label='Error')
            ]
            plt.legend(handles=legend_elements + [plt.Line2D([0], [0], color='red', linestyle='--', label='Target')])

            plot_path = os.path.join(output_dir, "all_trials_performance.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"📈 All trials performance chart saved: {plot_path}")

        except Exception as e:
            print(f"❌ Error plotting all trials performance: {e}")
            traceback.print_exc()


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = current_dir

    print(f"当前脚本目录: {current_dir}")
    print(f"项目根目录: {project_root}")

    # 配置文件路径
    config_path = os.path.join(project_root, "configs", "default111.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(project_root, "config.yaml")
        if not os.path.exists(config_path):
            print(f"错误: 配置文件不存在: {config_path}")
            return

    print(f"使用配置文件: {config_path}")

    # 数据路径
    data_path = os.path.join(project_root, "data", "processed_new")
    print(f"训练数据路径: {data_path}")

    if not os.path.exists(data_path):
        print(f"错误: 训练数据路径不存在: {data_path}")
        print("请先运行数据处理脚本生成 processed_new 目录")
        return

    # 检查必要文件
    required_files = ["train.csv", "val.csv"]
    for file in required_files:
        file_path = os.path.join(data_path, file)
        if not os.path.exists(file_path):
            print(f"错误: 缺少必要文件 {file} 在 {data_path}")
            return

    print(f"训练数据检查通过")

    # 创建训练器实例
    trainer = ContinuousModelTrainer(
        config_path=config_path,
        data_path=data_path,
        target_f1=0.945,
        target_count=13,
        max_trials=500
    )

    # 运行连续训练
    successful_models = trainer.run_continuous_training()

    return successful_models


if __name__ == "__main__":
    main()
