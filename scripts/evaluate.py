import os
import yaml
import torch
import numpy as np
import time
import pandas as pd
import pickle
from scipy.spatial import cKDTree
from src.data_loader import create_data_loader
from src.preprocessing import load_dataset
from src.utils import calculate_metrics, plot_roc_curve, plot_pr_curve, plot_confusion_matrix
from models.multimodal_cnn import AMPPredictionModel
from src.utils import progress_bar, time_estimator, Tokenizer, generate_sequence_indices


@time_estimator
def evaluate_model(config_path, model_path, test_data_path=None, threshold=0.5):
    """评估模型性能
    Args:
        threshold: 预测阈值，默认为0.5
    """
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # 创建 tokenizer 并添加到 config
    tokenizer = Tokenizer()
    config['data']['tokenizer'] = tokenizer
    # 加载测试数据
    if test_data_path is None:
        test_data_path = config['data']['test_path']
    print(f"[{time.strftime('%H:%M:%S')}] Loading test data...")
    test_sequences, test_labels, test_properties = load_dataset(
        test_data_path,
        max_len=config['data']['max_seq_len']
    )
    # 生成测试集序列索引
    print(f"[{time.strftime('%H:%M:%S')}] 生成序列索引...")
    test_seq_indices = generate_sequence_indices(test_sequences, test_labels)
    # 创建数据加载器
    test_loader = create_data_loader(
        sequences=test_sequences,
        labels=test_labels,
        properties=test_properties,
        config=config,
        shuffle=False
    )
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AMPPredictionModel(config['model']).to(device)
    model.eval()
    # 加载模型权重
    try:
        model.load_state_dict(torch.load(model_path), strict=True)
    except RuntimeError as e:
        print(f"严格加载失败: {e}")
        print("尝试宽松加载...")
        model.load_state_dict(torch.load(model_path), strict=False)
        print("宽松加载成功 - 部分参数未加载")
    # 收集预测结果
    print(f"[{time.strftime('%H:%M:%S')}] Running evaluation...")
    all_labels = []
    all_preds = []
    all_probs = []
    all_sequences = []
    eval_bar = progress_bar(test_loader, desc="Evaluating", unit="batch")
    with torch.no_grad():
        for batch_data in eval_bar:
            # 从批次数据中提取
            seq = batch_data['seq_ids'].to(device)
            labels = batch_data['labels'].to(device)
            props = batch_data['properties'].to(device)
            orig_sequences = batch_data['sequence_strs']
            # 模型前向传播
            outputs = model(seq, props, orig_sequences)
            # 转换为numpy数组
            outputs_np = outputs.detach().cpu().numpy().flatten()
            labels_np = labels.cpu().numpy().flatten()
            all_labels.extend(labels_np)
            all_preds.extend(outputs_np > threshold)  # 使用传入的阈值
            all_probs.extend(outputs_np)
            all_sequences.extend(orig_sequences)
            # 更新进度条
            current_preds = (outputs_np > threshold).astype(np.float32)
            current_acc = np.mean(current_preds == labels_np)
            eval_bar.set_postfix({"current_acc": f"{current_acc:.4f}"})
    # 计算评估指标
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    # 打印评估结果
    print(f"\nTest Metrics (Threshold={threshold}):")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"MCC: {metrics['mcc']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"PR AUC: {metrics['pr_auc']:.4f}")
    # 创建结果目录
    result_dir = os.path.join(os.path.dirname(model_path), "..", "evaluationResults")
    os.makedirs(result_dir, exist_ok=True)

    # 创建带阈值的子目录
    threshold_dir = os.path.join(result_dir, f"threshold_{threshold}")
    os.makedirs(threshold_dir, exist_ok=True)
    # 保存可视化结果（添加阈值到文件名）
    plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['roc_auc'],
                   os.path.join(threshold_dir, f"test_roc_curve.png"))

    plot_pr_curve(metrics['precision_curve'], metrics['recall_curve'], metrics['pr_auc'],
                  os.path.join(threshold_dir, f"test_pr_curve.png"))

    plot_confusion_matrix(metrics['confusion_matrix'],
                          os.path.join(threshold_dir, f"test_confusion_matrix.png"))
    # 保存评估结果
    metrics_file = os.path.join(threshold_dir, "test_metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write(f"Test Metrics (Threshold={threshold}):\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n")
        f.write(f"Specificity: {metrics['specificity']:.4f}\n")
        f.write(f"MCC: {metrics['mcc']:.4f}\n")
        f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n")
        f.write(f"PR AUC: {metrics['pr_auc']:.4f}\n")
    # 保存预测结果和序列
    results_df = pd.DataFrame({
        "Sequence": all_sequences,
        "Label": all_labels,
        "Prediction": all_preds.astype(int),
        "Probability": all_probs
    })
    csv_file = os.path.join(threshold_dir, "detailed_predictions.csv")
    results_df.to_csv(csv_file, index=False)
    # 检查文件是否保存成功
    if os.path.exists(metrics_file) and os.path.exists(csv_file):
        print(f"✅ 评估结果成功保存至: {threshold_dir}")
        print(f"   - 指标文件: {metrics_file}")
        print(f"   - 详细结果: {csv_file}")
    else:
        print(f"❌ 文件保存失败! 请检查目录权限: {threshold_dir}")
    return metrics


@time_estimator
def evaluate_with_zero_labels(config_path, model_path, threshold=0.5):  # 默认阈值改为0.5
    """评估模型性能（预测时使用零标签，但计算指标时使用真实标签）"""
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 创建 tokenizer 并添加到 config
    tokenizer = Tokenizer()
    config['data']['tokenizer'] = tokenizer

    # 加载测试数据并保留原始标签
    test_data_path = config['data']['test_path']
    print(f"[{time.strftime('%H:%M:%S')}] 加载测试数据（保留原始标签）...")
    test_sequences, test_labels, test_properties = load_dataset(
        test_data_path,
        max_len=config['data']['max_seq_len']
    )

    # 创建序列到索引的映射
    sequence_to_index = {seq: i for i, seq in enumerate(test_sequences)}

    # 创建数据加载器（使用零标签）
    print(f"[{time.strftime('%H:%M:%S')}] 创建零标签数据加载器...")
    zero_labels = np.zeros_like(test_labels)

    test_loader = create_data_loader(
        sequences=test_sequences,
        labels=zero_labels,
        properties=test_properties,
        config=config,
        shuffle=False
    )

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AMPPredictionModel(config['model']).to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 收集预测结果
    print(f"[{time.strftime('%H:%M:%S')}] 运行零标签预测（使用真实标签计算指标）...")
    print(f"使用预测阈值: {threshold}")
    all_preds = []
    all_probs = []
    all_sequences = []
    processed_indices = []

    eval_bar = progress_bar(test_loader, desc="Zero-Label Evaluating", unit="batch")

    with torch.no_grad():
        for batch_data in eval_bar:
            # 从批次数据中提取
            seq = batch_data['seq_ids'].to(device)
            props = batch_data['properties'].to(device)
            orig_sequences = batch_data['sequence_strs']

            # 获取当前批次中每个序列的原始索引
            batch_indices = [sequence_to_index[seq_str] for seq_str in orig_sequences]
            processed_indices.extend(batch_indices)

            # 模型前向传播
            outputs = model(seq, props, orig_sequences)

            # 转换为numpy数组
            outputs_np = outputs.detach().cpu().numpy().flatten()

            # 使用阈值进行二值化
            all_preds.extend(outputs_np > threshold)
            all_probs.extend(outputs_np)
            all_sequences.extend(orig_sequences)

    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # 使用原始真实标签计算评估指标
    print(f"[{time.strftime('%H:%M:%S')}] 使用原始真实标签计算评估指标...")
    print(f"原始标签数量: {len(test_labels)}, 预测结果数量: {len(all_preds)}")

    # 获取已处理样本的真实标签
    filtered_labels = test_labels[processed_indices]

    # 确保数据一致性
    assert len(filtered_labels) == len(all_preds), (
        f"标签数量 ({len(filtered_labels)}) 与预测数量 ({len(all_preds)}) 不一致"
    )

    metrics = calculate_metrics(filtered_labels, all_preds, all_probs)

    # 打印评估结果
    print(f"\n零标签预测 + 真实标签评估指标 (阈值={threshold}):")
    print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
    print(f"精确率 (Precision): {metrics['precision']:.4f}")
    print(f"召回率 (Recall): {metrics['recall']:.4f}")
    print(f"F1 分数: {metrics['f1']:.4f}")
    print(f"特异性 (Specificity): {metrics['specificity']:.4f}")
    print(f"MCC: {metrics['mcc']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"PR AUC: {metrics['pr_auc']:.4f}")
    print(f"假阳性率 (FPR): {1 - metrics['specificity']:.4f}")

    # 创建结果目录
    result_dir = os.path.join(os.path.dirname(model_path), "..", "evaluationResults")
    os.makedirs(result_dir, exist_ok=True)

    # 创建零标签评估子目录
    zero_label_dir = os.path.join(result_dir, "zero_label_evaluation")
    os.makedirs(zero_label_dir, exist_ok=True)

    # 保存可视化结果 - 修复：移除title参数
    plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['roc_auc'],
                   os.path.join(zero_label_dir, f"zero_label_roc_curve_threshold_{threshold}.png"))

    plot_pr_curve(metrics['precision_curve'], metrics['recall_curve'], metrics['pr_auc'],
                  os.path.join(zero_label_dir, f"zero_label_pr_curve_threshold_{threshold}.png"))

    plot_confusion_matrix(metrics['confusion_matrix'],
                          os.path.join(zero_label_dir, f"zero_label_confusion_matrix_threshold_{threshold}.png"))

    # 保存评估结果
    metrics_file = os.path.join(zero_label_dir, f"zero_label_pred_real_eval_metrics_threshold_{threshold}.txt")
    with open(metrics_file, 'w') as f:
        f.write(f"零标签预测 + 真实标签评估指标 (阈值={threshold}):\n")
        f.write(f"准确率 (Accuracy): {metrics['accuracy']:.4f}\n")
        f.write(f"精确率 (Precision): {metrics['precision']:.4f}\n")
        f.write(f"召回率 (Recall): {metrics['recall']:.4f}\n")
        f.write(f"F1 分数: {metrics['f1']:.4f}\n")
        f.write(f"特异性 (Specificity): {metrics['specificity']:.4f}\n")
        f.write(f"MCC: {metrics['mcc']:.4f}\n")
        f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n")
        f.write(f"PR AUC: {metrics['pr_auc']:.4f}\n")
        f.write(f"假阳性率 (FPR): {1 - metrics['specificity']:.4f}\n")

    # 保存预测结果和序列（包含原始真实标签）
    results_df = pd.DataFrame({
        "Sequence": all_sequences,
        "True_Label": filtered_labels,
        "Prediction": all_preds.astype(int),
        "Probability": all_probs,
        "Original_Index": processed_indices
    })
    csv_file = os.path.join(zero_label_dir, f"zero_label_pred_real_eval_detailed_threshold_{threshold}.csv")
    results_df.to_csv(csv_file, index=False)

    # 检查文件是否保存成功
    if os.path.exists(metrics_file) and os.path.exists(csv_file):
        print(f"✅ 评估结果成功保存至: {zero_label_dir}")
        print(f"   - 指标文件: {metrics_file}")
        print(f"   - 详细结果: {csv_file}")
    else:
        print(f"❌ 文件保存失败! 请检查目录权限: {zero_label_dir}")

    return metrics


if __name__ == "__main__":
    # 配置文件路径
    config_path = "checkpoints/config.yaml"
    # 模型路径
    model_path = "checkpoints/best_model.pth"


    # 第一轮评估：使用原始测试数据
    print("\n" + "=" * 50)
    print("第一轮评估：原始测试数据")
    print("=" * 50)
    evaluate_model(config_path, model_path, threshold=0.5)  # 使用默认阈值0.5
    # 第二轮评估：将所有标签设为0
    print("\n" + "=" * 50)
    print("第二轮评估：所有标签设为0")
    print("=" * 50)
    evaluate_with_zero_labels(config_path, model_path, threshold=1)