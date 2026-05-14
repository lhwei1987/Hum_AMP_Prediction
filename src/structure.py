import ast
import inspect
import os
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import graphviz
from torchviz import make_dot


def parse_model_structure(model_file):
    """
    分析模型的结构，提取类和函数信息
    """
    with open(model_file, "r") as f:
        content = f.read()

    tree = ast.parse(content)
    structure = {
        "classes": [],
        "functions": []
    }

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            structure["classes"].append({
                "name": node.name,
                "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
            })
        elif isinstance(node, ast.FunctionDef):
            structure["functions"].append({
                "name": node.name,
                "args": [arg.arg for arg in node.args.args]
            })

    return structure


def generate_model_diagram(model_instance, input_shape, diagram_path="model_structure.png"):
    """
    生成模型结构图并保存为PNG格式
    """
    try:
        # 创建示例输入
        input_tensor = torch.randn(input_shape)

        # 获取模型输出
        output = model_instance(input_tensor)

        # 生成计算图
        dot = make_dot(output, params=dict(list(model_instance.named_parameters())))
        dot.format = 'png'
        dot.render(diagram_path, cleanup=True)

        return diagram_path + ".png"
    except Exception as e:
        print(f"生成模型结构图时出错: {e}")
        return None


def model_summary_to_text(model, input_shape):
    """
    将模型摘要转换为文本
    """
    buffer = BytesIO()
    summary(model, input_shape, file=buffer)
    buffer.seek(0)
    return buffer.read().decode()


def create_pdf_report(model_file, output_pdf):
    """
    创建包含模型分析的PDF报告
    """
    # 分析模型结构
    structure = parse_model_structure(model_file)

    # 加载模型类定义
    spec = importlib.util.spec_from_file_location("model", model_file)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    # 尝试获取模型类
    model_classes = [cls for cls in inspect.getmembers(model_module, inspect.isclass) if "AMPPredictionModel" in cls[0]]

    if not model_classes:
        print("未找到AMPPredictionModel类定义")
        return

    model_class = model_classes[0][1]
    input_shape = (3, 224, 224)  # 假设输入形状，可根据实际情况修改

    # 创建模型实例所需的配置
    # 这里需要根据你的模型实际需要的配置来调整
    config = {}  # 你的模型配置字典

    # 创建模型实例
    try:
        # 如果模型的 __init__ 方法需要 config 参数
        model = model_class(config)
    except Exception as e:
        print(f"创建模型实例时出错: {e}")
        return

    # 生成模型结构图
    diagram_path = generate_model_diagram(model, input_shape)

    # 创建PDF文档
    doc = SimpleDocTemplate(output_pdf, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # 标题
    elements.append(Paragraph("Multimodal CNN Model Analysis Report", styles['Title']))
    elements.append(Spacer(1, 0.5 * inch))

    # 模型结构分析
    elements.append(Paragraph("Model Structure Analysis", styles['Heading1']))
    elements.append(Spacer(1, 0.25 * inch))

    # 类定义
    elements.append(Paragraph("Classes:", styles['Heading2']))
    for cls in structure["classes"]:
        elements.append(Paragraph(f"- {cls['name']}: Methods = {', '.join(cls['methods'])}", styles['Normal']))

    # 函数定义
    elements.append(Spacer(1, 0.25 * inch))
    elements.append(Paragraph("Functions:", styles['Heading2']))
    for func in structure["functions"]:
        elements.append(Paragraph(f"- {func['name']}({', '.join(func['args'])})", styles['Normal']))

    elements.append(Spacer(1, 0.5 * inch))

    # 模型结构图
    if diagram_path:
        elements.append(Paragraph("Model Architecture Diagram", styles['Heading1']))
        elements.append(Spacer(1, 0.25 * inch))

        # 添加模型结构图
        if os.path.exists(diagram_path):
            img = RLImage(diagram_path, width=7 * inch, height=4 * inch)
            elements.append(img)
        else:
            elements.append(Paragraph("无法生成模型结构图", styles['Normal']))

        elements.append(Spacer(1, 0.5 * inch))

    # 模型参数
    elements.append(Paragraph("Model Parameters", styles['Heading1']))
    elements.append(Spacer(1, 0.25 * inch))

    # 添加模型摘要
    model_summary = model_summary_to_text(model, input_shape)
    elements.append(Paragraph("Model Summary:", styles['Heading2']))
    elements.append(Paragraph(model_summary, styles['Normal']))

    # 保存PDF
    doc.build(elements)

    # 清理临时文件
    if diagram_path and os.path.exists(diagram_path):
        os.remove(diagram_path)


if __name__ == "__main__":
    import importlib.util

    model_file = "models/multimodal_cnn.py"
    output_pdf = "models/multimodal_cnn_analysis.pdf"

    print(f"分析模型文件: {model_file}")
    create_pdf_report(model_file, output_pdf)
    print(f"分析报告已生成: {output_pdf}")