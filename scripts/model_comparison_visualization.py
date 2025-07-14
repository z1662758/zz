"""
API服务生态系统情景生成 - 模型规模对比可视化脚本
用于生成不同规模大模型在API生态系统情景生成中的表现差异分析图表
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(str(ROOT_DIR))

from utils.logger import get_default_logger
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Any, Optional, Union, Tuple


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='生成模型规模对比分析图表')
    parser.add_argument('--model_data_file', type=str, required=True,
                        help='模型对比数据文件路径')
    parser.add_argument('--historical_data_file', type=str, required=True,
                        help='历史数据文件路径')
    parser.add_argument('--output_file', type=str, default=None,
                        help='输出文件路径，如果为None，则使用默认路径')
    parser.add_argument('--title', type=str, default='大模型规模对比分析',
                        help='图表标题')
    parser.add_argument('--width', type=int, default=16,
                        help='图表宽度')
    parser.add_argument('--height', type=int, default=12,
                        help='图表高度')
    parser.add_argument('--dpi', type=int, default=300,
                        help='图表分辨率')
    return parser.parse_args()


def plot_model_comparison_boundaries(
    historical_data: Dict[str, Dict[str, float]],
    model_data: Dict[str, Any],
    output_path: str,
    title: str = "大模型规模对比分析",
    figsize: Tuple[int, int] = (16, 12),
    dpi: int = 300
) -> str:
    """
    绘制不同规模模型的环境边界对比图
    
    Args:
        historical_data: 历史数据，格式为 {类别: {年份: 值}}
        model_data: 模型对比数据
        output_path: 输出路径
        title: 图表标题
        figsize: 图表大小
        dpi: 图表分辨率
        
    Returns:
        输出文件路径
    """
    # 创建图表
    fig = plt.figure(figsize=figsize)
    
    # 创建GridSpec
    gs = GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[2, 1])
    
    # 设置中文字体支持
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    except:
        print("警告: 无法设置中文字体，图表中的中文可能无法正确显示")
    
    # 提取模型数据
    models = model_data.get("models", {})
    model_names = list(models.keys())
    
    # 定义颜色映射
    colors = {
        "small": "#3366cc",    # 蓝色
        "medium": "#66cc66",   # 绿色
        "large": "#cc3333",    # 红色
    }
    
    # 提取年份
    all_years = set()
    for category, data in historical_data.items():
        all_years.update(map(int, data.keys()))
    years = sorted(list(all_years))
    
    # 1. 绘制环境边界对比图
    ax1 = fig.add_subplot(gs[0, :])
    
    # 设置背景色为浅灰色
    ax1.set_facecolor('#f8f8f8')
    
    # 添加网格线
    ax1.grid(True, linestyle='--', alpha=0.7, color='#cccccc')
    
    # 绘制历史趋势
    for category, data in historical_data.items():
        # 获取年份和值
        x = [int(year) for year in data.keys()]
        y = list(data.values())
        
        # 绘制带有点的虚线
        ax1.plot(x, y, marker='o', linestyle=':', color='gray', label=f"{category} 历史数据", 
                alpha=0.5, markersize=4, linewidth=1)
    
    # 绘制不同模型的边界
    for model_id, model_info in models.items():
        boundaries = model_info.get("boundaries", {})
        model_name = model_info.get("name", model_id)
        
        for category, boundary in boundaries.items():
            upper = boundary.get("upper", 0)
            lower = boundary.get("lower", 0)
            
            # 创建边界区域（使用渐变色填充）
            ax1.fill_between(
                years,
                [lower] * len(years),
                [upper] * len(years),
                color=colors.get(model_id, "gray"),
                alpha=0.15,
                label=f"{model_name} - {category}" if category == list(boundaries.keys())[0] else ""
            )
            
            # 绘制边界线
            ax1.plot(years, [upper] * len(years), '--', color=colors.get(model_id, "gray"), alpha=0.5, linewidth=1)
            ax1.plot(years, [lower] * len(years), '--', color=colors.get(model_id, "gray"), alpha=0.5, linewidth=1)
    
    # 设置图表标题和标签
    ax1.set_title("不同规模模型的环境边界对比", fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel("需求时间线 →", fontsize=12, fontweight='bold')
    ax1.set_ylabel("API调用量(百万)", fontsize=12, fontweight='bold')
    
    # 设置坐标轴范围
    ax1.set_xlim(min(years) - 0.5, max(years) + 0.5)
    
    # 美化坐标轴
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 添加图例
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=3, frameon=False)
    
    # 2. 绘制模型性能指标雷达图
    ax2 = fig.add_subplot(gs[1, 0], polar=True)
    
    # 提取性能指标
    metrics = ["accuracy", "coverage", "efficiency", "hallucination_rate", "coherence"]
    metrics_zh = ["准确性", "覆盖率", "效率", "幻觉率", "一致性"]
    
    # 计算角度
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    # 绘制雷达图
    ax2.set_theta_offset(np.pi / 2)
    ax2.set_theta_direction(-1)
    
    # 绘制轴线
    plt.xticks(angles[:-1], metrics_zh)
    
    # 绘制每个模型的雷达图
    for model_id, model_info in models.items():
        model_metrics = model_info.get("metrics", {})
        values = [model_metrics.get(metric, 0) for metric in metrics]
        values += values[:1]  # 闭合
        
        # 绘制雷达图
        ax2.plot(angles, values, 'o-', linewidth=2, label=model_info.get("name", model_id), color=colors.get(model_id, "gray"))
        ax2.fill(angles, values, alpha=0.1, color=colors.get(model_id, "gray"))
    
    # 设置雷达图标题
    ax2.set_title("模型性能指标对比", fontsize=14, fontweight='bold', pad=15)
    
    # 添加图例
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    # 3. 绘制模型幻觉率与边界宽度关系图
    ax3 = fig.add_subplot(gs[1, 1])
    
    # 设置背景色为浅灰色
    ax3.set_facecolor('#f8f8f8')
    
    # 添加网格线
    ax3.grid(True, linestyle='--', alpha=0.7, color='#cccccc')
    
    # 计算每个模型的平均边界宽度
    boundary_widths = []
    hallucination_rates = []
    model_labels = []
    
    for model_id, model_info in models.items():
        boundaries = model_info.get("boundaries", {})
        
        # 计算平均边界宽度
        total_width = 0
        for category, boundary in boundaries.items():
            upper = boundary.get("upper", 0)
            lower = boundary.get("lower", 0)
            total_width += (upper - lower)
        
        avg_width = total_width / len(boundaries) if boundaries else 0
        boundary_widths.append(avg_width)
        
        # 获取幻觉率
        hallucination_rate = model_info.get("metrics", {}).get("hallucination_rate", 0)
        hallucination_rates.append(hallucination_rate)
        
        # 获取模型名称
        model_labels.append(model_info.get("name", model_id))
    
    # 绘制散点图
    for i, (width, rate, label) in enumerate(zip(boundary_widths, hallucination_rates, model_labels)):
        ax3.scatter(rate, width, s=100, color=colors.get(model_names[i], "gray"), label=label)
        
        # 添加标签
        ax3.annotate(
            label,
            (rate, width),
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    # 拟合趋势线
    if len(hallucination_rates) > 1:
        z = np.polyfit(hallucination_rates, boundary_widths, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(hallucination_rates) * 0.9, max(hallucination_rates) * 1.1, 100)
        ax3.plot(x_trend, p(x_trend), "--", color='gray', alpha=0.7)
    
    # 设置图表标题和标签
    ax3.set_title("模型幻觉率与边界宽度关系", fontsize=14, fontweight='bold', pad=15)
    ax3.set_xlabel("幻觉率", fontsize=12, fontweight='bold')
    ax3.set_ylabel("平均边界宽度", fontsize=12, fontweight='bold')
    
    # 美化坐标轴
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # 设置图表标题
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 保存图表
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return output_path


def main():
    """主函数"""
    args = parse_args()
    logger = get_default_logger()
    logger.info("启动模型规模对比可视化脚本")
    
    # 设置默认输出路径
    if args.output_file is None:
        output_dir = os.path.join(ROOT_DIR, "output", "model_comparison")
        os.makedirs(output_dir, exist_ok=True)
        args.output_file = os.path.join(output_dir, "model_comparison.png")
    
    # 检查模型数据文件
    if not os.path.exists(args.model_data_file):
        logger.error(f"模型数据文件不存在: {args.model_data_file}")
        return
    
    # 检查历史数据文件
    if not os.path.exists(args.historical_data_file):
        logger.error(f"历史数据文件不存在: {args.historical_data_file}")
        return
    
    # 读取模型数据
    try:
        # 尝试多种编码格式
        encodings = ['utf-8-sig', 'utf-8', 'utf-16', 'gbk', 'latin1']
        for encoding in encodings:
            try:
                with open(args.model_data_file, 'r', encoding=encoding) as f:
                    model_data = json.load(f)
                logger.info(f"成功使用 {encoding} 编码加载模型数据")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"使用 {encoding} 编码加载模型数据失败: {str(e)}")
                continue
        
        # 如果仍未加载成功，尝试二进制读取
        if 'model_data' not in locals():
            with open(args.model_data_file, 'rb') as f:
                content = f.read()
                import chardet
                detected = chardet.detect(content)
                encoding = detected.get('encoding')
                if encoding:
                    model_data = json.loads(content.decode(encoding))
                    logger.info(f"成功使用自动检测的 {encoding} 编码加载模型数据")
    except Exception as e:
        logger.error(f"读取模型数据文件失败: {str(e)}")
        return
    
    # 读取历史数据
    try:
        # 尝试多种编码格式
        encodings = ['utf-8-sig', 'utf-8', 'utf-16', 'gbk', 'latin1']
        for encoding in encodings:
            try:
                with open(args.historical_data_file, 'r', encoding=encoding) as f:
                    historical_data = json.load(f)
                logger.info(f"成功使用 {encoding} 编码加载历史数据")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"使用 {encoding} 编码加载历史数据失败: {str(e)}")
                continue
        
        # 如果仍未加载成功，尝试二进制读取
        if 'historical_data' not in locals():
            with open(args.historical_data_file, 'rb') as f:
                content = f.read()
                import chardet
                detected = chardet.detect(content)
                encoding = detected.get('encoding')
                if encoding:
                    historical_data = json.loads(content.decode(encoding))
                    logger.info(f"成功使用自动检测的 {encoding} 编码加载历史数据")
    except Exception as e:
        logger.error(f"读取历史数据文件失败: {str(e)}")
        return
    
    # 绘制图表
    logger.info(f"绘制模型规模对比图表: {args.output_file}")
    output_path = plot_model_comparison_boundaries(
        historical_data=historical_data,
        model_data=model_data,
        output_path=args.output_file,
        title=args.title,
        figsize=(args.width, args.height),
        dpi=args.dpi
    )
    
    logger.info(f"图表已生成: {output_path}")


if __name__ == "__main__":
    main() 