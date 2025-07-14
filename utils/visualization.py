"""
API服务生态系统情景生成 - 可视化工具
提供各种可视化功能，生成environment_boundaries.png等
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, List, Any, Optional, Union, Tuple
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

# 设置中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
except:
    print("警告: 无法设置中文字体，图表中的中文可能无法正确显示")

def plot_environment_boundaries(
    historical_data: Dict[str, Dict[str, float]],
    upper_boundary: Dict[str, float],
    lower_boundary: Dict[str, float],
    output_path: str,
    title: str = "极限环境边界生成情况分析",
    years: List[int] = None,
    monitoring_points: List[Tuple[int, str, float]] = None,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 300
) -> str:
    """
    绘制环境边界生成情况分析图
    
    Args:
        historical_data: 历史数据，格式为 {类别: {年份: 值}}
        upper_boundary: 上限边界，格式为 {类别: 值}
        lower_boundary: 下限边界，格式为 {类别: 值}
        output_path: 输出路径
        title: 图表标题
        years: 年份列表，如果为None则从历史数据中提取
        monitoring_points: 监测点列表，格式为 [(年份, 类别, 值)]
        figsize: 图表大小
        dpi: 图表分辨率
        
    Returns:
        输出文件路径
    """
    # 如果未提供年份，则从历史数据中提取
    if years is None:
        all_years = set()
        for category, data in historical_data.items():
            all_years.update(map(int, data.keys()))
        years = sorted(list(all_years))
    
    # 如果历史数据为空，则报错
    if not historical_data:
        raise ValueError("必须提供有效的历史数据。模拟数据生成功能已被禁用。请使用从programmableweb_2022(1).sql和programmableweb_dataset(1).sql导入的真实数据。")
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    
    # 设置背景色为浅灰色
    ax.set_facecolor('#f8f8f8')
    fig.set_facecolor('white')
    
    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.7, color='#cccccc')
    
    # 定义颜色映射
    colors = {
        "基础设施类": "#3366cc",  # 蓝色
        "生活服务类": "#66cc66",  # 绿色
        "企业管理类": "#cc3333",  # 红色
        "社交娱乐类": "#ff9900",  # 橙色
        "Enterprise Infrastructure": "#3366cc",  # 英文映射
        "Lifestyle Services": "#66cc66",
        "Business Management": "#cc3333",
        "Social & Entertainment": "#ff9900",
        "Social": "#3366cc",  # 添加historical_data.json中的类别
        "Enterprise": "#66cc66",
        "Tools": "#cc3333",
        "Financial": "#ff9900"
    }
    
    # 英文到中文的映射
    en_to_zh = {
        "Enterprise Infrastructure": "基础设施类",
        "Lifestyle Services": "生活服务类",
        "Business Management": "企业管理类",
        "Social & Entertainment": "社交娱乐类",
        "Social": "社交类",  # 添加historical_data.json中的类别映射
        "Enterprise": "企业类",
        "Tools": "工具类",
        "Financial": "金融类"
    }
    
    # 中文到英文的映射
    zh_to_en = {v: k for k, v in en_to_zh.items()}
    
    # 绘制历史趋势和边界
    for category, data in historical_data.items():
        # 获取年份和值
        x = [int(year) for year in data.keys()]
        y = list(data.values())
        
        # 绘制趋势线
        color = colors.get(category, "gray")
        
        # 获取英文类别名称（用于图例）
        en_category = zh_to_en.get(category, category)
        
        # 绘制带有点的虚线
        ax.plot(x, y, marker='o', linestyle=':', color=color, label=en_category, 
                alpha=0.8, markersize=5, linewidth=1.5)
        
        # 绘制环境边界
        if category in upper_boundary and category in lower_boundary:
            upper = upper_boundary[category]
            lower = lower_boundary[category]
            
            # 创建边界区域（使用渐变色填充）
            ax.fill_between(
                x,
                [lower] * len(x),
                [upper] * len(x),
                color=color,
                alpha=0.15
            )
            
            # 绘制边界线
            ax.plot(x, [upper] * len(x), '--', color=color, alpha=0.5, linewidth=1)
            ax.plot(x, [lower] * len(x), '--', color=color, alpha=0.5, linewidth=1)
    
    # 绘制监测点
    if monitoring_points:
        for year, category, value in monitoring_points:
            color = colors.get(category, "gray")
            ax.scatter(year, value, s=100, color=color, edgecolor='black', linewidth=1.5, zorder=5)
    
    # 设置图表标题和标签
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("需求时间线 →", fontsize=12, fontweight='bold')
    ax.set_ylabel("API调用量(百万)", fontsize=12, fontweight='bold')
    
    # 设置坐标轴范围
    ax.set_xlim(min(years) - 0.5, max(years) + 0.5)
    
    # 美化坐标轴
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    # 添加图例框
    legend_bg = plt.Rectangle(
        (0.01, 0.01), 0.25, 0.15, 
        fill=True, color='white', alpha=0.8, 
        transform=ax.transAxes, zorder=5
    )
    ax.add_patch(legend_bg)
    
    # 添加数据特征说明
    ax.text(0.03, 0.13, "Data Features:", transform=ax.transAxes, fontsize=9, fontweight='bold')
    ax.text(0.03, 0.09, "• Original Data Trend", transform=ax.transAxes, fontsize=8)
    ax.text(0.03, 0.06, "- - Environment Bounds", transform=ax.transAxes, fontsize=8)
    ax.text(0.03, 0.03, "• Monitoring Points", transform=ax.transAxes, fontsize=8)
    
    # 添加数据集类型图例
    legend_elements = []
    for category, color in colors.items():
        if category in historical_data or category in en_to_zh and en_to_zh[category] in historical_data:
            legend_elements.append(plt.Line2D([0], [0], marker='', color=color, linestyle='-', 
                                            label=category, linewidth=2))
    
    # 放置图例在右下角
    dataset_legend = ax.legend(
        handles=legend_elements, 
        loc='lower right', 
        title="Dataset Types:", 
        fontsize=8,
        title_fontsize=9,
        framealpha=0.8
    )
    
    # 保存图表
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_environment_boundaries_from_files(
    scenario_file: str,
    historical_data_file: Optional[str] = None,
    output_path: str = None,
    title: str = "极限环境边界生成情况分析",
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 300
) -> str:
    """
    从文件中读取数据并绘制环境边界生成情况分析图
    
    Args:
        scenario_file: 情景文件路径
        historical_data_file: 历史数据文件路径，必须提供有效的文件
        output_path: 输出路径，如果为None则使用默认路径
        title: 图表标题
        figsize: 图表大小
        dpi: 图表分辨率
        
    Returns:
        输出文件路径
    """
    # 读取情景文件
    with open(scenario_file, 'r', encoding='utf-8') as f:
        scenario = json.load(f)
    
    # 提取需求变化
    demand_changes = scenario.get('demand_changes', {})
    
    # 计算上下限边界
    upper_boundary = {}
    lower_boundary = {}
    for category, value in demand_changes.items():
        # 使用需求变化的1.5倍作为上限，0.5倍作为下限
        upper_boundary[category] = value * 1.5 if value > 0 else value * 0.5
        lower_boundary[category] = value * 0.5 if value > 0 else value * 1.5
    
    # 读取历史数据
    historical_data = {}
    if historical_data_file and os.path.exists(historical_data_file):
        try:
            # 尝试多种编码格式
            encodings = ['utf-8', 'utf-16', 'gbk', 'latin1']
            for encoding in encodings:
                try:
                    with open(historical_data_file, 'r', encoding=encoding) as f:
                        historical_data = json.load(f)
                    if historical_data:
                        break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"使用{encoding}编码加载历史数据失败: {str(e)}")
            
            # 如果仍未加载成功，尝试二进制读取
            if not historical_data:
                with open(historical_data_file, 'rb') as f:
                    content = f.read()
                    import chardet
                    detected = chardet.detect(content)
                    encoding = detected.get('encoding')
                    if encoding:
                        historical_data = json.loads(content.decode(encoding))
        except Exception as e:
            raise ValueError(f"加载历史数据文件失败: {str(e)}")
    else:
        raise ValueError("必须提供有效的历史数据文件。模拟数据生成功能已被禁用。请使用从programmableweb_2022(1).sql和programmableweb_dataset(1).sql导入的真实数据。")
    
    # 确保历史数据不为空
    if not historical_data:
        raise ValueError("历史数据为空或加载失败。请检查历史数据文件格式是否正确。")
    
    # 设置默认输出路径
    if output_path is None:
        output_dir = os.path.dirname(scenario_file)
        output_path = os.path.join(output_dir, 'environment_boundaries.png')
    
    # 绘制图表
    return plot_environment_boundaries(
        historical_data=historical_data,
        upper_boundary=upper_boundary,
        lower_boundary=lower_boundary,
        output_path=output_path,
        title=title,
        figsize=figsize,
        dpi=dpi
    )

def load_historical_data(
    historical_data_file: str,
    output_file: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    从文件加载历史数据
    
    Args:
        historical_data_file: 历史数据文件路径
        output_file: 输出文件路径，如果为None则不保存
        
    Returns:
        历史数据，格式为 {类别: {年份: 值}}
    """
    # 检查文件是否存在
    if not os.path.exists(historical_data_file):
        raise FileNotFoundError(f"历史数据文件不存在: {historical_data_file}")
    
    # 尝试多种编码加载数据
    historical_data = None
    encodings = ['utf-8-sig', 'utf-8', 'utf-16', 'gbk', 'latin1', None]
    
    for encoding in encodings:
        try:
            if encoding:
                with open(historical_data_file, 'r', encoding=encoding) as f:
                    historical_data = json.load(f)
            else:
                # 二进制读取，尝试自动检测编码
                with open(historical_data_file, 'rb') as f:
                    content = f.read()
                    import chardet
                    detected = chardet.detect(content)
                    encoding = detected['encoding']
                    if encoding:
                        historical_data = json.loads(content.decode(encoding))
            
            if historical_data:
                break
        except Exception:
            continue
    
    if not historical_data:
        raise ValueError(f"无法加载历史数据文件: {historical_data_file}，请检查文件格式和编码。")
    
    # 如果需要保存到文件
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(historical_data, f, indent=2, ensure_ascii=False)
    
    return historical_data

def plot_multi_charts(
    scenarios: List[Dict[str, Any]],
    historical_data: Dict[str, Dict[str, float]],
    output_path: str,
    title: str = "API服务生态系统情景分析",
    years: List[int] = None,
    figsize: Tuple[int, int] = (18, 12),
    dpi: int = 300
) -> str:
    """
    绘制多个图表并排显示
    
    Args:
        scenarios: 情景列表，每个情景包含需求变化等信息
        historical_data: 历史数据，格式为 {类别: {年份: 值}}
        output_path: 输出路径
        title: 图表标题
        years: 年份列表，如果为None则从历史数据中提取
        figsize: 图表大小
        dpi: 图表分辨率
        
    Returns:
        输出文件路径
    """
    # 如果未提供年份，则从历史数据中提取
    if years is None:
        all_years = set()
        for category, data in historical_data.items():
            all_years.update(map(int, data.keys()))
        years = sorted(list(all_years))
    
    # 创建图表
    fig = plt.figure(figsize=figsize)
    
    # 创建网格布局
    n_scenarios = len(scenarios)
    n_cols = min(3, n_scenarios)  # 最多3列
    n_rows = (n_scenarios + n_cols - 1) // n_cols  # 计算行数
    
    # 创建GridSpec
    gs = GridSpec(n_rows + 1, n_cols, height_ratios=[1] * n_rows + [0.1])
    
    # 定义颜色映射
    colors = {
        "基础设施类": "#3366cc",  # 蓝色
        "生活服务类": "#66cc66",  # 绿色
        "企业管理类": "#cc3333",  # 红色
        "社交娱乐类": "#ff9900",  # 橙色
        "Enterprise Infrastructure": "#3366cc",  # 英文映射
        "Lifestyle Services": "#66cc66",
        "Business Management": "#cc3333",
        "Social & Entertainment": "#ff9900",
        "Social": "#3366cc",  # 添加historical_data.json中的类别
        "Enterprise": "#66cc66",
        "Tools": "#cc3333",
        "Financial": "#ff9900"
    }
    
    # 英文到中文的映射
    en_to_zh = {
        "Enterprise Infrastructure": "基础设施类",
        "Lifestyle Services": "生活服务类",
        "Business Management": "企业管理类",
        "Social & Entertainment": "社交娱乐类",
        "Social": "社交类",  # 添加historical_data.json中的类别映射
        "Enterprise": "企业类",
        "Tools": "工具类",
        "Financial": "金融类"
    }
    
    # 中文到英文的映射
    zh_to_en = {v: k for k, v in en_to_zh.items()}
    
    # 绘制每个情景的图表
    for i, scenario in enumerate(scenarios):
        row = i // n_cols
        col = i % n_cols
        
        # 创建子图
        ax = fig.add_subplot(gs[row, col])
        
        # 设置背景色为浅灰色
        ax.set_facecolor('#f8f8f8')
        
        # 添加网格线
        ax.grid(True, linestyle='--', alpha=0.7, color='#cccccc')
        
        # 提取需求变化
        demand_changes = scenario.get('demand_changes', {})
        
        # 计算上下限边界
        upper_boundary = {}
        lower_boundary = {}
        for category, value in demand_changes.items():
            # 使用需求变化的1.5倍作为上限，0.5倍作为下限
            upper_boundary[category] = value * 1.5 if value > 0 else value * 0.5
            lower_boundary[category] = value * 0.5 if value > 0 else value * 1.5
        
        # 绘制历史趋势和边界
        for category, data in historical_data.items():
            # 获取年份和值
            x = [int(year) for year in data.keys()]
            y = list(data.values())
            
            # 绘制趋势线
            color = colors.get(category, "gray")
            
            # 获取英文类别名称（用于图例）
            en_category = zh_to_en.get(category, category)
            
            # 绘制带有点的虚线
            ax.plot(x, y, marker='o', linestyle=':', color=color, label=en_category, 
                    alpha=0.8, markersize=3, linewidth=1)
            
            # 绘制环境边界
            if category in upper_boundary and category in lower_boundary:
                upper = upper_boundary[category]
                lower = lower_boundary[category]
                
                # 创建边界区域（使用渐变色填充）
                ax.fill_between(
                    x,
                    [lower] * len(x),
                    [upper] * len(x),
                    color=color,
                    alpha=0.15
                )
                
                # 绘制边界线
                ax.plot(x, [upper] * len(x), '--', color=color, alpha=0.5, linewidth=1)
                ax.plot(x, [lower] * len(x), '--', color=color, alpha=0.5, linewidth=1)
        
        # 设置子图标题
        scenario_title = scenario.get('title', f'情景 {i+1}')
        probability = scenario.get('probability', 0) * 100
        ax.set_title(f"{scenario_title} (概率: {probability:.1f}%)", fontsize=12, fontweight='bold')
        
        # 设置坐标轴标签
        if row == n_rows - 1:
            ax.set_xlabel("需求时间线", fontsize=10)
        if col == 0:
            ax.set_ylabel("API调用量(百万)", fontsize=10)
        
        # 设置坐标轴范围
        ax.set_xlim(min(years) - 0.5, max(years) + 0.5)
        
        # 美化坐标轴
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        
        # 添加情景描述
        description = scenario.get('description', '')
        if description:
            # 截断描述，保留前100个字符
            short_desc = description[:100] + '...' if len(description) > 100 else description
            ax.text(0.02, 0.02, short_desc, transform=ax.transAxes, fontsize=8, 
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # 创建共享图例
    legend_ax = fig.add_subplot(gs[n_rows, :])
    legend_ax.axis('off')
    
    # 添加数据集类型图例
    legend_elements = []
    for category, color in colors.items():
        if category in historical_data or category in en_to_zh and en_to_zh[category] in historical_data:
            legend_elements.append(plt.Line2D([0], [0], marker='', color=color, linestyle='-', 
                                            label=category, linewidth=2))
    
    # 添加数据特征图例
    feature_elements = [
        plt.Line2D([0], [0], marker='o', color='gray', linestyle=':', label='原始数据趋势', 
                  markerfacecolor='gray', markersize=5),
        plt.Line2D([0], [0], marker='', color='gray', linestyle='--', label='环境边界', alpha=0.5),
    ]
    
    # 合并图例元素
    all_elements = feature_elements + legend_elements
    
    # 放置图例在底部中央
    legend = legend_ax.legend(
        handles=all_elements, 
        loc='center', 
        ncol=len(all_elements), 
        fontsize=10,
        framealpha=0.8
    )
    
    # 设置总标题
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图表
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_scenario_comparison(
    scenarios: List[Dict[str, Any]],
    output_path: str,
    title: str = "极端情景需求变化对比",
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 300
) -> str:
    """
    绘制情景需求变化对比图表
    
    Args:
        scenarios: 情景列表，每个情景包含需求变化等信息
        output_path: 输出路径
        title: 图表标题
        figsize: 图表大小
        dpi: 图表分辨率
        
    Returns:
        输出文件路径
    """
    # 定义颜色映射
    colors = {
        "基础设施类": "#3366cc",  # 蓝色
        "生活服务类": "#66cc66",  # 绿色
        "企业管理类": "#cc3333",  # 红色
        "社交娱乐类": "#ff9900",  # 橙色
        "Enterprise Infrastructure": "#3366cc",  # 英文映射
        "Lifestyle Services": "#66cc66",
        "Business Management": "#cc3333",
        "Social & Entertainment": "#ff9900",
        "Social": "#3366cc",  # 添加historical_data.json中的类别
        "Enterprise": "#66cc66",
        "Tools": "#cc3333",
        "Financial": "#ff9900"
    }
    
    # 英文到中文的映射
    en_to_zh = {
        "Enterprise Infrastructure": "基础设施类",
        "Lifestyle Services": "生活服务类",
        "Business Management": "企业管理类",
        "Social & Entertainment": "社交娱乐类",
        "Social": "社交类",  # 添加historical_data.json中的类别映射
        "Enterprise": "企业类",
        "Tools": "工具类",
        "Financial": "金融类"
    }
    
    # 中文到英文的映射
    zh_to_en = {v: k for k, v in en_to_zh.items()}
    
    # 提取所有类别
    all_categories = set()
    for scenario in scenarios:
        demand_changes = scenario.get('demand_changes', {})
        all_categories.update(demand_changes.keys())
    
    all_categories = sorted(list(all_categories))
    
    # 提取需求变化数据
    data = []
    scenario_names = []
    for scenario in scenarios:
        demand_changes = scenario.get('demand_changes', {})
        scenario_names.append(scenario.get('title', '未命名情景'))
        
        # 提取每个类别的需求变化
        row = []
        for category in all_categories:
            row.append(demand_changes.get(category, 0))
        
        data.append(row)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    
    # 设置背景色为浅灰色
    ax.set_facecolor('#f8f8f8')
    fig.set_facecolor('white')
    
    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.7, color='#cccccc', axis='y')
    
    # 设置柱状图的位置
    x = np.arange(len(all_categories))
    width = 0.8 / len(scenarios)
    
    # 绘制柱状图
    for i, (row, name) in enumerate(zip(data, scenario_names)):
        offset = width * i - width * len(scenarios) / 2 + width / 2
        bars = ax.bar(x + offset, row, width, label=name)
        
        # 在柱状图上添加数值标签
        for bar in bars:
            height = bar.get_height()
            if height >= 0:
                va = 'bottom'
                y_offset = 3
            else:
                va = 'top'
                y_offset = -12
            
            ax.text(
                bar.get_x() + bar.get_width() / 2, height + y_offset,
                f'{height:.1f}%', ha='center', va=va, fontsize=8, rotation=90
            )
    
    # 设置坐标轴标签
    ax.set_xlabel('API类别', fontsize=12, fontweight='bold')
    ax.set_ylabel('需求变化百分比', fontsize=12, fontweight='bold')
    
    # 设置x轴刻度标签
    ax.set_xticks(x)
    ax.set_xticklabels(all_categories)
    
    # 添加零线
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 美化坐标轴
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    # 设置标题
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # 添加图例
    ax.legend(loc='upper right', fontsize=10)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return output_path


def generate_sample_historical_data():
    return None