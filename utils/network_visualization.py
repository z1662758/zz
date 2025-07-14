"""
API服务生态系统情景生成 - 网络可视化工具
提供网络可视化相关功能
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
from typing import Dict, List, Any, Optional, Union, Tuple
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

# 设置中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
except:
    print("警告: 无法设置中文字体，图表中的中文可能无法正确显示")

def plot_network_comparison(
    networks: Dict[str, Dict[str, Any]],
    methods: List[str],
    years: List[int],
    output_path: str,
    title: str = "API关系骨架计算实验设计框架",
    figsize: Tuple[int, int] = (18, 14),
    dpi: int = 300,
    node_size: int = 150,  # 增加节点大小
    edge_width: float = 1.0,  # 增加边宽度
    highlight_best: bool = True
) -> str:
    """
    绘制不同方法和不同年份的API关系网络对比图
    
    Args:
        networks: 网络数据，格式为 {年份: {方法: 网络数据}}
        methods: 方法列表
        years: 年份列表
        output_path: 输出路径
        title: 图表标题
        figsize: 图表大小
        dpi: 图表分辨率
        node_size: 节点大小
        edge_width: 边宽度
        highlight_best: 是否高亮最佳方法
        
    Returns:
        输出文件路径
    """
    # 创建图表
    fig = plt.figure(figsize=figsize)
    
    # 设置网格布局
    n_rows = len(years)
    n_cols = len(methods)
    
    # 创建GridSpec
    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.1, hspace=0.3)
    
    # 定义颜色映射 - 不同类别的API使用不同颜色
    category_colors = {
        "基础设施类": "#3366cc",  # 蓝色
        "生活服务类": "#66cc66",  # 绿色
        "企业管理类": "#cc3333",  # 红色
        "社交娱乐类": "#ff9900",  # 橙色
        "其他": "#999999"  # 灰色
    }
    
    # 绘制每个子图
    for i, year in enumerate(years):
        for j, method in enumerate(methods):
            # 创建子图
            ax = plt.subplot(gs[i, j])
            
            # 获取网络数据
            network_data = networks.get(str(year), {}).get(method, None)
            
            if network_data:
                # 创建网络图
                G = nx.Graph()
                
                # 添加节点
                for node in network_data.get('nodes', []):
                    node_id = node.get('id', '')
                    category = node.get('category', '其他')
                    G.add_node(node_id, category=category)
                
                # 添加边
                for edge in network_data.get('edges', []):
                    source = edge.get('source', '')
                    target = edge.get('target', '')
                    weight = edge.get('weight', 1.0)
                    G.add_edge(source, target, weight=weight)
                
                # 固定每种方法每个年份的种子值，确保相同的布局
                seed_val = i * 100 + j * 10 + 42
                
                # 如果节点数量少于10，使用圆形布局
                if len(G.nodes) < 10:
                    pos = nx.circular_layout(G)
                else:
                    # 使用spring布局，增加k值让节点分布更开
                    pos = nx.spring_layout(G, seed=seed_val, k=0.5)
                
                # 绘制边
                nx.draw_networkx_edges(
                    G, pos, 
                    alpha=0.5, 
                    width=edge_width,
                    edge_color='gray'
                )
                
                # 按类别绘制节点
                for category, color in category_colors.items():
                    node_list = [node for node, data in G.nodes(data=True) if data.get('category', '其他') == category]
                    nx.draw_networkx_nodes(
                        G, pos, 
                        nodelist=node_list,
                        node_size=node_size, 
                        node_color=color, 
                        alpha=0.8
                    )
            
            # 设置子图标题
            if i == 0:
                ax.set_title(method, fontsize=12, fontweight='bold')
            
            # 设置年份标签
            if j == 0:
                ax.text(-0.1, 0.5, f"API关系骨架 - {year}", 
                         transform=ax.transAxes, 
                         fontsize=12, fontweight='bold',
                         rotation=90, ha='center', va='center')
            
            # 如果是最佳方法，添加对勾标记
            if highlight_best and method == "Ours":
                # 添加对勾标记
                ax.text(0.95, 0.05, "✓", 
                        transform=ax.transAxes,
                        fontsize=24, color='#00aa00',
                        ha='right', va='bottom',
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='circle'))
            
            # 隐藏坐标轴
            ax.axis('off')
    
    # 添加图例
    legend_elements = []
    for category, color in category_colors.items():
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=color, markersize=8, 
                                         label=category))
    
    # 添加图例到图表底部
    fig.legend(handles=legend_elements, loc='lower center', 
               ncol=len(category_colors), fontsize=10, 
               bbox_to_anchor=(0.5, 0.02))
    
    # 添加总标题
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # 添加说明文本
    description = "输出一个情景集以及评估指标（包括对生成情景的质量评估和实验效率评估）"
    fig.text(0.5, 0.01, description, ha='center', va='bottom', fontsize=10, style='italic')
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # 保存图表
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_network_metrics_comparison(
    networks: Dict[str, Dict[str, Any]],
    methods: List[str],
    years: List[int],
    output_path: str,
    title: str = "网络指标比较",
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 300
) -> str:
    """
    绘制网络指标比较图表
    
    Args:
        networks: 网络数据字典，格式为 {年份: {方法: 网络数据}}
        methods: 方法列表
        years: 年份列表
        output_path: 输出文件路径
        title: 图表标题
        figsize: 图表大小
        dpi: 图表分辨率
        
    Returns:
        输出文件的完整路径
    """
    # 确保years是整数列表
    years = [int(y) if isinstance(y, str) else y for y in years]
    years.sort()  # 确保年份按顺序排列
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 如果output_path没有扩展名，添加.png
    if not output_path.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf', '.svg')):
        output_path = output_path + ".png"
    
    # 提取指标
    metrics = ["density", "avg_clustering", "avg_path_length", "modularity"]
    metric_names = ["网络密度", "平均聚类系数", "平均路径长度", "模块度"]
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # 定义颜色映射 - 不同方法使用不同颜色
    method_colors = {
        "Original": "#000000",  # 黑色
        "Cluster": "#3366cc",   # 蓝色
        "GT": "#ff9900",        # 橙色
        "HSS": "#66cc66",       # 绿色
        "PLA": "#9966cc",       # 紫色
        "Ours": "#cc3333"       # 红色
    }
    
    # 定义标记映射 - 不同年份使用不同标记
    year_markers = {
        2006: 'o',  # 圆形
        2010: 's',  # 方形
        2015: '^',  # 三角形
        2020: 'd'   # 菱形
    }
    
    # 为每个指标绘制折线图
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i]
        
        # 设置背景色
        ax.set_facecolor('#f8f8f8')
        
        # 添加网格线
        ax.grid(True, linestyle='--', alpha=0.7, color='#cccccc')
        
        # 为每种方法绘制一条线
        for method in methods:
            # 收集数据点
            x_values = []
            y_values = []
            
            for year in years:
                year_str = str(year)
                if year_str in networks and method in networks[year_str]:
                    network_metrics = networks[year_str][method].get("metrics", {})
                    if metric in network_metrics:
                        x_values.append(year)
                        y_values.append(network_metrics[metric])
            
            # 绘制折线
            if x_values and y_values:
                ax.plot(x_values, y_values, 
                       marker=year_markers.get(x_values[0], 'o'), 
                       color=method_colors.get(method, "gray"),
                       label=method,
                       linewidth=2 if method == "Ours" else 1.5,
                       alpha=1.0 if method in ["Original", "Ours"] else 0.7)
        
        # 设置标题和标签
        ax.set_title(metric_name, fontsize=14, fontweight='bold')
        ax.set_xlabel('年份', fontsize=12)
        ax.set_ylabel('值', fontsize=12)
        
        # 设置x轴刻度
        ax.set_xticks(years)
        ax.set_xticklabels([str(year) for year in years])
        
        # 美化坐标轴
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # 添加图例
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', 
              ncol=len(methods), bbox_to_anchor=(0.5, 0.02),
              frameon=True, fancybox=True, shadow=True)
    
    # 添加总标题
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    # 保存图表
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    return output_path

def generate_sample_network_data(
    years: List[int],
    methods: List[str],
    node_count: int = 100,
    edge_density: float = 0.05,
    categories: List[str] = None,
    output_file: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    生成示例网络数据的功能已被禁用
    
    Args:
        years: 年份列表
        methods: 方法列表
        node_count: 节点数量
        edge_density: 边密度
        categories: API类别列表
        output_file: 输出文件路径，如果为None则不保存
        
    Returns:
        网络数据，格式为 {年份: {方法: 网络数据}}
    """
    raise ValueError("模拟数据生成功能已被禁用。请使用从programmableweb_2022(1).sql和programmableweb_dataset(1).sql导入的真实数据。")

def plot_experiment_framework(
    output_path: str,
    years: List[int] = None,
    methods: List[str] = None,
    network_data_file: Optional[str] = None,
    network_data: Optional[Dict] = None,
    title: str = "API关系骨架计算实验设计框架",
    figsize: Tuple[int, int] = (18, 14),
    dpi: int = 300
) -> str:
    """
    绘制计算实验设计框架图表
    
    Args:
        output_path: 输出路径
        years: 年份列表，如果为None则使用默认值
        methods: 方法列表，如果为None则使用默认值
        network_data_file: 网络数据文件路径
        network_data: 直接提供的网络数据，优先使用此参数
        title: 图表标题
        figsize: 图表大小
        dpi: 图表分辨率
        
    Returns:
        输出文件路径
    """
    # 设置默认值
    if years is None:
        years = [2006, 2010, 2015, 2020]
    
    if methods is None:
        methods = ["Original", "Cluster", "GT", "HSS", "PLA", "Ours"]
    
    # 获取网络数据
    networks = None
    
    # 优先使用直接提供的网络数据
    if network_data is not None:
        networks = network_data.get('networks', {})
    # 其次从文件加载
    elif network_data_file and os.path.exists(network_data_file):
        try:
            # 尝试多种编码格式
            encodings = ['utf-8', 'utf-16', 'gbk', 'latin1']
            for encoding in encodings:
                try:
                    with open(network_data_file, 'r', encoding=encoding) as f:
                        data = json.load(f)
                        networks = data.get('networks', {})
                    if networks:
                        break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"使用{encoding}编码加载网络数据失败: {str(e)}")
            
            # 如果仍未加载成功，尝试二进制读取
            if not networks:
                with open(network_data_file, 'rb') as f:
                    content = f.read()
                    import chardet
                    detected = chardet.detect(content)
                    encoding = detected.get('encoding')
                    if encoding:
                        data = json.loads(content.decode(encoding))
                        networks = data.get('networks', {})
        except Exception as e:
            raise ValueError(f"加载网络数据文件失败: {str(e)}")
    
    # 如果仍未获取到网络数据，则报错
    if not networks:
        raise ValueError("必须提供有效的网络数据。模拟数据生成功能已被禁用。请使用从programmableweb_2022(1).sql和programmableweb_dataset(1).sql导入的真实数据。")
    
    # 如果传入的是字符串形式的年份和方法，则转换为列表
    if isinstance(years, str):
        years = [int(y.strip()) for y in years.split(',')]
    elif isinstance(years[0], str):
        years = [int(y) for y in years]
    
    if isinstance(methods, str):
        methods = [m.strip() for m in methods.split(',')]
    
    # 绘制网络指标对比图
    return plot_network_metrics_comparison(
        networks=networks,
        methods=methods,
        years=years,
        output_path=output_path,
        title=title,
        figsize=figsize,
        dpi=dpi
    ) 