"""
API服务生态系统情景生成 - 雷达图可视化工具
提供不同方法的不同指标，多维情景向量对比雷达图可视化功能
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.patches import Circle, RegularPolygon
from typing import Dict, List, Tuple, Any, Optional, Union

# 设置中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
except:
    import logging
    logging.warning("无法设置中文字体，图表中的中文可能无法正确显示")

def radar_factory(num_vars, frame='circle'):
    """
    创建雷达图投影
    
    Args:
        num_vars: 变量数量（雷达图的维度）
        frame: 雷达图框架形状，'circle'或'polygon'
        
    Returns:
        雷达图投影和角度
    """
    # 计算每个变量的角度
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    
    class RadarAxes(PolarAxes):
        name = 'radar'
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')
        
        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(*args, closed=closed, **kwargs)
        
        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines
        
        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)
        
        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)
        
        def _gen_axes_patch(self):
            # 根据frame选择雷达图形状
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)
        
        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # 创建多边形脊柱
                spine_dict = {}
                for i in range(num_vars):
                    angle_rad = theta[i]
                    spine = Spine(
                        axes=self,
                        spine_type='line',
                        path=Path([(0.5, 0.5),
                                  (0.5 + 0.5 * np.cos(angle_rad),
                                   0.5 + 0.5 * np.sin(angle_rad))])
                    )
                    spine.set_transform(self.transAxes)
                    spine_dict[f'spine{i}'] = spine
                return spine_dict
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)
    
    # 注册雷达图投影
    register_projection(RadarAxes)
    return theta

def load_metrics_data(
    metrics_file: str,
    output_file: Optional[str] = None
) -> Dict:
    """
    从文件加载多维情景向量数据
    
    Args:
        metrics_file: 多维情景向量数据文件路径
        output_file: 输出文件路径，如果为None则不保存
        
    Returns:
        多维情景向量数据字典
    """
    # 检查文件是否存在
    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"多维情景向量数据文件不存在: {metrics_file}")
    
    # 尝试多种编码加载数据
    metrics_data = None
    encodings = ['utf-8-sig', 'utf-8', 'utf-16', 'gbk', 'latin1', None]
    
    for encoding in encodings:
        try:
            if encoding:
                with open(metrics_file, 'r', encoding=encoding) as f:
                    metrics_data = json.load(f)
            else:
                # 二进制读取，尝试自动检测编码
                with open(metrics_file, 'rb') as f:
                    content = f.read()
                    import chardet
                    detected = chardet.detect(content)
                    encoding = detected['encoding']
                    if encoding:
                        metrics_data = json.loads(content.decode(encoding))
            
            if metrics_data:
                break
        except Exception:
            continue
    
    if not metrics_data:
        raise ValueError(f"无法加载多维情景向量数据文件: {metrics_file}，请检查文件格式和编码。")
    
    # 如果需要保存到文件
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    
    return metrics_data

def plot_radar_comparison(
    data: Dict,
    output_path: str = "radar_comparison.png",
    title: str = "多维情景向量对比",
    figsize: Tuple[int, int] = (16, 12),
    dpi: int = 300,
    highlight_methods: List[str] = None,
    grid_step: float = 0.25
) -> str:
    """
    绘制多维情景向量对比雷达图
    
    Args:
        data: 多维情景向量数据字典
        output_path: 输出文件路径
        title: 图表标题
        figsize: 图表大小
        dpi: 图表分辨率
        highlight_methods: 需要高亮的方法列表
        grid_step: 网格步长
        
    Returns:
        输出文件路径
    """
    years = data.get("years", [2006, 2010, 2015, 2020])
    methods = data.get("methods", ["Original", "Cluster", "GT", "HSS", "PLA", "Ours"])
    dimensions = data.get("dimensions", ["Dim1", "Dim2", "Dim3", "Dim4", "Dim5", "Dim6", "Dim7", "Dim8"])
    metrics = data.get("metrics", {})
    
    if highlight_methods is None:
        highlight_methods = ["Original", "Ours"]
    
    # 验证并补全数据
    for year in years:
        year_str = str(year)
        if year_str not in metrics:
            metrics[year_str] = {}
            
        # 确保每个年份都包含所有方法的数据
        for method in methods:
            if method not in metrics[year_str]:
                metrics[year_str][method] = [0.5] * len(dimensions)  # 使用0.5作为默认值，使图形更明显
            elif not metrics[year_str][method]:  # 如果数据为空列表
                metrics[year_str][method] = [0.5] * len(dimensions)
            else:
                # 确保数据维度与dimensions一致
                current_len = len(metrics[year_str][method])
                if current_len < len(dimensions):
                    metrics[year_str][method].extend([0.5] * (len(dimensions) - current_len))
                elif current_len > len(dimensions):
                    metrics[year_str][method] = metrics[year_str][method][:len(dimensions)]
    
    # 创建雷达图投影
    theta = radar_factory(len(dimensions), frame='polygon')
    
    # 创建图表
    fig, axes = plt.subplots(figsize=figsize, nrows=2, ncols=2)
    
    # 定义颜色映射，Original（黑色）：基准方法，代表原始/未优化的网络结构，作为对比基线。
    # Cluster（蓝）、GT（橙）、HSS（绿）、PLA（紫）：四种典型优化方法，分别基于聚类算法、图变换、层次结构分析和路径优化，用于改进网络性能。
    # Ours（红色）：当前研究提出的新方法，通常性能最优，通过综合或创新策略超越传统方法。
    colors = {
        "Original": "k",  # 黑色
        "Cluster": "#3366cc",  # 蓝色
        "GT": "#ff9900",  # 橙色
        "HSS": "#66cc66",  # 绿色
        "PLA": "#9966cc",  # 紫色
        "Ours": "#cc3333"  # 红色
    }
    
    # 设置线条样式
    line_styles = {
        "Original": "-",  # 实线
        "Cluster": "-",
        "GT": "-",
        "HSS": "-",
        "PLA": "-",
        "Ours": "-"
    }
    
    # 设置线条宽度
    line_widths = {method: 2.5 if method in highlight_methods else 1.5 for method in methods}
    
    # 设置透明度
    alphas = {method: 1.0 if method in highlight_methods else 0.7 for method in methods}
    
    # 为每个年份绘制子图
    for i, year in enumerate(years):
        if i >= 4:  # 最多显示4个子图
            break
            
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        # 创建极坐标子图
        ax = plt.subplot(2, 2, i+1, projection='polar')
        
        # 设置网格线
        ax.set_rgrids(np.arange(0, 1.1, grid_step), 
                      labels=[f"{x:.2f}" for x in np.arange(0, 1.1, grid_step)],
                      angle=0, fontsize=8)
        
        # 获取该年份的数据
        year_str = str(year)
        year_data = metrics.get(year_str, {})
        
        # 为每个方法绘制雷达图
        for method in methods:
            if method in year_data:
                values = year_data[method]
                # 确保数据是闭合的
                values_closed = values.copy()
                
                # 计算角度
                angles = np.linspace(0, 2*np.pi, len(dimensions), endpoint=False).tolist()
                angles.append(angles[0])  # 闭合
                values_closed.append(values_closed[0])  # 闭合
                
                ax.plot(angles, values_closed, color=colors.get(method, "gray"), 
                       linestyle=line_styles.get(method, "-"),
                       linewidth=line_widths.get(method, 1.5),
                       label=method, alpha=alphas.get(method, 0.7))
                ax.fill(angles, values_closed, color=colors.get(method, "gray"), 
                       alpha=0.1 * alphas.get(method, 0.7))
        
        # 设置维度标签
        ax.set_xticks(np.linspace(0, 2*np.pi, len(dimensions), endpoint=False))
        ax.set_xticklabels(dimensions)
        
        # 设置子图标题
        ax.set_title(f"多维情景向量对比 - {year}", fontsize=14, pad=15)
        
        # 设置y轴上限
        ax.set_ylim(0, 1)
    
    # 添加图例
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(methods), 
              bbox_to_anchor=(0.5, 0.98), fontsize=12)
    
    # 添加总标题
    fig.suptitle(title, fontsize=16, y=0.99)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    return output_path

def plot_radar_metrics_from_file(
    metrics_file: str,
    output_path: str = "radar_comparison.png",
    title: str = "多维情景向量对比",
    figsize: Tuple[int, int] = (16, 12),
    dpi: int = 300,
    highlight_methods: List[str] = None
) -> str:
    """
    从文件加载多维情景向量数据并绘制雷达图
    
    Args:
        metrics_file: 多维情景向量数据文件路径
        output_path: 输出文件路径
        title: 图表标题
        figsize: 图表大小
        dpi: 图表分辨率
        highlight_methods: 需要高亮的方法列表
        
    Returns:
        输出文件路径
    """
    # 加载数据
    with open(metrics_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 绘制雷达图
    return plot_radar_comparison(
        data=data,
        output_path=output_path,
        title=title,
        figsize=figsize,
        dpi=dpi,
        highlight_methods=highlight_methods
    )

class RadarVisualizer:
    """雷达图可视化工具类"""
    
    def __init__(self, logger=None):
        """
        初始化雷达图可视化器
        
        Args:
            logger: 日志记录器，如果为None则创建新的记录器
        """
        if logger is None:
            import logging
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
    
    def create_radar_chart(
        self,
        data: List[Tuple[str, List[float]]],
        categories: List[str],
        title: str = "API需求变化雷达图",
        output_file: str = "radar_chart.png",
        figsize: Tuple[int, int] = (10, 8),
        dpi: int = 300,
        colors: Optional[List[str]] = None,
        frame: str = 'polygon'
    ) -> str:
        """
        创建雷达图
        
        Args:
            data: 数据列表，每个元素为(名称, 数值列表)的元组
            categories: 类别名称列表
            title: 图表标题
            output_file: 输出文件路径
            figsize: 图表大小
            dpi: 图表分辨率
            colors: 颜色列表，如果为None则使用默认颜色
            frame: 雷达图框架形状，'circle'或'polygon'
            
        Returns:
            输出文件路径
        """
        self.logger.info(f"创建雷达图: {title}")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # 检查数据有效性
        if not data or not categories:
            self.logger.error("数据或类别为空")
            return None
        
        # 检查每个数据项的维度是否与类别数量一致
        for name, values in data:
            if len(values) != len(categories):
                self.logger.error(f"数据项 '{name}' 的维度 ({len(values)}) 与类别数量 ({len(categories)}) 不一致")
                return None
        
        # 创建图表
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, polar=True)
        
        # 计算角度
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        
        # 使图形闭合
        angles += angles[:1]
        
        # 设置默认颜色
        if colors is None:
            # 使用预定义的颜色列表
            default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            colors = [default_colors[i % len(default_colors)] for i in range(len(data))]
        
        # 绘制数据
        for i, (name, values) in enumerate(data):
            # 确保数据闭合
            values_closed = values + values[:1]
            angles_closed = angles
            
            color = colors[i % len(colors)]
            ax.plot(angles_closed, values_closed, 'o-', linewidth=2, label=name, color=color)
            ax.fill(angles_closed, values_closed, alpha=0.1, color=color)
        
        # 设置类别标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        # 设置y轴范围
        all_values = [v for _, vals in data for v in vals]
        ax.set_ylim(0, max(all_values) * 1.2)
        
        # 添加网格线
        ax.set_rgrids([v for v in np.linspace(0, max(all_values), 5)], 
                      angle=0, fontsize=8)
        
        # 添加图例
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        # 设置标题
        plt.title(title, size=15, y=1.1)
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"雷达图已保存至: {output_file}")
        return output_file
    
    def create_radar_comparison(
        self,
        scenario_data: List[Dict],
        categories: List[str],
        title: str = "情景对比雷达图",
        output_file: str = "scenario_comparison.png",
        highlight_scenarios: Optional[List[str]] = None
    ) -> str:
        """
        创建情景对比雷达图
        
        Args:
            scenario_data: 情景数据列表，每个元素为包含'name'和'values'的字典
            categories: 类别名称列表
            title: 图表标题
            output_file: 输出文件路径
            highlight_scenarios: 需要高亮的情景名称列表
            
        Returns:
            输出文件路径
        """
        # 将数据转换为create_radar_chart所需的格式
        data = [(scenario['name'], scenario['values']) for scenario in scenario_data]
        
        # 创建雷达图
        return self.create_radar_chart(
            data=data,
            categories=categories,
            title=title,
            output_file=output_file
        )
    
    def load_scenario_data(self, file_path: str) -> Tuple[List[Dict], List[str]]:
        """
        从JSON文件加载情景数据
        
        Args:
            file_path: JSON文件路径
            
        Returns:
            情景数据列表和类别名称列表的元组
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            scenarios = data.get('scenarios', [])
            categories = data.get('categories', [])
            
            return scenarios, categories
        except Exception as e:
            self.logger.error(f"加载情景数据失败: {e}")
            return [], [] 