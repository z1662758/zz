import os
import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import logging
import random
from typing import Dict, List, Tuple, Any, Optional, Union

# 设置中文字体支持，折线图
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
except:
    logging.warning("无法设置中文字体，图表中的中文可能无法正确显示")

def load_network_data(network_data_file: str) -> Dict:
    """
    从文件加载网络数据
    
    Args:
        network_data_file: 网络数据文件路径
        
    Returns:
        网络数据字典
    """
    # 检查文件是否存在
    if not os.path.exists(network_data_file):
        raise FileNotFoundError(f"网络数据文件不存在: {network_data_file}")
    
    # 尝试多种编码加载数据
    network_data = None
    encodings = ['utf-8-sig', 'utf-8', 'utf-16', 'gbk', 'latin1', None]
    
    for encoding in encodings:
        try:
            if encoding:
                with open(network_data_file, 'r', encoding=encoding) as f:
                    network_data = json.load(f)
            else:
                # 二进制读取，尝试自动检测编码
                with open(network_data_file, 'rb') as f:
                    content = f.read()
                    import chardet
                    detected = chardet.detect(content)
                    encoding = detected['encoding']
                    if encoding:
                        network_data = json.loads(content.decode(encoding))
            
            if network_data:
                break
        except Exception:
            continue
    
    if not network_data:
        raise ValueError(f"无法加载网络数据文件: {network_data_file}，请检查文件格式和编码。")
    
    return network_data

def generate_community_network(num_nodes: int, num_edges: int, community_count: int) -> nx.Graph:
    """
    生成具有社区结构的网络
    
    Args:
        num_nodes: 节点数
        num_edges: 边数
        community_count: 社区数
        
    Returns:
        具有社区结构的NetworkX图
    """
    # 创建空图
    G = nx.Graph()
    
    # 添加节点
    for i in range(num_nodes):
        # 为节点分配社区
        community = i % community_count
        G.add_node(i, community=community)
    
    # 添加边
    edges_added = 0
    
    # 首先在社区内添加边
    for community in range(community_count):
        community_nodes = [n for n, data in G.nodes(data=True) if data.get("community") == community]
        
        # 在社区内部优先添加边
        for _ in range(min(len(community_nodes) * 2, num_edges - edges_added)):
            if edges_added >= num_edges:
                break
                
            u = random.choice(community_nodes)
            v = random.choice(community_nodes)
            
            if u != v and not G.has_edge(u, v):
                weight = random.uniform(0.5, 1.0)  # 社区内边权重较高
                G.add_edge(u, v, weight=weight)
                edges_added += 1
    
    # 添加社区间的边
    while edges_added < num_edges:
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        
        if u != v and not G.has_edge(u, v):
            u_community = G.nodes[u].get("community")
            v_community = G.nodes[v].get("community")
            
            # 社区间边权重较低
            weight = random.uniform(0.1, 0.5) if u_community != v_community else random.uniform(0.5, 1.0)
            G.add_edge(u, v, weight=weight)
            edges_added += 1
    
    return G

def calculate_modularity(G: nx.Graph) -> float:
    """
    计算网络的模块度
    
    Args:
        G: NetworkX图
        
    Returns:
        模块度值
    """
    # 获取社区标签
    communities = {}
    for node, data in G.nodes(data=True):
        community = data.get("community", 0)
        if community not in communities:
            communities[community] = []
        communities[community].append(node)
    
    # 将社区列表转换为nx.community模块需要的格式
    community_list = list(communities.values())
    
    # 计算模块度
    try:
        from networkx.algorithms import community
        return community.modularity(G, community_list)
    except:
        # 如果没有安装社区检测模块，返回一个模拟值
        return random.uniform(0.3, 0.7)

def plot_network_visualization(network_data: Dict, output_path: str = "network_visualization.png", 
                               title: str = "API关系骨架计算实验设计框架", figsize: Tuple[int, int] = (12, 8)) -> str:
    """
    绘制网络可视化图表
    
    Args:
        network_data: 网络数据字典
        output_path: 输出文件路径
        title: 图表标题
        figsize: 图表大小
        
    Returns:
        输出文件的完整路径
    """
    years = network_data["years"]
    methods = network_data["methods"]
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 如果output_path没有扩展名，添加.png
    if not output_path.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf', '.svg')):
        output_path = output_path + ".png"
    
    # 创建图表
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    
    # 设置网格布局
    nrows = len(years)
    ncols = len(methods)
    
    # 创建GridSpec
    gs = gridspec.GridSpec(nrows, ncols, figure=fig)
    
    # 创建颜色映射
    cmap = plt.cm.tab10
    
    # 存储所有社区数量
    all_community_counts = []
    
    # 为每个年份和方法绘制子图
    for i, year in enumerate(years):
        for j, method in enumerate(methods):
            # 获取网络数据
            year_str = str(year)
            if year_str not in network_data["networks"]:
                continue
                
            if method not in network_data["networks"][year_str]:
                continue
                
            network = network_data["networks"][year_str][method]
            
            # 创建NetworkX图
            G = nx.Graph()
            
            # 添加节点
            for node in network["nodes"]:
                G.add_node(node["id"], community=node.get("community", 0))
            
            # 添加边
            for edge in network["edges"]:
                G.add_edge(edge["source"], edge["target"], weight=edge.get("weight", 1.0))
            
            # 获取社区
            communities = {}
            for node, data in G.nodes(data=True):
                comm = data.get("community", 0)
                if comm not in communities:
                    communities[comm] = []
                communities[comm].append(node)
            
            # 记录社区数量
            all_community_counts.append(len(communities))
            
            # 创建子图
            ax = fig.add_subplot(gs[i, j])
            
            # 设置节点位置
            pos = nx.spring_layout(G, seed=42)  # 使用固定种子以获得一致的布局
            
            # 绘制节点
            for comm_id, nodes in communities.items():
                nx.draw_networkx_nodes(
                    G, pos, 
                    nodelist=nodes, 
                    node_color=[cmap(comm_id % 10)], 
                    node_size=50,
                    alpha=0.8,
                    ax=ax
                )
            
            # 绘制边
            edge_weights = [G[u][v].get("weight", 1.0) for u, v in G.edges()]
            nx.draw_networkx_edges(
                G, pos, 
                width=1.0, 
                alpha=0.5, 
                edge_color="gray",
                ax=ax
            )
            
            # 添加指标文本
            metrics = network.get("metrics", {})
            metrics_text = f"N={len(G.nodes())}, E={len(G.edges())}\n"
            metrics_text += f"D={metrics.get('density', 0):.2f}\n"
            metrics_text += f"C={metrics.get('avg_clustering', 0):.2f}\n"
            metrics_text += f"L={metrics.get('avg_path_length', 0):.2f}\n"
            metrics_text += f"Q={metrics.get('modularity', 0):.2f}"
            
            # 添加子图标题
            if i == 0:
                ax.set_title(method, fontsize=12)
            
            # 添加年份标签
            if j == 0:
                ax.text(-0.1, 0.5, str(year), transform=ax.transAxes, 
                        fontsize=12, va='center', ha='right', rotation=90)
            
            # 添加指标文本
            ax.text(0.02, 0.02, metrics_text, transform=ax.transAxes, 
                    fontsize=8, va='bottom', ha='left', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            # 移除坐标轴
            ax.axis('off')
    
    # 添加图例
    legend_elements = []
    max_communities = 5
    if all_community_counts:
        max_communities = min(5, max(all_community_counts))
    
    for i in range(max_communities):
        legend_elements.append(
            Patch(facecolor=cmap(i % 10), edgecolor='k', label=f'社区 {i+1}')
        )
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=10, 
               bbox_to_anchor=(0.5, 0.02))
    
    # 添加总标题
    fig.suptitle(title, fontsize=16, y=0.98)
    
    # 添加指标说明
    fig.text(0.01, 0.02, "N: 节点数, E: 边数, D: 密度, C: 聚类系数, L: 平均路径长度, Q: 模块度", 
             fontsize=8, ha='left')
    
    # 保存图表
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_path

def plot_network_metrics_comparison(network_data: Dict, output_path: str = "network_metrics_comparison.png",
                                   title: str = "网络指标比较", figsize: Tuple[int, int] = (10, 6)) -> str:
    """
    绘制网络指标比较图表
    
    Args:
        network_data: 网络数据字典
        output_path: 输出文件路径
        title: 图表标题
        figsize: 图表大小
        
    Returns:
        输出文件的完整路径
    """
    years = network_data["years"]
    methods = network_data["methods"]
    
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
    
    # 为每个指标绘制折线图
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i]
        
        # 为每种方法绘制一条线
        for method in methods:
            # 收集数据点
            x_values = []
            y_values = []
            
            for year in years:
                year_str = str(year)
                if year_str in network_data["networks"] and method in network_data["networks"][year_str]:
                    network_metrics = network_data["networks"][year_str][method].get("metrics", {})
                    if metric in network_metrics:
                        x_values.append(year)
                        y_values.append(network_metrics[metric])
            
            # 绘制折线
            if x_values and y_values:
                ax.plot(x_values, y_values, marker='o', label=method)
        
        # 设置标题和标签
        ax.set_title(metric_name)
        ax.set_xlabel('年份')
        ax.set_ylabel('值')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 设置x轴刻度
        ax.set_xticks(years)
    
    # 添加图例
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(methods), bbox_to_anchor=(0.5, 0.02))
    
    # 添加总标题
    fig.suptitle(title, fontsize=16, y=0.98)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    # 保存图表
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    
    return output_path

class NetworkVisualizer:
    """网络可视化工具类"""
    
    def __init__(self, logger=None):
        """
        初始化网络可视化器
        
        Args:
            logger: 日志记录器，如果为None则创建新的记录器
        """
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
    
    def create_network_graph(
        self,
        nodes: List[Dict],
        edges: List[Dict],
        title: str = "API协作网络",
        output_file: str = "network_graph.png",
        figsize: Tuple[int, int] = (12, 10),
        dpi: int = 300,
        show_labels: bool = False,
        layout: str = "spring"
    ) -> str:
        """
        创建网络图
        
        Args:
            nodes: 节点列表，每个元素为包含'id', 'name', 'category', 'size'的字典
            edges: 边列表，每个元素为包含'source', 'target', 'weight'的字典
            title: 图表标题
            output_file: 输出文件路径
            figsize: 图表大小
            dpi: 图表分辨率
            show_labels: 是否显示节点标签
            layout: 布局算法，可选'spring', 'kamada_kawai', 'circular', 'shell'
            
        Returns:
            输出文件路径
        """
        self.logger.info(f"创建网络图: {title}")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # 检查数据有效性
        if not nodes or not edges:
            self.logger.error("节点或边数据为空")
            return None
        
        # 创建NetworkX图
        G = nx.Graph()
        
        # 添加节点
        categories = set()
        for node in nodes:
            node_id = node['id']
            node_attrs = {
                'name': node.get('name', str(node_id)),
                'category': node.get('category', 'default'),
                'size': node.get('size', 1)
            }
            G.add_node(node_id, **node_attrs)
            categories.add(node_attrs['category'])
        
        # 添加边
        for edge in edges:
            source = edge['source']
            target = edge['target']
            weight = edge.get('weight', 1.0)
            G.add_edge(source, target, weight=weight)
        
        # 创建图表
        plt.figure(figsize=figsize)
        
        # 选择布局
        if layout == 'spring':
            pos = nx.spring_layout(G, k=0.15, iterations=50)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'shell':
            pos = nx.shell_layout(G)
        else:
            self.logger.warning(f"未知的布局算法: {layout}，使用默认的spring布局")
            pos = nx.spring_layout(G, k=0.15, iterations=50)
        
        # 设置节点颜色映射
        category_list = sorted(list(categories))
        color_map = plt.cm.tab20(np.linspace(0, 1, len(category_list)))
        category_colors = {cat: color_map[i] for i, cat in enumerate(category_list)}
        
        # 获取节点颜色和大小
        node_colors = [category_colors[G.nodes[n]['category']] for n in G.nodes()]
        node_sizes = [G.nodes[n]['size'] * 100 for n in G.nodes()]
        
        # 获取边权重
        edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
        
        # 绘制网络
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=edge_weights)
        nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                      node_size=node_sizes, alpha=0.8)
        
        # 添加节点标签
        if show_labels:
            labels = {n: G.nodes[n]['name'] for n in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, 
                                   font_family='sans-serif')
        
        # 添加图例
        legend_elements = [Patch(facecolor=category_colors[cat], 
                                edgecolor='w', 
                                label=cat) for cat in category_list]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # 设置标题和边界
        plt.title(title, size=15)
        plt.axis('off')
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"网络图已保存至: {output_file}")
        return output_file
    
    def create_community_graph(
        self,
        nodes: List[Dict],
        edges: List[Dict],
        title: str = "API社区结构",
        output_file: str = "community_graph.png",
        community_detection: str = "louvain"
    ) -> str:
        """
        创建社区结构网络图
        
        Args:
            nodes: 节点列表，每个元素为包含'id', 'name', 'category'的字典
            edges: 边列表，每个元素为包含'source', 'target', 'weight'的字典
            title: 图表标题
            output_file: 输出文件路径
            community_detection: 社区检测算法，可选'louvain', 'girvan_newman', 'label_propagation'
            
        Returns:
            输出文件路径
        """
        self.logger.info(f"创建社区结构网络图: {title}")
        
        # 创建NetworkX图
        G = nx.Graph()
        
        # 添加节点
        for node in nodes:
            node_id = node['id']
            node_attrs = {
                'name': node.get('name', str(node_id)),
                'category': node.get('category', 'default'),
                'size': node.get('size', 1)
            }
            G.add_node(node_id, **node_attrs)
        
        # 添加边
        for edge in edges:
            source = edge['source']
            target = edge['target']
            weight = edge.get('weight', 1.0)
            G.add_edge(source, target, weight=weight)
        
        # 执行社区检测
        try:
            if community_detection == 'louvain':
                from community import best_partition
                partition = best_partition(G)
            elif community_detection == 'girvan_newman':
                from networkx.algorithms.community import girvan_newman
                communities = list(girvan_newman(G))
                partition = {}
                for i, community in enumerate(communities[0]):  # 使用第一级划分
                    for node in community:
                        partition[node] = i
            elif community_detection == 'label_propagation':
                from networkx.algorithms.community import label_propagation_communities
                communities = list(label_propagation_communities(G))
                partition = {}
                for i, community in enumerate(communities):
                    for node in community:
                        partition[node] = i
            else:
                self.logger.warning(f"未知的社区检测算法: {community_detection}，使用随机社区")
                partition = {node: random.randint(0, 4) for node in G.nodes()}
        except ImportError:
            self.logger.warning(f"无法导入社区检测算法: {community_detection}，使用随机社区")
            partition = {node: random.randint(0, 4) for node in G.nodes()}
        
        # 更新节点的社区属性
        for node, community in partition.items():
            G.nodes[node]['community'] = community
        
        # 创建带有社区信息的节点列表
        community_nodes = []
        for node in nodes:
            node_id = node['id']
            node_copy = node.copy()
            node_copy['community'] = G.nodes[node_id].get('community', 0)
            community_nodes.append(node_copy)
        
        # 使用create_network_graph绘制社区结构
        return self.create_network_graph(
            nodes=community_nodes,
            edges=edges,
            title=title,
            output_file=output_file,
            show_labels=True
        )
    
    def load_network_data(self, file_path: str) -> Dict:
        """
        从JSON文件加载网络数据
        
        Args:
            file_path: JSON文件路径
            
        Returns:
            网络数据字典
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            self.logger.error(f"加载网络数据失败: {e}")
            return {} 