"""
可视化模块的测试脚本
1）通过RadarVisualizer生成雷达图；
2）通过NetworkVisualizer生成网络拓扑图，呈现API之间的调用关系（如支付API与地图API的连接），
并支持按节点类别着色、按边权重调整线条粗细。测试结果会保存到../output/visualization_test/目录下的PNG文件中，同时输出日志
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.radar_visualizer import RadarVisualizer, radar_factory
from utils.network_visualizer import NetworkVisualizer
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_radar_visualizer():
    """测试雷达图可视化功能"""
    print("开始测试雷达图可视化...")
    
    # 创建测试数据
    categories = ['基础设施', '生活服务', '商业管理', '社交娱乐', '开发工具']
    data = [
        ('情景1', [30, 25, 15, 20, 10]),
        ('情景2', [15, 30, 25, 10, 20]),
        ('情景3', [20, 10, 30, 25, 15])
    ]
    
    # 创建保存目录
    output_dir = '../output/visualization_test'
    os.makedirs(output_dir, exist_ok=True)
    
    # 确保radar投影已注册
    num_vars = len(categories)
    theta = radar_factory(num_vars, frame='polygon')
    
    # 初始化雷达图可视化器
    radar_viz = RadarVisualizer(logger=logger)
    
    # 生成雷达图
    output_file = os.path.join(output_dir, 'test_radar_chart.png')
    try:
        result = radar_viz.create_radar_chart(
            data=data,
            categories=categories,
            title='API需求变化雷达图测试',
            output_file=output_file
        )
        print(f"雷达图已保存至: {result}")
        return result
    except Exception as e:
        print(f"生成雷达图失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_network_visualizer():
    """测试网络可视化功能"""
    print("开始测试网络可视化...")
    
    # 创建测试数据
    nodes = [
        {"id": "API1", "name": "支付API", "category": "基础设施", "size": 10},
        {"id": "API2", "name": "地图API", "category": "生活服务", "size": 8},
        {"id": "API3", "name": "社交API", "category": "社交娱乐", "size": 12},
        {"id": "API4", "name": "云存储API", "category": "基础设施", "size": 9},
        {"id": "API5", "name": "消息推送API", "category": "开发工具", "size": 7}
    ]
    
    edges = [
        {"source": "API1", "target": "API2", "weight": 0.8},
        {"source": "API1", "target": "API4", "weight": 0.6},
        {"source": "API2", "target": "API3", "weight": 0.7},
        {"source": "API3", "target": "API5", "weight": 0.5},
        {"source": "API4", "target": "API5", "weight": 0.4},
        {"source": "API2", "target": "API5", "weight": 0.3}
    ]
    
    # 创建保存目录
    output_dir = '../output/visualization_test'
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化网络可视化器
    network_viz = NetworkVisualizer(logger=logger)
    
    # 生成网络图
    output_file = os.path.join(output_dir, 'test_network_graph.png')
    try:
        result = network_viz.create_network_graph(
            nodes=nodes,
            edges=edges,
            title='API协作网络测试',
            output_file=output_file,
            show_labels=True
        )
        print(f"网络图已保存至: {result}")
        return result
    except Exception as e:
        print(f"生成网络图失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    print("开始测试可视化模块...")
    
    # 测试雷达图可视化
    radar_output = test_radar_visualizer()
    
    # 测试网络可视化
    network_output = test_network_visualizer()
    
    print("可视化测试完成!")
    print(f"雷达图输出: {radar_output}")
    print(f"网络图输出: {network_output}")

if __name__ == "__main__":
    main() 