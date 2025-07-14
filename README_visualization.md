# API服务生态系统可视化工具

本文档介绍了API服务生态系统情景生成项目中的可视化功能，包括雷达图可视化、网络可视化和HTML报告生成。

## 功能概述

1. **雷达图可视化**：展示API需求变化情况，支持多情景对比和单情景详情展示
2. **网络可视化**：展示API协作网络结构，支持社区划分和关键节点识别
3. **HTML报告生成**：整合可视化结果，生成美观的HTML报告

## 使用方法

### 生成示例数据

首先，可以使用示例数据生成工具来创建测试数据：

```bash
python utils/generate_sample_data.py
```

这将在`../output/sample_data/`目录下生成以下文件：
- `sample_scenarios.json`：示例情景数据
- `sample_network.json`：示例网络数据

### 运行可视化功能

使用以下命令运行可视化功能：

```bash
python main.py --mode visualization --scenarios_file <情景数据文件> --network_file <网络数据文件> --output_dir <输出目录> [--generate_report]
```

参数说明：
- `--mode visualization`：指定运行模式为可视化模式
- `--scenarios_file`：指定情景数据文件路径
- `--network_file`：指定网络数据文件路径
- `--output_dir`：指定输出目录
- `--generate_report`：是否生成HTML报告（可选）
- `--report_title`：指定HTML报告标题（可选）

示例：

```bash
python main.py --mode visualization --scenarios_file ../output/sample_data/sample_scenarios.json --network_file ../output/sample_data/sample_network.json --output_dir ../output/visualization_test --generate_report
```

### 输出文件

运行可视化功能后，将在指定的输出目录中生成以下文件：

1. **雷达图可视化**：
   - `scenarios_comparison.png`：所有情景的对比雷达图
   - `scenario_<情景名称>.png`：每个情景的单独雷达图

2. **网络可视化**：
   - `api_network.png`：API协作网络图
   - `api_communities.png`：API社区结构图

3. **HTML报告**：
   - `report.html`：整合了所有可视化结果的HTML报告

## 自定义可视化

### 自定义雷达图

可以使用`RadarVisualizer`类来自定义雷达图：

```python
from utils.radar_visualizer import RadarVisualizer

# 初始化雷达图可视化器
radar_viz = RadarVisualizer()

# 创建雷达图
radar_viz.create_radar_chart(
    data=[('情景1', [30, 25, 15, 20, 10])],
    categories=['基础设施', '生活服务', '商业管理', '社交娱乐', '开发工具'],
    title='API需求变化雷达图',
    output_file='radar_chart.png'
)
```

### 自定义网络图

可以使用`NetworkVisualizer`类来自定义网络图：

```python
from utils.network_visualizer import NetworkVisualizer

# 初始化网络可视化器
network_viz = NetworkVisualizer()

# 创建网络图
network_viz.create_network_graph(
    nodes=[
        {"id": "API1", "name": "支付API", "category": "基础设施", "size": 10},
        {"id": "API2", "name": "地图API", "category": "生活服务", "size": 8}
    ],
    edges=[
        {"source": "API1", "target": "API2", "weight": 0.8}
    ],
    title='API协作网络',
    output_file='network_graph.png',
    show_labels=True
)
```

### 自定义HTML报告

可以使用`generate_html_report`函数来自定义HTML报告：

```python
from utils.report_generator import generate_html_report

# 生成HTML报告
generate_html_report(
    scenarios_file='scenarios.json',
    network_file='network.json',
    visualization_dir='visualization',
    output_file='report.html',
    title='API服务生态系统情景分析报告'
)
```

## 数据格式

### 情景数据格式

情景数据应为JSON格式，包含以下字段：

```json
{
    "categories": ["基础设施", "生活服务", "商业管理", "社交娱乐", "开发工具"],
    "scenarios": [
        {
            "name": "情景1",
            "description": "这是情景1的描述",
            "demand_changes": {
                "基础设施": 30,
                "生活服务": 25,
                "商业管理": 15,
                "社交娱乐": 20,
                "开发工具": 10
            },
            "probability": 0.8,
            "duration": 12,
            "long_term_impacts": "情景1的长期影响描述"
        }
    ]
}
```

### 网络数据格式

网络数据应为JSON格式，包含以下字段：

```json
{
    "nodes": [
        {
            "id": "API1",
            "name": "支付API",
            "category": "基础设施",
            "size": 10
        }
    ],
    "edges": [
        {
            "source": "API1",
            "target": "API2",
            "weight": 0.8
        }
    ]
}
```

## 依赖库

- numpy
- matplotlib
- networkx
- python-louvain (用于社区检测)

