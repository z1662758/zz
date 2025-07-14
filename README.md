# API服务生态系统情景生成

本项目旨在构建一个多智能体系统，用于模拟API服务生态系统中的各种情景，并分析其对生态系统的影响。

## 项目结构

```
api_ecosystem_simulation/
├── agents/                 # 智能体实现
│   ├── base_agent.py       # 基础智能体
│   ├── environment_agent.py # 环境智能体
│   ├── social_agent.py     # 社会智能体
│   └── ...
├── data/                   # 数据目录
│   ├── raw/                # 原始数据
│   └── processed/          # 处理后的数据
├── experiments/            # 实验配置
│   └── config.py           # 实验配置
├── models/                 # 模型实现
│   └── ...
├── utils/                  # 工具函数
│   ├── logger.py           # 日志工具
│   ├── llm_interface.py    # LLM接口
│   ├── prompt_templates.py # 提示词模板
│   ├── visualization.py    # 可视化工具
│   ├── network_visualizer.py # 网络可视化工具
│   └── report_generator.py # 报告生成工具
└── main.py                 # 主入口
```

## 功能模块

### 1. 环境智能体 (Environment Agent)

环境智能体负责生成极端情景，模拟API服务生态系统中可能出现的各种极端情况。

### 2. 社会智能体 (Social Agent)

社会智能体负责构建协同网络，反映API服务之间的交互关系。

### 3. 规划智能体 (Planner Agent)

规划智能体负责生成实验规则，指导API服务生态系统的健康发展。

### 4. 验证智能体 (Validation Agent)

验证智能体负责评估情景的合理性，确保生成的情景符合实际情况。

## 运行方式

### 测试模式

```bash
python main.py --mode test [--llm_config <配置文件路径>] [--mock]
```

### 运行模式

```bash
python main.py --mode run [--llm_config <配置文件路径>] [--output <输出目录>] [--mock]
```

### 测试环境智能体

```bash
python main.py --mode test_env_agent [--llm_config <配置文件路径>] [--mock]
```

### 测试社会智能体

```bash
python main.py --mode test_social_agent [--llm_config <配置文件路径>] [--mock]
```

### 测试规划智能体

```bash
python main.py --mode test_planner_agent [--llm_config <配置文件路径>] [--mock]
```

### 可视化模式

```bash
python main.py --mode visualize [--scenario_file <情景文件路径>] [--historical_data_file <历史数据文件路径>] [--vis_title <图表标题>] [--output <输出目录>] [--report]
```

### 网络可视化模式

```bash
python main.py --mode network_vis [--network_data_file <网络数据文件路径>] [--network_years <年份列表>] [--network_methods <方法列表>] [--network_title <图表标题>] [--output <输出目录>]
```

### 雷达图可视化模式

```bash
python main.py --mode radar_vis [--radar_file <雷达图数据文件路径>] [--radar_title <图表标题>] [--highlight_methods <高亮方法列表>] [--output <输出目录>]
```

### 一次性生成所有可视化图表模式

```bash
python main.py --mode all_vis [--network_file <网络数据文件路径>] [--radar_file <雷达图数据文件路径>] [--scenario_file <情景文件路径>] [--historical_data_file <历史数据文件路径>] [--output_dir <输出目录>]
```

这个模式会依次执行网络可视化、雷达图可视化和情景可视化，并将所有结果输出到指定目录。

## 网络可视化功能

网络可视化功能用于生成API关系骨架计算实验设计框架图表，可以帮助用户理解不同方法对API关系网络的影响，以及随着时间的推移，API网络的演化过程。

### 参数说明

- `--network_data_file`：网络数据文件路径，如果不指定，将自动生成模拟数据
- `--network_years`：网络可视化年份列表，用逗号分隔，默认为"2006,2010,2015,2020"
- `--network_methods`：网络可视化方法列表，用逗号分隔，默认为"Original,Cluster,GT,HSS,PLA,Ours"
- `--network_title`：网络可视化图表标题，默认为"API关系骨架计算实验设计框架"
- `--output`：输出目录，默认为"output/network_vis"
- `--report`：是否生成HTML报告，默认为否

### 示例

```bash
# 使用默认参数生成网络可视化图表
python main.py --mode network_vis

# 指定年份和方法生成网络可视化图表
python main.py --mode network_vis --network_years "2010,2015,2020" --network_methods "Original,HSS,Ours"

# 生成网络可视化图表和HTML报告
python main.py --mode network_vis --report

# 指定输出目录
python main.py --mode network_vis --output "../output/my_network_vis"
```

### 输出文件

- `network_data.json`：网络数据文件
- `network_visualization.png`：网络可视化图表
- `network_metrics_comparison.png`：网络指标比较图表
- `network_analysis_report.html`：HTML报告（如果指定了`--report`参数）


## 依赖

- Python 3.8+
- OpenAI Python SDK
- NetworkX
- Matplotlib
- NumPy
- Pandas 