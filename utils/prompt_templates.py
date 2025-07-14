"""
API服务生态系统情景生成 - 提示词模板
提供各种Agent使用的提示词模板
"""

from string import Template


# 环境Agent提示词模板
ENVIRONMENT_AGENT_SYSTEM_PROMPT = """
你是一个专门负责生成API服务市场极端环境情景的智能体。你需要基于历史数据和领域知识，生成具有长尾分布的极端情景。
你的目标是生成在现实中可能出现但在历史数据中未曾出现的情景，这些情景应该符合业务逻辑和领域知识。

你需要考虑以下因素：
1. 经济环境波动（如经济衰退、技术革新等）
2. 政策监管变化（如数据隐私政策、反垄断政策等）
3. 市场竞争态势（如新兴平台崛起、巨头垄断等）
4. 技术发展趋势（如区块链、人工智能等新技术）

请基于提供的历史数据和领域知识，生成符合要求的极端情景。
"""

ENVIRONMENT_AGENT_PROMPT_TEMPLATE = Template("""
# 任务背景
你需要为API服务市场生成极端环境情景，这些情景应该是在现实中可能出现但在历史数据中未曾出现的。

# 历史数据概要
时间范围：$time_range
API类别：$api_categories
历史需求波动趋势：
$historical_trends

# 领域知识
$domain_knowledge

# 对抗性提示
$adversarial_prompt

# 请生成以下内容
1. 极端情景描述：详细描述这个极端情景的背景、触发因素和影响
2. 各类API在这个情景下的需求变化（用百分比表示增减）
3. 这个情景的发生概率（用百分比表示）
4. 这个情景的持续时间（用月份表示）
5. 这个情景对API服务市场的长期影响

请确保你生成的情景符合业务逻辑和领域知识，并具有合理的极端性。
""")


# 社会Agent提示词模板
SOCIAL_AGENT_SYSTEM_PROMPT = """
你是一个专门负责构建API服务市场协同网络的智能体。你需要基于API之间的调用关系和功能相似性，构建反映主体间交互关系的协同网络。
你的目标是提取社交网络的骨干结构，保留关键连接，同时反映跨社区协作。

你需要考虑以下因素：
1. API之间的调用频率和依赖关系
2. API的功能相似性和互补性
3. API所属类别和生态位
4. 历史协作模式和演化趋势

请基于提供的数据，构建符合要求的协同网络。
"""

SOCIAL_AGENT_PROMPT_TEMPLATE = Template("""
# 任务背景
你需要为API服务市场构建协同网络，这个网络应该反映API之间的交互关系和协作模式。

# API数据概要
API总数：$api_count
API类别：$api_categories
关系数据：
$relationship_data

# 请生成以下内容
1. 社区划分：将API划分为不同的社区，每个社区内的API具有更紧密的联系
2. 关键节点：识别网络中的关键节点（中心性高的API）
3. 跨社区连接：识别连接不同社区的关键边
4. 建议的关系阈值：建议一个合适的关系强度阈值，用于过滤弱连接

请确保你构建的网络保留了原始网络的核心结构和特性，同时去除了冗余和噪声。
""")


# 规划Agent提示词模板
PLANNER_AGENT_SYSTEM_PROMPT = """
你是一个专门负责生成API服务市场实验规则和优化实验方案的智能体。你需要基于行业规范、环境边界和社会关系网络，生成系统约束条件和实验计划。
你的目标是生成可执行的逻辑表达式和优化后的实验方案。

你需要考虑以下因素：
1. 行业规范和标准
2. 环境分布的上下限边界
3. 社会关系网络的结构特性
4. 实验目标和评价指标

请基于提供的信息，生成符合要求的系统约束条件和实验计划。
"""

PLANNER_AGENT_PROMPT_TEMPLATE = Template("""
# 任务背景
你需要为API服务市场生成实验规则和优化实验方案。

# 行业规范概要
$standards

# 环境边界
$boundaries

# 社会关系网络
$relationships

# 任务类型
$task

# 请根据任务类型完成相应工作
${constraint}
""")

# 规划Agent生成约束条件的提示模板
PLANNER_AGENT_GENERATE_CONSTRAINTS_TEMPLATE = """
请生成一组系统约束条件，这些约束条件应该基于行业规范、环境边界和社会关系网络。

约束条件应该包括以下几类：
1. 环境约束：限制环境参数的取值范围和变化速率
2. 社会约束：限制社会关系网络的结构和演化
3. 交互约束：限制环境和社会关系之间的相互作用

请以JSON格式输出约束条件，每个约束条件包含以下字段：
- id: 约束条件的唯一标识符
- description: 约束条件的自然语言描述
- type: 约束条件的类型（environment, social, interaction）
- severity: 约束条件的严重程度（high, medium, low）

示例输出：
```json
[
  {
    "id": "1",
    "description": "在经济衰退期，基础设施类API的需求下降不应超过30%",
    "type": "environment",
    "severity": "high"
  },
  {
    "id": "2",
    "description": "社区内部的连接密度应该至少是跨社区连接密度的3倍",
    "type": "social",
    "severity": "medium"
  }
]
```

请确保生成的约束条件是具体的、可量化的，并且与API服务市场的实际情况相符。
"""

# 规划Agent编译逻辑表达式的提示模板
PLANNER_AGENT_COMPILE_EXPRESSION_TEMPLATE = """
请将以下约束条件转换为可执行的逻辑表达式：

```json
$constraint
```

你需要生成以下内容：
1. 表达式：用自然语言描述的逻辑表达式
2. Python代码：实现该逻辑表达式的Python函数

Python函数应该接受实验方案作为输入，并返回一个布尔值，表示该方案是否满足约束条件。

示例输出：
表达式：对于任意时间点t，如果经济处于衰退期（recession_flag[t] == True），则基础设施类API的需求变化（infrastructure_demand_change[t]）应该大于-0.3

```python
def check_constraint(experiment_plan):
    for t in experiment_plan["years"]:
        if experiment_plan["recession_flag"].get(t, False):
            if experiment_plan["environment_plan"]["基础设施类"].get(t, 0) < -0.3:
                return False
    return True
```

请确保你的逻辑表达式准确反映了约束条件的含义，并且Python代码能够正确实现该逻辑。
"""

# 验证Agent提示词模板
VERIFICATION_AGENT_SYSTEM_PROMPT = """
你是一个专门负责验证API服务市场情景合理性的智能体。你需要验证生成情景的合理性，防止大模型幻觉。
你的目标是确保生成的情景符合领域知识、历史数据和逻辑一致性。

你需要考虑以下因素：
1. 知识一致性：情景是否符合领域知识
2. 历史一致性：情景是否与历史数据趋势相符
3. 逻辑一致性：情景内部是否逻辑自洽
4. 极端合理性：极端情景是否在合理范围内

请基于提供的信息，验证情景的合理性。
"""

VERIFICATION_AGENT_PROMPT_TEMPLATE = Template("""
# 任务背景
你需要验证一个API服务市场情景的合理性，防止大模型幻觉。

# 领域知识
$domain_knowledge

# 历史数据趋势
$historical_trends

# 待验证情景
$scenario

# 请进行以下验证
1. 知识一致性检查：情景是否符合领域知识（给出分数1-10和理由）
2. 历史一致性检查：情景是否与历史数据趋势相符（给出分数1-10和理由）
3. 逻辑一致性检查：情景内部是否逻辑自洽（给出分数1-10和理由）
4. 极端合理性检查：极端情景是否在合理范围内（给出分数1-10和理由）
5. 总体评价：综合以上四点，给出总体评价（合理/部分合理/不合理）和理由
6. 修改建议：如果情景不完全合理，给出具体的修改建议

请确保你的验证是客观、全面的，并基于事实和逻辑进行判断。
""")


# 通用CoT提示词模板
COT_PROMPT_PREFIX = "让我们一步步思考这个问题：\n\n"

COT_SYSTEM_PROMPT = """
你是一个擅长逐步思考问题的AI助手。请一步步分析问题，展示你的思考过程，然后给出最终答案。
""" 