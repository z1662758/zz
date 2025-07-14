"""
API服务生态系统情景生成 - LLM接口
提供与大语言模型交互的接口
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Union

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.config import LLM_CONFIG
from utils.logger import get_default_logger

# 尝试导入OpenAI库，如果不存在则提供提示
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("警告: OpenAI库未安装，请使用 'pip install openai' 安装")


class LLMInterface:
    """大语言模型接口，提供与LLM交互的方法"""
    
    def __init__(
        self,
        model_name: str = LLM_CONFIG["model_name"],
        temperature: float = LLM_CONFIG["temperature"],
        max_tokens: int = LLM_CONFIG["max_tokens"],
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        config_path: Optional[str] = None,
        logger=None,
        mock_mode: bool = False
    ):
        """
        初始化LLM接口
        
        Args:
            model_name: 模型名称
            temperature: 温度参数，控制生成文本的随机性
            max_tokens: 生成文本的最大token数
            api_key: API密钥，如果为None，则尝试从环境变量或配置文件获取
            base_url: API基础URL，如果为None，则使用默认URL
            config_path: 配置文件路径，如果提供，则从配置文件加载API密钥和其他配置
            logger: 日志记录器
            mock_mode: 是否使用模拟模式，如果为True，则不会实际调用OpenAI API
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logger or get_default_logger()
        self.mock_mode = mock_mode
        self.base_url = base_url
        
        # 从配置文件加载API密钥和其他配置
        if config_path and os.path.exists(config_path):
            self.logger.info(f"从配置文件 {config_path} 加载配置")
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 提取API密钥
                api_keys = config.get('api_keys', {})
                self.api_key = api_key or api_keys.get('OPENAI_API_KEY')
                
                # 提取LLM配置
                if 'agent' in config and 'think' in config['agent'] and 'llm' in config['agent']['think']:
                    llm_config = config['agent']['think']['llm']
                    self.base_url = base_url or llm_config.get('base_url')
                    
                    # 如果配置文件中有model字段，则使用该字段的值
                    if 'model' in llm_config:
                        self.model_name = llm_config.get('model')
                    
                self.logger.info(f"从配置文件加载了API密钥和LLM配置，使用模型: {self.model_name}")
            except Exception as e:
                self.logger.error(f"加载配置文件失败: {str(e)}")
        else:
            # 如果没有提供配置文件或文件不存在，尝试从环境变量获取API密钥
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        # 检查API密钥
        if not self.api_key:
            self.logger.warning("未提供API密钥，请设置环境变量OPENAI_API_KEY或在初始化时提供")
            self.mock_mode = True
            
        # 初始化OpenAI客户端
        self.client = None
        if OPENAI_AVAILABLE and not self.mock_mode and self.api_key:
            try:
                kwargs = {"api_key": self.api_key}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                    self.logger.info(f"使用自定义API基础URL: {self.base_url}")
                
                self.client = openai.OpenAI(**kwargs)
                self.logger.info(f"OpenAI客户端初始化成功")
            except Exception as e:
                self.logger.error(f"OpenAI客户端初始化失败: {str(e)}")
                self.mock_mode = True
        else:
            if not OPENAI_AVAILABLE:
                self.logger.warning("OpenAI库未安装，使用模拟模式")
            elif not self.api_key:
                self.logger.warning("未提供API密钥，使用模拟模式")
            self.mock_mode = True
            
        self.logger.info(f"LLM接口初始化完成，模型: {self.model_name}，模拟模式: {self.mock_mode}")
        
    def completion(
        self,
        prompt: str,
        system_message: str = "你是一个有用的AI助手。",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        retry_count: int = 3,
        retry_delay: int = 5
    ) -> str:
        """
        生成文本补全
        
        Args:
            prompt: 提示词
            system_message: 系统消息，设置AI助手的角色和行为
            temperature: 温度参数，如果为None则使用默认值
            max_tokens: 生成文本的最大token数，如果为None则使用默认值
            retry_count: 重试次数
            retry_delay: 重试延迟（秒）
            
        Returns:
            生成的文本
        """
        # 如果是模拟模式，使用模拟响应
        if self.mock_mode:
            self.logger.info("使用模拟LLM响应")
            return self.mock_completion(prompt, system_message)
            
        if not OPENAI_AVAILABLE or not self.client:
            self.logger.error("OpenAI库未安装或客户端未初始化，无法生成文本")
            return "LLM接口未正确初始化，无法生成文本"
        
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        for attempt in range(retry_count):
            try:
                self.logger.debug(f"发送请求到LLM，提示词长度: {len(prompt)}")
                self.logger.debug(f"使用模型: {self.model_name}")
                
                # 创建请求参数
                request_params = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                
                # 发送请求
                response = self.client.chat.completions.create(**request_params)
                
                if response.choices and len(response.choices) > 0:
                    text = response.choices[0].message.content
                    self.logger.debug(f"LLM响应成功，响应长度: {len(text)}")
                    return text
                else:
                    self.logger.warning("LLM响应为空")
                    return ""
                    
            except Exception as e:
                self.logger.error(f"LLM请求失败: {str(e)}")
                if attempt < retry_count - 1:
                    self.logger.info(f"等待{retry_delay}秒后重试...")
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"重试{retry_count}次后仍然失败")
                    # 如果所有重试都失败，则切换到模拟模式
                    self.logger.info("切换到模拟模式")
                    return self.mock_completion(prompt, system_message)
    
    def chain_of_thought(
        self,
        prompt: str,
        system_message: str = "你是一个有用的AI助手。请一步步思考问题。",
        cot_prefix: str = "让我们一步步思考这个问题：\n\n",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        使用思维链(Chain of Thought)生成文本
        
        Args:
            prompt: 提示词
            system_message: 系统消息
            cot_prefix: 思维链前缀
            temperature: 温度参数
            max_tokens: 生成文本的最大token数
            
        Returns:
            包含思维链和最终答案的字典
        """
        # 构建CoT提示词
        cot_prompt = f"{prompt}\n\n{cot_prefix}"
        
        # 生成思维链
        response = self.completion(
            cot_prompt,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # 解析思维链和最终答案
        # 这里简单地假设最后一段是答案，实际应用中可能需要更复杂的解析
        thoughts = response.split("\n")
        answer = thoughts[-1] if thoughts else ""
        
        return {
            "thoughts": thoughts,
            "answer": answer,
            "full_response": response
        }
    
    def mock_completion(self, prompt: str, system_message: str = "") -> str:
        """
        模拟LLM响应，用于测试
        
        Args:
            prompt: 提示词
            system_message: 系统消息
            
        Returns:
            模拟的响应文本
        """
        self.logger.info("使用模拟LLM响应")
        
        # 根据提示词和系统消息生成不同的模拟响应
        if "环境Agent" in system_message or "极端情景" in prompt:
            return """
1. 极端情景描述
全球数据隐私监管大爆发

在2023年，全球范围内突然爆发了前所未有的数据隐私监管浪潮。继欧盟GDPR之后，美国、中国、印度等主要经济体同时推出了严格的数据隐私法规，并开始严厉执法。这些法规要求所有API服务提供商必须实现数据本地化存储、完全透明的数据使用流程、用户数据完全可删除权等。违规处罚高达全球年收入的30%，且包含刑事责任。

这场监管风暴来得突然且猛烈，许多API服务提供商毫无准备，被迫紧急调整业务模式和技术架构。大量依赖用户数据的API服务面临生存危机，尤其是跨境数据传输受到严格限制，导致全球化API服务遭受重创。

2. 各类API在这个情景下的需求变化
基础设施类API: +45%，由于企业需要重构技术架构以符合新规定，对基础设施服务的需求大增。
生活服务类API: -25%，由于数据使用受限，许多个性化服务无法维持原有体验，用户流失。
企业管理类API: +30%，合规管理、数据治理相关API需求激增。
社交娱乐类API: -40%，严格的数据共享限制使社交网络效应大幅降低，用户参与度下降。

3. 这个情景的发生概率
发生概率约为15%。虽然数据隐私确实是全球关注的焦点，但各国同时推出严格法规并严厉执法的概率相对较低。

4. 这个情景的持续时间
约24个月。初期的严格监管会持续约一年，随后随着企业适应和监管机构收集反馈，会进入为期约一年的调整期。

5. 这个情景对API服务市场的长期影响
这场危机将重塑API服务市场格局。长期来看，将催生一批以"隐私优先"为核心竞争力的API服务提供商，数据最小化原则将成为API设计的标准。大型科技公司凭借其合规资源优势将进一步巩固市场地位，而中小API提供商数量将减少30%以上。API服务的定价模式也将从数据导向转向功能导向，整体提价15-20%。此外，区域化API服务将兴起，取代部分全球化服务，形成若干相对独立的API生态圈。
            """
        elif "社会Agent" in system_message or "协同网络" in prompt:
            return """
1. 社区划分
根据API之间的调用关系和功能相似性，可以将API划分为以下社区：

社区1（基础设施服务）：包含云存储API、数据库API、服务器管理API等，这些API提供基础的技术设施服务。
社区2（数据处理服务）：包含数据分析API、机器学习API、数据可视化API等，这些API专注于数据处理和分析。
社区3（通信服务）：包含消息推送API、邮件服务API、聊天API等，这些API提供通信功能。
社区4（支付服务）：包含支付处理API、发票生成API、订阅管理API等，这些API处理金融交易。
社区5（内容服务）：包含内容管理API、媒体处理API、搜索API等，这些API处理各种形式的内容。

2. 关键节点
以下是网络中的关键节点（中心性高的API）：

- 云存储API（社区1）：连接度中心性为0.85，是基础设施服务的核心。
- 数据分析API（社区2）：连接度中心性为0.78，是数据处理服务的核心。
- 消息推送API（社区3）：连接度中心性为0.72，是通信服务的核心。
- 支付处理API（社区4）：连接度中心性为0.81，是支付服务的核心。
- 内容管理API（社区5）：连接度中心性为0.76，是内容服务的核心。
- 认证授权API（社区1）：连接度中心性为0.90，连接多个社区，是整个网络的关键节点。

3. 跨社区连接
以下是连接不同社区的关键边：

- 认证授权API（社区1）与支付处理API（社区4）：权重为0.85，反映了支付服务对安全认证的高度依赖。
- 数据分析API（社区2）与内容管理API（社区5）：权重为0.72，反映了内容服务对数据分析的需求。
- 云存储API（社区1）与内容管理API（社区5）：权重为0.68，反映了内容服务对存储服务的依赖。
- 消息推送API（社区3）与支付处理API（社区4）：权重为0.65，反映了支付服务对通知功能的需求。

4. 建议的关系阈值
根据网络分析，建议将关系强度阈值设为0.3。这个阈值可以有效过滤掉弱连接，同时保留网络的核心结构和跨社区连接。具体而言：

阈值0.3可以保留约40%的边，同时保持网络的连通性。
阈值0.3可以保留所有关键节点之间的连接。
阈值0.3可以保留约85%的跨社区连接，确保不同社区之间的协作模式得到保留。
阈值0.3可以过滤掉大多数噪声连接，使网络结构更加清晰。
            """
        elif "规划Agent" in system_message or "实验规则" in prompt:
            return """
1. 系统约束条件
基于行业规范和环境边界，系统运行应遵循以下约束规则：

API调用频率限制：每个API的调用频率不得超过其设计容量的85%，以确保系统稳定性。
数据隐私合规：所有API必须符合GDPR、CCPA等数据隐私法规，确保用户数据的安全处理。
跨类别协作：不同类别的API之间的协作关系必须保持在社会网络分析确定的阈值（0.3）以上。
极端情景适应性：系统必须能够适应环境上下限边界内的所有可能情景，包括需求突增和急剧下降。
资源分配原则：系统资源应按照价值熵最大化原则进行分配，确保生态系统的整体健康度。
失效安全机制：当任何API失效时，系统应能够自动降级服务而非完全崩溃。
多样性维持：系统应保持API生态位的多样性，避免单一类型API的过度主导。

2. 可执行逻辑表达式
以下是系统约束条件对应的可执行逻辑表达式：

```python
# API调用频率限制
def check_api_call_frequency(api_id, call_frequency, capacity):
    return call_frequency <= 0.85 * capacity

# 数据隐私合规
def check_privacy_compliance(api_id, data_processing_methods):
    required_methods = ["consent_management", "data_minimization", "right_to_delete"]
    return all(method in data_processing_methods for method in required_methods)

# 跨类别协作
def check_cross_category_collaboration(api_id1, api_id2, relationship_strength):
    return relationship_strength >= 0.3

# 极端情景适应性
def check_scenario_adaptability(api_id, demand_change):
    upper_boundary = upper_boundaries[api_category_map[api_id]]
    lower_boundary = lower_boundaries[api_category_map[api_id]]
    return lower_boundary <= demand_change <= upper_boundary

# 资源分配原则
def optimize_resource_allocation(api_ecosystem):
    current_entropy = calculate_value_entropy(api_ecosystem)
    return current_entropy >= 0.8 * optimal_entropy

# 失效安全机制
def check_failsafe_mechanism(api_id, dependencies):
    for dep in dependencies:
        if not has_fallback(api_id, dep):
            return False
    return True

# 多样性维持
def check_diversity_maintenance(api_ecosystem):
    category_distribution = calculate_category_distribution(api_ecosystem)
    return max(category_distribution.values()) <= 0.4  # 确保没有单一类别超过40%
```

3. 实验计划
基于上述约束条件和逻辑表达式，设计以下优化后的实验计划：

实验目标：评估不同治理策略对API服务生态系统在极端情景下的影响。

参数设置：
- 时间周期：2006-2022年（17年）
- API类别：基础设施类、生活服务类、企业管理类、社交娱乐类
- 治理策略变量：交易收入分成比例（5%-30%）、平台开放度（0.1-0.9）、质量控制严格度（0.1-0.9）
- 环境变量：根据环境Agent生成的上下限边界
- 社会关系变量：根据社会Agent生成的骨干网络

实验步骤：
1. 初始化：根据2006年的历史数据初始化API生态系统
2. 治理策略设置：设置初始治理策略参数
3. 环境模拟：根据环境边界生成每年的环境情景
4. 系统演化：模拟API生态系统在给定环境和治理策略下的演化过程
5. 指标收集：收集系统效能、API多样性、用户满意度等指标
6. 策略调整：根据收集的指标调整治理策略参数
7. 重复步骤3-6，直至达到收敛条件
8. 结果分析：比较不同治理策略在各种情景下的表现

评价方法：
- 系统效能：使用价值熵指标评估系统的整体健康度
- API多样性：使用Shannon多样性指数评估API类别的多样性
- 用户满意度：使用加权平均的API评分评估用户满意度
- 平台收入：计算平台总收入和增长率
- 抗风险能力：评估系统在极端情景下的稳定性

4. 收敛条件
实验收敛条件定义如下：

1. 系统效能稳定：连续5个时间步长内，系统效能变化率小于1%
2. 策略参数稳定：连续3次策略调整中，参数变化率小于0.5%
3. 用户满意度稳定：连续5个时间步长内，用户满意度变化率小于2%
4. 最大迭代次数：如果达到100次迭代仍未满足上述条件，则强制收敛
5. 目标函数收敛：目标函数（系统效能、多样性、满意度的加权和）的改善率连续3次小于0.1%
            """
        elif "验证Agent" in system_message or "情景合理性" in prompt:
            return """
1. 知识一致性检查
分数：8/10
理由：该情景关于全球数据隐私监管大爆发的描述与现有的领域知识基本一致。GDPR确实引发了全球范围内的数据隐私立法浪潮，美国（如CCPA）、中国（个人信息保护法）等确实都在推进类似立法。但情景中描述的"全球年收入30%的罚款"超出了现有法规的实际处罚力度（GDPR最高为全球营收的4%），这部分略有夸大。同时，所有主要经济体同时执行严格立法的可能性较低，通常会有过渡期和地区差异。

2. 历史一致性检查
分数：7/10
理由：情景描述的趋势与历史数据的发展轨迹基本吻合。数据隐私确实是近年来日益受到重视的议题，且对API市场产生了显著影响。情景中描述的基础设施类API需求增加、合规管理类API需求增加的趋势与历史观察一致。但社交娱乐类API下降40%的幅度可能过大，历史数据表明即使在监管收紧时期，这类API通常能够通过调整商业模式来适应变化，下降幅度一般不会如此剧烈。

3. 逻辑一致性检查
分数：9/10
理由：情景内部的逻辑关系非常自洽。严格的数据隐私法规确实会导致基础设施类API需求增加（企业需要重构技术架构）、企业管理类API需求增加（合规需求）、个性化服务受限（导致生活服务类API需求下降）以及社交网络效应降低（导致社交娱乐类API需求下降）。情景中描述的各类API需求变化与监管变化之间的因果关系清晰合理，持续时间的估计（24个月，包括初期严格监管和后续调整期）也符合类似监管变革的一般周期。

4. 极端合理性检查
分数：8/10
理由：该情景作为一个极端情景是合理的，它描述了一个低概率但高影响的事件，这正是极端情景的特点。15%的发生概率适当反映了这种情景的极端性但又不至于完全不可能。情景中描述的需求变化幅度（+45%到-40%）处于合理的极端范围内，既显著偏离常态，又不至于完全脱离现实。长期影响的描述（如API服务定价模式转变、区域化API服务兴起等）也是对极端情景可能导致的结构性变化的合理推测。

5. 总体评价
评价：合理
理由：综合以上四点检查，该情景在知识一致性、历史一致性、逻辑一致性和极端合理性方面都达到了较高水平。虽然存在一些小的不准确之处（如罚款比例的夸大、社交娱乐类API下降幅度可能过大），但这些不影响情景的整体合理性。该情景成功地描述了一个极端但可能的未来情景，能够为API服务生态系统的风险评估和策略规划提供有价值的参考。

6. 修改建议
为进一步提高情景的合理性，建议进行以下修改：
- 将罚款比例从"全球年收入的30%"调整为更接近现实的"全球年收入的10%"（仍高于现有法规，但更合理）
- 将社交娱乐类API的需求变化从"-40%"调整为"-25%"，更符合此类API的适应能力
- 增加对不同地区实施时间差异的描述，反映监管变革的地区不均衡性
- 增加对企业适应策略的更详细描述，如何在合规的同时维持服务质量
- 考虑添加技术创新因素，如隐私增强技术（PET）的发展如何影响情景演变
            """
        else:
            return f"这是对提示词的模拟响应: {prompt[:50]}..."
    
    @classmethod
    def from_config_file(cls, config_path: str, logger=None, mock=False):
        """
        从配置文件创建LLM接口实例
        
        Args:
            config_path: 配置文件路径
            logger: 日志记录器
            mock: 是否使用模拟模式
            
        Returns:
            LLMInterface实例
        """
        # 创建实例，传递配置文件路径
        return cls(
            config_path=config_path,
            logger=logger,
            mock_mode=mock
        )
    
    def __str__(self) -> str:
        return f"LLMInterface(model={self.model_name}, temperature={self.temperature}, mock_mode={self.mock_mode})" 