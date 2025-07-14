"""
API服务生态系统情景生成 - 环境Agent
负责生成具有长尾分布的极端情景，提前模拟和应对小概率但影响大的事件
"""

import os
import json
import random
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import BaseAgent
from utils.llm_interface import LLMInterface
from utils.prompt_templates import ENVIRONMENT_AGENT_SYSTEM_PROMPT, ENVIRONMENT_AGENT_PROMPT_TEMPLATE
from experiments.config import ENVIRONMENT_AGENT_CONFIG, EXPERIMENT_CONFIG


class EnvironmentAgent(BaseAgent):
    """
    环境Agent，负责生成具有长尾分布的极端情景
    """
    
    def __init__(
        self,
        name: str = "环境Agent",
        description: str = "负责生成具有长尾分布的极端情景",
        memory_capacity: int = 100,
        logger=None,
        config: Dict[str, Any] = None,
        llm_interface: Optional[LLMInterface] = None
    ):
        """
        初始化环境Agent
        
        Args:
            name: Agent名称
            description: Agent描述
            memory_capacity: 记忆容量
            logger: 日志记录器
            config: 配置信息
            llm_interface: LLM接口，如果为None则创建新的接口
        """
        super().__init__(name, description, memory_capacity, logger)
        
        # 加载配置
        self.config = config or ENVIRONMENT_AGENT_CONFIG
        
        # 初始化LLM接口
        self._llm = llm_interface or LLMInterface(logger=logger)
        
        # 历史数据
        self.historical_data = {}
        
        # 领域知识
        self.domain_knowledge = []
        
        # 对抗性提示
        self.adversarial_prompts = self.config.get("adversarial_prompts", [])
        
        # 生成的情景
        self.scenarios = []
        
        # 环境分布的上下限边界
        self.upper_boundary = {}
        self.lower_boundary = {}
        
        self.log("环境Agent初始化完成")
        
    def load_historical_data(self, data_path: str) -> None:
        """
        加载历史数据
        
        Args:
            data_path: 数据文件路径
        """
        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                self.historical_data = json.load(f)
            
            self.log(f"历史数据加载完成，包含{len(self.historical_data)}条记录")
        else:
            self.log(f"历史数据文件{data_path}不存在", level="warning")
            
    def load_domain_knowledge(self, knowledge_path: str) -> None:
        """
        加载领域知识
        
        Args:
            knowledge_path: 知识文件路径
        """
        if os.path.exists(knowledge_path):
            with open(knowledge_path, "r", encoding="utf-8") as f:
                self.domain_knowledge = json.load(f)
            
            self.log(f"领域知识加载完成，包含{len(self.domain_knowledge)}条记录")
        else:
            self.log(f"领域知识文件{knowledge_path}不存在", level="warning")
    
    def percept(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        感知环境
        
        Args:
            environment: 环境信息
            
        Returns:
            感知结果
        """
        perception = {
            "historical_data": environment.get("historical_data", self.historical_data),
            "domain_knowledge": environment.get("domain_knowledge", self.domain_knowledge),
            "time_range": environment.get("time_range", "2006-2022"),
            "api_categories": environment.get("api_categories", EXPERIMENT_CONFIG["api_categories"])
        }
        
        self.log(f"感知环境: {perception}", level="debug")
        return perception
    
    def think(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """
        思考，生成极端情景
        
        Args:
            perception: 感知结果
            
        Returns:
            思考结果，包含生成的极端情景
        """
        self.log("开始思考，生成极端情景")
        
        # 提取感知信息
        historical_data = perception["historical_data"]
        domain_knowledge = perception["domain_knowledge"]
        time_range = perception["time_range"]
        api_categories = perception["api_categories"]
        
        # 构建历史趋势描述
        historical_trends = self._build_historical_trends(historical_data)
        
        # 选择对抗性提示
        adversarial_prompt = random.choice(self.adversarial_prompts)
        
        # 构建领域知识描述
        domain_knowledge_text = self._build_domain_knowledge_text(domain_knowledge)
        
        # 构建提示词
        prompt = ENVIRONMENT_AGENT_PROMPT_TEMPLATE.substitute(
            time_range=time_range,
            api_categories=", ".join(api_categories),
            historical_trends=historical_trends,
            domain_knowledge=domain_knowledge_text,
            adversarial_prompt=adversarial_prompt
        )
        
        # 使用LLM生成极端情景
        try:
            response = self._llm.completion(
                prompt=prompt,
                system_message=ENVIRONMENT_AGENT_SYSTEM_PROMPT,
                temperature=0.7  # 使用较高的温度以增加多样性
            )
            
            # 解析响应
            scenario = self._parse_scenario_response(response)
            
            # 添加到情景列表
            self.scenarios.append(scenario)
            
            # 更新思维链
            self.thinking_chain.append({
                "type": "scenario_generation",
                "prompt": prompt,
                "response": response,
                "parsed_scenario": scenario
            })
            
            thought = {
                "scenario": scenario,
                "reasoning": f"基于历史数据和对抗性提示'{adversarial_prompt}'生成了极端情景"
            }
            
            self.log(f"生成极端情景: {scenario['title']}")
            return thought
            
        except Exception as e:
            self.log(f"生成极端情景失败: {str(e)}", level="error")
            return {
                "error": str(e),
                "reasoning": "生成极端情景失败"
            }
    
    def act(self, thought: Dict[str, Any]) -> Dict[str, Any]:
        """
        行动，计算环境分布的上下限边界
        
        Args:
            thought: 思考结果
            
        Returns:
            行动结果，包含环境分布的上下限边界
        """
        self.log("开始行动，计算环境分布的上下限边界")
        
        if "error" in thought:
            return {
                "status": "error",
                "error": thought["error"]
            }
        
        scenario = thought["scenario"]
        
        # 提取各类API的需求变化
        demand_changes = scenario.get("demand_changes", {})
        
        # 更新环境分布的上下限边界
        self._update_boundaries(demand_changes)
        
        return {
            "status": "success",
            "scenario": scenario,
            "upper_boundary": self.upper_boundary,
            "lower_boundary": self.lower_boundary
        }
    
    def generate_extreme_scenario(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成极端情景的完整流程
        
        Args:
            environment: 环境信息
            
        Returns:
            生成的极端情景和环境分布的上下限边界
        """
        self.log("开始生成极端情景")
        
        # 感知环境
        perception = self.percept(environment)
        
        # 思考，生成极端情景
        thought = self.think(perception)
        
        # 行动，计算环境分布的上下限边界
        action = self.act(thought)
        
        return action
    
    def _build_historical_trends(self, historical_data: Dict[str, Any]) -> str:
        """
        构建历史趋势描述
        
        Args:
            historical_data: 历史数据
            
        Returns:
            历史趋势描述
        """
        # 如果没有历史数据，返回模拟的趋势描述
        if not historical_data:
            return """
            基础设施类API: 2006年至2010年稳步增长，2010年至2015年快速增长，2015年后增长放缓。
            生活服务类API: 2006年至2012年缓慢增长，2012年至2018年快速增长，2018年后趋于稳定。
            企业管理类API: 2006年至2014年持续稳定增长，2014年后增长加速。
            社交娱乐类API: 2006年至2008年缓慢增长，2008年至2016年爆发式增长，2016年后增长放缓。
            """
        
        # TODO: 根据实际历史数据构建趋势描述
        trends = []
        for category, data in historical_data.items():
            trends.append(f"{category}: {data['trend_description']}")
        
        return "\n".join(trends)
    
    def _build_domain_knowledge_text(self, domain_knowledge: List[Dict[str, Any]]) -> str:
        """
        构建领域知识描述
        
        Args:
            domain_knowledge: 领域知识
            
        Returns:
            领域知识描述
        """
        # 如果没有领域知识，返回模拟的知识描述
        if not domain_knowledge:
            return """
            1. API经济在2013年前后出现爆发式增长，主要由移动应用和云服务驱动。
            2. 数据隐私政策（如GDPR）对API市场产生了显著影响，特别是对数据服务类API。
            3. 基础设施类API通常具有较高的稳定性和较低的波动性。
            4. 社交娱乐类API受技术趋势和用户行为变化影响最大。
            5. API市场存在明显的网络效应，成功的API平台往往吸引更多的开发者和用户。
            """
        
        # TODO: 根据实际领域知识构建描述
        knowledge_texts = []
        for i, k in enumerate(domain_knowledge):
            knowledge_texts.append(f"{i+1}. {k['description']}")
        
        return "\n".join(knowledge_texts)
    
    def _parse_scenario_response(self, response: str) -> Dict[str, Any]:
        """
        解析LLM生成的情景响应
        
        Args:
            response: LLM响应
            
        Returns:
            解析后的情景
        """
        # 简单解析，实际应用中可能需要更复杂的解析逻辑
        lines = response.strip().split("\n")
        
        scenario = {
            "title": "未命名情景",
            "description": "",
            "demand_changes": {},
            "probability": 0.0,
            "duration": 0,
            "long_term_impact": ""
        }
        
        # 提取情景描述
        description_lines = []
        demand_changes = {}
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 检测章节
            if line.startswith("1. 极端情景描述"):
                current_section = "description"
                continue
            elif line.startswith("2. 各类API在这个情景下的需求变化"):
                current_section = "demand_changes"
                continue
            elif line.startswith("3. 这个情景的发生概率"):
                current_section = "probability"
                continue
            elif line.startswith("4. 这个情景的持续时间"):
                current_section = "duration"
                continue
            elif line.startswith("5. 这个情景对API服务市场的长期影响"):
                current_section = "long_term_impact"
                continue
                
            # 根据当前章节处理内容
            if current_section == "description":
                if not scenario["title"] or scenario["title"] == "未命名情景":
                    scenario["title"] = line  # 第一行作为标题
                else:
                    description_lines.append(line)
            elif current_section == "demand_changes":
                # 尝试提取需求变化
                for category in EXPERIMENT_CONFIG["api_categories"]:
                    if category in line:
                        # 尝试提取百分比
                        import re
                        matches = re.findall(r'([+-]?\d+(?:\.\d+)?)%', line)
                        if matches:
                            demand_changes[category] = float(matches[0])
            elif current_section == "probability":
                # 尝试提取概率
                import re
                matches = re.findall(r'(\d+(?:\.\d+)?)%', line)
                if matches:
                    scenario["probability"] = float(matches[0]) / 100.0
            elif current_section == "duration":
                # 尝试提取持续时间
                import re
                matches = re.findall(r'(\d+)个月', line)
                if matches:
                    scenario["duration"] = int(matches[0])
                else:
                    matches = re.findall(r'(\d+)', line)
                    if matches:
                        scenario["duration"] = int(matches[0])
            elif current_section == "long_term_impact":
                if scenario["long_term_impact"]:
                    scenario["long_term_impact"] += " " + line
                else:
                    scenario["long_term_impact"] = line
        
        # 设置描述和需求变化
        scenario["description"] = "\n".join(description_lines)
        scenario["demand_changes"] = demand_changes
        
        # 如果没有提取到需求变化，设置默认值
        if not demand_changes:
            for category in EXPERIMENT_CONFIG["api_categories"]:
                scenario["demand_changes"][category] = random.uniform(-30, 50)
        
        return scenario
    
    def _update_boundaries(self, demand_changes: Dict[str, float]) -> None:
        """
        更新环境分布的上下限边界
        
        Args:
            demand_changes: 各类API的需求变化
        """
        for category, change in demand_changes.items():
            if category not in self.upper_boundary or change > self.upper_boundary.get(category, -float('inf')):
                self.upper_boundary[category] = change
                
            if category not in self.lower_boundary or change < self.lower_boundary.get(category, float('inf')):
                self.lower_boundary[category] = change
                
        self.log(f"更新环境分布的上下限边界: 上限={self.upper_boundary}, 下限={self.lower_boundary}", level="debug")
    
    def get_boundaries(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        获取环境分布的上下限边界
        
        Returns:
            环境分布的上下限边界
        """
        return self.upper_boundary, self.lower_boundary
    
    def get_scenarios(self) -> List[Dict[str, Any]]:
        """
        获取生成的情景
        
        Returns:
            生成的情景列表
        """
        return self.scenarios 