"""
API服务生态系统情景生成 - 验证Agent
负责评估情景的合理性，确保生成的情景符合实际情况，避免幻觉问题
"""

import os
import re
import json
import logging
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.config import VERIFICATION_AGENT_CONFIG
from utils.llm_interface import LLMInterface
from agents.base_agent import BaseAgent

# 验证Agent的系统提示词
VALIDATION_AGENT_SYSTEM_PROMPT = """你是一个专业的API生态系统验证专家，负责评估生成的情景是否合理、一致且符合实际情况。
你需要根据历史数据、领域知识和逻辑一致性来验证情景，避免幻觉和不合理的预测。
请保持客观、严谨的态度，对每个情景进行全面评估。"""

# 验证Agent的提示词模板
VALIDATION_AGENT_PROMPT_TEMPLATE = """
请对以下API生态系统情景进行验证，评估其合理性、一致性和可信度：

情景标题: ${title}
情景描述: ${description}
时间范围: ${time_range}
API类别变化: 
${api_changes}

历史数据参考:
${historical_data}

领域知识参考:
${domain_knowledge}

请从以下三个方面进行验证：
1. 知识一致性：情景是否符合已知的API生态系统发展规律和领域知识
2. 历史一致性：情景是否与历史数据趋势相符，变化是否合理
3. 逻辑一致性：情景内部是否存在逻辑矛盾或不合理之处

对于每个方面，请给出评分（1-10分）和详细分析。如发现问题，请指出具体问题并提出修正建议。
最后，请给出总体评估结果（通过/需要修改/拒绝）和综合评分（1-10分）。
"""


class ValidationAgent(BaseAgent):
    """验证Agent，负责评估情景的合理性，确保生成的情景符合实际情况"""

    def __init__(
        self,
        name: str = "验证Agent",
        description: str = "负责评估情景的合理性，确保生成的情景符合实际情况",
        memory_capacity: int = 100,
        logger: Optional[logging.Logger] = None,
        config: Dict[str, Any] = None,
        llm_interface: Optional[LLMInterface] = None
    ):
        """
        初始化验证Agent
        
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
        self.config = config or VERIFICATION_AGENT_CONFIG
        
        # 初始化LLM接口
        self._llm = llm_interface or LLMInterface(logger=logger)
        
        # 验证结果
        self.validation_results = {}
        
        # 幻觉检测阈值
        self.hallucination_threshold = self.config.get("hallucination_threshold", 0.8)
        
        # 一致性检查样本数量
        self.consistency_check_samples = self.config.get("consistency_check_samples", 10)
        
        # 验证方法
        self.validation_methods = self.config.get("validation_methods", 
                                                ["knowledge_consistency", 
                                                 "historical_consistency", 
                                                 "logical_consistency"])
        
        self.log("验证Agent初始化完成")
    
    def percept(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        感知环境
        
        Args:
            environment: 环境信息，包含待验证的情景、历史数据和领域知识
            
        Returns:
            感知结果
        """
        perception = {
            "scenario": environment.get("scenario", {}),
            "historical_data": environment.get("historical_data", {}),
            "domain_knowledge": environment.get("domain_knowledge", []),
            "time_range": environment.get("time_range", "")
        }
        
        self.log(f"感知环境: 情景={perception['scenario'].get('title', '无标题')}", level="debug")
        return perception
    
    def think(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """
        思考，验证情景的合理性
        
        Args:
            perception: 感知结果
            
        Returns:
            思考结果，包含验证评估
        """
        self.log("开始思考，验证情景的合理性")
        
        # 提取感知信息
        scenario = perception["scenario"]
        historical_data = perception["historical_data"]
        domain_knowledge = perception["domain_knowledge"]
        time_range = perception["time_range"]
        
        # 检查情景是否为空
        if not scenario:
            self.log("情景为空，无法进行验证", level="error")
            return {
                "error": "情景为空，无法进行验证",
                "reasoning": "需要提供有效的情景数据"
            }
        
        # 准备API类别变化的文本描述
        api_changes_text = ""
        if "api_changes" in scenario:
            for category, changes in scenario["api_changes"].items():
                api_changes_text += f"{category}: {changes}\n"
        
        # 准备历史数据的文本描述
        historical_data_text = ""
        if historical_data:
            for category, years in historical_data.items():
                historical_data_text += f"{category}: {years}\n"
        else:
            historical_data_text = "无历史数据"
        
        # 准备领域知识的文本描述
        domain_knowledge_text = ""
        if domain_knowledge:
            for i, knowledge in enumerate(domain_knowledge):
                domain_knowledge_text += f"{i+1}. {knowledge}\n"
        else:
            domain_knowledge_text = "无领域知识"
        
        # 构建提示词
        prompt = VALIDATION_AGENT_PROMPT_TEMPLATE.replace("${title}", scenario.get("title", "无标题"))
        prompt = prompt.replace("${description}", scenario.get("description", "无描述"))
        prompt = prompt.replace("${time_range}", time_range)
        prompt = prompt.replace("${api_changes}", api_changes_text)
        prompt = prompt.replace("${historical_data}", historical_data_text)
        prompt = prompt.replace("${domain_knowledge}", domain_knowledge_text)
        
        # 使用LLM验证情景
        try:
            response = self._llm.completion(
                prompt=prompt,
                system_message=VALIDATION_AGENT_SYSTEM_PROMPT,
                temperature=0.3  # 使用较低的温度以增加一致性
            )
            
            # 解析验证结果
            validation_result = self._parse_validation_response(response)
            
            # 更新思维链
            self.thinking_chain.append({
                "type": "scenario_validation",
                "prompt": prompt,
                "response": response,
                "parsed_result": validation_result
            })
            
            # 记录验证结果
            self.validation_results[scenario.get("title", "无标题")] = validation_result
            
            thought = {
                "validation_result": validation_result,
                "reasoning": f"基于历史数据和领域知识验证了情景的合理性，总体评分: {validation_result['overall_score']}"
            }
            
            self.log(f"情景验证完成，总体评估: {validation_result['overall_assessment']}, 评分: {validation_result['overall_score']}")
            return thought
            
        except Exception as e:
            self.log(f"验证情景失败: {str(e)}", level="error")
            return {
                "error": str(e),
                "reasoning": "验证情景失败"
            }
    
    def act(self, thought: Dict[str, Any]) -> Dict[str, Any]:
        """
        行动，生成验证报告
        
        Args:
            thought: 思考结果
            
        Returns:
            行动结果，包含验证报告
        """
        self.log("开始行动，生成验证报告")
        
        if "error" in thought:
            return {
                "status": "error",
                "error": thought["error"]
            }
        
        validation_result = thought["validation_result"]
        
        # 根据验证结果决定是否接受情景
        if validation_result["overall_assessment"] == "通过":
            status = "accepted"
            message = "情景验证通过，符合实际情况"
        elif validation_result["overall_assessment"] == "需要修改":
            status = "needs_revision"
            message = "情景需要修改，存在一些不合理之处"
        else:
            status = "rejected"
            message = "情景被拒绝，存在严重的不合理之处"
        
        # 生成验证报告
        validation_report = {
            "status": status,
            "message": message,
            "validation_result": validation_result,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return {
            "status": "success",
            "validation_report": validation_report
        }
    
    def validate_scenario(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证情景的完整流程
        
        Args:
            environment: 环境信息，包含待验证的情景、历史数据和领域知识
            
        Returns:
            验证报告
        """
        self.log("开始验证情景")
        
        # 感知环境
        perception = self.percept(environment)
        
        # 思考，验证情景
        thought = self.think(perception)
        
        # 行动，生成验证报告
        action = self.act(thought)
        
        return action
    
    def _parse_validation_response(self, response: str) -> Dict[str, Any]:
        """
        解析LLM的验证响应
        
        Args:
            response: LLM的响应文本
            
        Returns:
            解析后的验证结果
        """
        try:
            # 提取知识一致性评分
            knowledge_score_match = re.search(r"知识一致性[：:]\s*(\d+)[分/]", response)
            knowledge_score = int(knowledge_score_match.group(1)) if knowledge_score_match else 5
            
            # 提取历史一致性评分
            historical_score_match = re.search(r"历史一致性[：:]\s*(\d+)[分/]", response)
            historical_score = int(historical_score_match.group(1)) if historical_score_match else 5
            
            # 提取逻辑一致性评分
            logical_score_match = re.search(r"逻辑一致性[：:]\s*(\d+)[分/]", response)
            logical_score = int(logical_score_match.group(1)) if logical_score_match else 5
            
            # 提取总体评估
            if "通过" in response.lower():
                overall_assessment = "通过"
            elif "修改" in response.lower():
                overall_assessment = "需要修改"
            elif "拒绝" in response.lower():
                overall_assessment = "拒绝"
            else:
                overall_assessment = "需要修改"  # 默认值
            
            # 提取综合评分
            overall_score_match = re.search(r"综合评分[：:]\s*(\d+)[分/]", response)
            overall_score = int(overall_score_match.group(1)) if overall_score_match else (knowledge_score + historical_score + logical_score) // 3
            
            # 构建验证结果
            validation_result = {
                "knowledge_consistency": {
                    "score": knowledge_score,
                    "analysis": self._extract_section(response, "知识一致性")
                },
                "historical_consistency": {
                    "score": historical_score,
                    "analysis": self._extract_section(response, "历史一致性")
                },
                "logical_consistency": {
                    "score": logical_score,
                    "analysis": self._extract_section(response, "逻辑一致性")
                },
                "overall_assessment": overall_assessment,
                "overall_score": overall_score,
                "suggestions": self._extract_section(response, "修正建议") or self._extract_section(response, "建议")
            }
            
            return validation_result
            
        except Exception as e:
            self.log(f"解析验证响应失败: {str(e)}", level="error")
            # 返回默认值
            return {
                "knowledge_consistency": {"score": 5, "analysis": "解析失败"},
                "historical_consistency": {"score": 5, "analysis": "解析失败"},
                "logical_consistency": {"score": 5, "analysis": "解析失败"},
                "overall_assessment": "需要修改",
                "overall_score": 5,
                "suggestions": "无法解析验证响应，请手动检查情景合理性。"
            }
    
    def _extract_section(self, text: str, section_name: str) -> str:
        """
        从文本中提取指定部分的内容
        
        Args:
            text: 文本内容
            section_name: 部分名称
            
        Returns:
            提取的内容
        """
        pattern = rf"{section_name}[：:](.*?)(?:\d+\.|$|综合评分|总体评估|{section_name})"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
    
    def save_validation_results(self, output_dir: str) -> bool:
        """
        保存验证结果
        
        Args:
            output_dir: 输出目录
        
        Returns:
            bool: 保存是否成功
        """
        try:
            self.log(f"开始保存验证结果到目录: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存验证结果
            with open(os.path.join(output_dir, 'validation_results.json'), 'w', encoding='utf-8') as f:
                json.dump(self.validation_results, f, ensure_ascii=False, indent=2)
            
            self.log(f"成功保存验证结果到目录: {output_dir}")
            return True
            
        except Exception as e:
            self.log(f"保存验证结果失败: {str(e)}", level="error")
            return False 