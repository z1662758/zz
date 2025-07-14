"""
API服务生态系统情景生成 - 规划Agent
负责生成API服务市场实验规则和优化实验方案
"""

import os
import json
import random
from typing import Dict, List, Any, Optional, Union, Tuple

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import BaseAgent
from utils.llm_interface import LLMInterface
from utils.prompt_templates import (
    PLANNER_AGENT_SYSTEM_PROMPT, 
    PLANNER_AGENT_PROMPT_TEMPLATE,
    PLANNER_AGENT_GENERATE_CONSTRAINTS_TEMPLATE,
    PLANNER_AGENT_COMPILE_EXPRESSION_TEMPLATE
)
from experiments.config import PLANNER_AGENT_CONFIG, EXPERIMENT_CONFIG


class PlannerAgent(BaseAgent):
    """
    规划Agent，负责生成API服务市场实验规则和优化实验方案
    """
    
    def __init__(
        self,
        name: str = "规划Agent",
        description: str = "负责生成API服务市场实验规则和优化实验方案",
        memory_capacity: int = 100,
        logger=None,
        config: Dict[str, Any] = None,
        llm_interface: Optional[LLMInterface] = None
    ):
        """
        初始化规划Agent
        
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
        self.config = config or PLANNER_AGENT_CONFIG
        
        # 初始化LLM接口
        self._llm = llm_interface or LLMInterface(logger=logger)
        
        # 行业规范
        self.standards = []
        
        # 环境边界
        self.environment_boundaries = {}
        
        # 社会关系网络
        self.social_relationships = {}
        
        # 生成的约束条件
        self.constraints = []
        
        # 编译后的逻辑表达式
        self.compiled_expressions = []
        
        # 优化后的实验方案
        self.optimized_plans = []
        
        self.log("规划Agent初始化完成")
        
    def load_standards(self, standards_path: str) -> None:
        """
        加载行业规范
        
        Args:
            standards_path: 规范文件路径
        """
        if os.path.exists(standards_path):
            with open(standards_path, "r", encoding="utf-8") as f:
                self.standards = json.load(f)
            
            self.log(f"行业规范加载完成，包含{len(self.standards)}条记录")
        else:
            self.log(f"行业规范文件{standards_path}不存在", level="warning")
    
    def set_environment_boundaries(self, boundaries: Dict[str, Any]) -> None:
        """
        设置环境边界
        
        Args:
            boundaries: 环境边界信息
        """
        self.environment_boundaries = boundaries
        self.log(f"环境边界设置完成")
    
    def set_social_relationships(self, relationships: Dict[str, Any]) -> None:
        """
        设置社会关系网络
        
        Args:
            relationships: 社会关系网络信息
        """
        self.social_relationships = relationships
        self.log(f"社会关系网络设置完成")
    
    def percept(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        感知环境
        
        Args:
            environment: 环境信息
            
        Returns:
            感知结果
        """
        perception = {
            "standards": environment.get("standards", self.standards),
            "environment_boundaries": environment.get("environment_boundaries", self.environment_boundaries),
            "social_relationships": environment.get("social_relationships", self.social_relationships),
            "task": environment.get("task", "generate_constraints"),
            "constraint": environment.get("constraint", None)
        }
        
        self.log(f"感知环境: {perception}", level="debug")
        return perception
    
    def think(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """
        思考，根据任务类型生成约束条件或编译逻辑表达式
        
        Args:
            perception: 感知结果
            
        Returns:
            思考结果
        """
        self.log("开始思考，执行规划任务")
        
        # 提取感知信息
        standards = perception["standards"]
        environment_boundaries = perception["environment_boundaries"]
        social_relationships = perception["social_relationships"]
        task = perception["task"]
        constraint = perception["constraint"]
        
        # 根据任务类型执行不同的操作
        if task == "generate_constraints":
            return self._generate_constraints(standards, environment_boundaries, social_relationships)
        elif task == "compile_expression":
            return self._compile_expression(constraint)
        elif task == "optimize_plan":
            return self._optimize_plan(standards, environment_boundaries, social_relationships)
        else:
            self.log(f"未知任务类型: {task}", level="error")
            return {"error": f"未知任务类型: {task}"}
    
    def act(self, thought: Dict[str, Any]) -> Dict[str, Any]:
        """
        行动，保存生成的结果
        
        Args:
            thought: 思考结果
            
        Returns:
            行动结果
        """
        if "error" in thought:
            self.log(f"执行失败: {thought['error']}", level="error")
            return {"status": "failed", "error": thought["error"]}
        
        if "constraints" in thought:
            # 保存生成的约束条件
            self.constraints = thought["constraints"]
            self.log(f"生成了{len(self.constraints)}个约束条件")
            
            return {
                "status": "success",
                "message": f"生成了{len(self.constraints)}个约束条件",
                "constraints": self.constraints
            }
        
        elif "expression" in thought:
            # 保存编译后的逻辑表达式
            expression = {
                "description": thought["expression"],
                "code": thought["code"]
            }
            self.compiled_expressions.append(expression)
            self.log(f"编译了逻辑表达式: {expression['description']}")
            
            return {
                "status": "success",
                "message": f"编译了逻辑表达式",
                "expression": expression
            }
        
        elif "optimized_plan" in thought:
            # 保存优化后的实验方案
            self.optimized_plans.append(thought["optimized_plan"])
            self.log(f"优化了实验方案")
            
            return {
                "status": "success",
                "message": f"优化了实验方案",
                "plan": thought["optimized_plan"]
            }
        
        else:
            self.log("未知思考结果类型", level="warning")
            return {"status": "unknown", "thought": thought}
    
    def _generate_constraints(
        self, 
        standards: List[Dict[str, Any]], 
        environment_boundaries: Dict[str, Any], 
        social_relationships: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        生成约束条件
        
        Args:
            standards: 行业规范
            environment_boundaries: 环境边界
            social_relationships: 社会关系网络
            
        Returns:
            生成的约束条件
        """
        self.log("开始生成约束条件")
        
        # 格式化行业规范
        standards_text = json.dumps(standards, ensure_ascii=False, indent=2)
        
        # 格式化环境边界
        boundaries_text = json.dumps(environment_boundaries, ensure_ascii=False, indent=2)
        
        # 格式化社会关系网络
        relationships_text = json.dumps(social_relationships, ensure_ascii=False, indent=2)
        
        # 构建提示词
        prompt = PLANNER_AGENT_PROMPT_TEMPLATE.substitute(
            standards=standards_text,
            boundaries=boundaries_text,
            relationships=relationships_text,
            task="generate_constraints",
            constraint=PLANNER_AGENT_GENERATE_CONSTRAINTS_TEMPLATE
        )
        
        try:
            # 使用LLM生成约束条件
            response = self._llm.completion(
                prompt=prompt,
                system_message=PLANNER_AGENT_SYSTEM_PROMPT,
                temperature=0.3  # 使用较低的温度以增加一致性
            )
            
            # 解析响应中的JSON
            constraints = self._extract_json_from_response(response)
            
            # 更新思维链
            self.thinking_chain.append({
                "type": "constraint_generation",
                "prompt": prompt,
                "response": response,
                "parsed_constraints": constraints
            })
            
            return {"constraints": constraints}
            
        except Exception as e:
            self.log(f"生成约束条件失败: {str(e)}", level="error")
            return {"error": str(e)}
    
    def _compile_expression(self, constraint: Dict[str, Any]) -> Dict[str, Any]:
        """
        编译逻辑表达式
        
        Args:
            constraint: 约束条件
            
        Returns:
            编译后的逻辑表达式
        """
        self.log(f"开始编译约束条件: {constraint['id']}")
        
        # 格式化约束条件
        constraint_text = json.dumps(constraint, ensure_ascii=False, indent=2)
        
        # 构建提示词
        prompt = PLANNER_AGENT_COMPILE_EXPRESSION_TEMPLATE.replace(
            "$constraint", constraint_text
        )
        
        try:
            # 使用LLM编译逻辑表达式
            response = self._llm.completion(
                prompt=prompt,
                system_message=PLANNER_AGENT_SYSTEM_PROMPT,
                temperature=0.2  # 使用较低的温度以增加一致性
            )
            
            # 解析响应
            expression, code = self._parse_expression_response(response)
            
            # 更新思维链
            self.thinking_chain.append({
                "type": "expression_compilation",
                "prompt": prompt,
                "response": response,
                "parsed_expression": expression,
                "parsed_code": code
            })
            
            return {
                "expression": expression,
                "code": code
            }
            
        except Exception as e:
            self.log(f"编译逻辑表达式失败: {str(e)}", level="error")
            return {"error": str(e)}
    
    def _optimize_plan(
        self, 
        standards: List[Dict[str, Any]], 
        environment_boundaries: Dict[str, Any], 
        social_relationships: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        优化实验方案
        
        Args:
            standards: 行业规范
            environment_boundaries: 环境边界
            social_relationships: 社会关系网络
            
        Returns:
            优化后的实验方案
        """
        self.log("开始优化实验方案")
        
        # 此处实现优化实验方案的逻辑
        # 由于优化实验方案比较复杂，这里暂时返回一个简单的方案
        optimized_plan = {
            "name": "优化后的实验方案",
            "description": "基于行业规范和环境边界优化的实验方案",
            "parameters": {
                "learning_rate": self.config.get("learning_rate", 0.01),
                "optimization_iterations": self.config.get("optimization_iterations", 100),
                "convergence_threshold": self.config.get("convergence_threshold", 0.001)
            },
            "constraints": self.constraints
        }
        
        return {"optimized_plan": optimized_plan}
    
    def _extract_json_from_response(self, response: str) -> List[Dict[str, Any]]:
        """
        从响应中提取JSON
        
        Args:
            response: LLM响应
            
        Returns:
            提取的JSON对象
        """
        try:
            # 查找JSON开始和结束的位置
            start = response.find("[")
            end = response.rfind("]") + 1
            
            if start != -1 and end != -1:
                json_str = response[start:end]
                return json.loads(json_str)
            else:
                # 尝试查找```json和```之间的内容
                start = response.find("```json")
                if start != -1:
                    start = response.find("[", start)
                    end = response.find("```", start)
                    if end != -1:
                        json_str = response[start:end].strip()
                        return json.loads(json_str)
                
                self.log("无法从响应中提取JSON", level="warning")
                return []
                
        except json.JSONDecodeError as e:
            self.log(f"JSON解析错误: {str(e)}", level="error")
            return []
    
    def _parse_expression_response(self, response: str) -> Tuple[str, str]:
        """
        解析表达式响应
        
        Args:
            response: LLM响应
            
        Returns:
            表达式和代码
        """
        # 查找表达式
        expression_start = response.find("表达式：")
        if expression_start == -1:
            expression_start = 0
        else:
            expression_start += 4
        
        code_start = response.find("```python")
        if code_start == -1:
            expression = response[expression_start:].strip()
            code = ""
        else:
            expression = response[expression_start:code_start].strip()
            code_end = response.find("```", code_start + 8)
            if code_end == -1:
                code = response[code_start + 8:].strip()
            else:
                code = response[code_start + 8:code_end].strip()
        
        return expression, code
    
    def generate_constraints(self) -> List[Dict[str, Any]]:
        """
        生成约束条件
        
        Returns:
            生成的约束条件列表
        """
        environment = {
            "standards": self.standards,
            "environment_boundaries": self.environment_boundaries,
            "social_relationships": self.social_relationships,
            "task": "generate_constraints"
        }
        
        perception = self.percept(environment)
        thought = self.think(perception)
        action = self.act(thought)
        
        return action.get("constraints", [])
    
    def compile_constraint(self, constraint: Dict[str, Any]) -> Dict[str, Any]:
        """
        编译约束条件
        
        Args:
            constraint: 约束条件
            
        Returns:
            编译后的表达式
        """
        environment = {
            "task": "compile_expression",
            "constraint": constraint
        }
        
        perception = self.percept(environment)
        thought = self.think(perception)
        action = self.act(thought)
        
        return action.get("expression", {})
    
    def get_constraints(self) -> List[Dict[str, Any]]:
        """
        获取生成的约束条件
        
        Returns:
            约束条件列表
        """
        return self.constraints
    
    def get_compiled_expressions(self) -> List[Dict[str, Any]]:
        """
        获取编译后的逻辑表达式
        
        Returns:
            逻辑表达式列表
        """
        return self.compiled_expressions 