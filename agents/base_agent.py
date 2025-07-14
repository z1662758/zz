"""
API服务生态系统情景生成 - Agent基类
借鉴Generative Agents中的设计，实现基本的思考、感知和记忆功能，self.memories定义初始记忆列表，thinking_chain
"""

import os
import json
import datetime
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.config import LLM_CONFIG


class BaseAgent(ABC):
    """Agent基类，定义了所有Agent的共同接口和基本功能"""

    def __init__(
        self,
        name: str,
        description: str,
        memory_capacity: int = 100,
        logger=None
    ):
        """
        初始化Agent
        
        Args:
            name: Agent名称
            description: Agent描述
            memory_capacity: 记忆容量
            logger: 日志记录器
        """
        self.name = name
        self.description = description
        self.memory_capacity = memory_capacity
        self.logger = logger
        self.memories = []  # 记忆列表
        self.status = {"active": True}  # Agent状态
        self._llm = None  # LLM接口，后续初始化
        self.thinking_chain = []  # 思维链，记录思考过程
        
    def log(self, message: str, level: str = "info") -> None:
        """
        记录日志
        
        Args:
            message: 日志消息
            level: 日志级别，可选值：debug, info, warning, error
        """
        if self.logger:
            if level == "debug":
                self.logger.debug(f"[{self.name}] {message}")
            elif level == "info":
                self.logger.info(f"[{self.name}] {message}")
            elif level == "warning":
                self.logger.warning(f"[{self.name}] {message}")
            elif level == "error":
                self.logger.error(f"[{self.name}] {message}")
        else:
            print(f"[{self.name}] [{level.upper()}] {message}")
            
    def add_memory(self, memory: Dict[str, Any]) -> None:
        """
        添加记忆
        
        Args:
            memory: 记忆内容，字典格式
        """
        # 为记忆添加时间戳
        memory["timestamp"] = datetime.datetime.now().isoformat()
        self.memories.append(memory)
        
        # 如果记忆超过容量，删除最旧的记忆
        if len(self.memories) > self.memory_capacity:
            self.memories.pop(0)
            
        self.log(f"添加记忆: {memory}", level="debug")
        
    def retrieve_memories(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        检索记忆
        
        Args:
            query: 检索关键词
            top_k: 返回的最大记忆数量
            
        Returns:
            符合条件的记忆列表
        """
        # 简单实现：按时间倒序返回最近的记忆
        # 实际应用中可以使用向量数据库实现语义检索
        return sorted(self.memories, key=lambda x: x["timestamp"], reverse=True)[:top_k]
    
    @abstractmethod
    def percept(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        感知环境
        
        Args:
            environment: 环境信息
            
        Returns:
            感知结果
        """
        pass
    
    @abstractmethod
    def think(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """
        思考
        
        Args:
            perception: 感知结果
            
        Returns:
            思考结果
        """
        pass
    
    @abstractmethod
    def act(self, thought: Dict[str, Any]) -> Dict[str, Any]:
        """
        行动
        
        Args:
            thought: 思考结果
            
        Returns:
            行动结果
        """
        pass
    
    def reflect(self) -> Dict[str, Any]:
        """
        反思，总结经验并更新记忆
        
        Returns:
            反思结果
        """
        # 默认实现：提取最近的思考和行动，生成反思
        recent_thoughts = [m for m in self.memories if m.get("type") == "thought"][-5:]
        recent_actions = [m for m in self.memories if m.get("type") == "action"][-5:]
        
        reflection = {
            "type": "reflection",
            "content": f"{self.name}的反思: 基于{len(recent_thoughts)}条思考和{len(recent_actions)}条行动",
            "thoughts": recent_thoughts,
            "actions": recent_actions
        }
        
        self.add_memory(reflection)
        return reflection
    
    def step(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行一个完整的感知-思考-行动循环
        
        Args:
            environment: 环境信息
            
        Returns:
            行动结果
        """
        self.log(f"开始执行步骤，环境: {environment}", level="debug")
        
        # 感知环境
        perception = self.percept(environment)
        self.add_memory({"type": "perception", "content": perception})
        
        # 思考
        thought = self.think(perception)
        self.add_memory({"type": "thought", "content": thought})
        
        # 行动
        action = self.act(thought)
        self.add_memory({"type": "action", "content": action})
        
        # 每隔一定步数进行反思
        if len(self.memories) % 10 == 0:
            self.reflect()
            
        return action
    
    def save_state(self, file_path: str) -> None:
        """
        保存Agent状态
        
        Args:
            file_path: 文件路径
        """
        state = {
            "name": self.name,
            "description": self.description,
            "memories": self.memories,
            "status": self.status,
            "thinking_chain": self.thinking_chain
        }
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
            
        self.log(f"状态已保存到 {file_path}", level="info")
        
    def load_state(self, file_path: str) -> None:
        """
        加载Agent状态
        
        Args:
            file_path: 文件路径
        """
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                state = json.load(f)
                
            self.name = state.get("name", self.name)
            self.description = state.get("description", self.description)
            self.memories = state.get("memories", [])
            self.status = state.get("status", {"active": True})
            self.thinking_chain = state.get("thinking_chain", [])
            
            self.log(f"状态已从 {file_path} 加载", level="info")
        else:
            self.log(f"状态文件 {file_path} 不存在", level="warning")
            
    def __str__(self) -> str:
        return f"{self.name} - {self.description}"
    
    def __repr__(self) -> str:
        return self.__str__() 