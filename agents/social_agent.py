"""
API服务生态系统情景生成 - 社会Agent
负责构建API服务市场协同网络，反映主体间交互关系
"""

import os
import json
import logging
import networkx as nx
import numpy as np
import community as community_louvain
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple
import random

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import BaseAgent
from utils.llm_interface import LLMInterface
from utils.prompt_templates import SOCIAL_AGENT_SYSTEM_PROMPT, SOCIAL_AGENT_PROMPT_TEMPLATE
from experiments.config import SOCIAL_AGENT_CONFIG, EXPERIMENT_CONFIG
from utils.logger import get_default_logger


class SocialAgent(BaseAgent):
    """
    社会Agent，负责构建API服务市场协同网络，反映主体间交互关系
    """
    
    def __init__(
        self,
        name: str = "社会Agent",
        description: str = "负责构建API服务市场协同网络，反映主体间交互关系",
        memory_capacity: int = 100,
        logger: Optional[logging.Logger] = None,
        config: Dict[str, Any] = None,
        llm_interface: Optional[LLMInterface] = None
    ):
        """
        初始化社会Agent
        
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
        self.config = config or SOCIAL_AGENT_CONFIG
        
        # 初始化LLM接口
        self._llm = llm_interface or LLMInterface(logger=logger)
        
        # API关系数据
        self.api_data = {}
        
        # 社区划分结果
        self.communities = {}
        
        # 关键节点
        self.key_nodes = []
        
        # 跨社区连接
        self.cross_community_edges = []
        
        # 关系阈值
        self.relationship_threshold = self.config.get("relationship_threshold", 0.3)
        
        # 从配置中获取参数
        self.community_detection_algorithm = self.config.get('community_detection_algorithm', 'louvain')
        self.backbone_extraction_method = self.config.get('backbone_extraction_method', 'custom')
        self.max_communities = self.config.get('max_communities', 10)
        
        # 性能优化参数
        self.max_nodes = self.config.get('max_nodes', 5000)  # 限制最大节点数
        self.max_edges = self.config.get('max_edges', 50000)  # 限制最大边数
        self.betweenness_sample = self.config.get('betweenness_sample', 0.1)  # 介数中心性采样比例
        
        # 初始化网络图
        self.network = nx.Graph()
        
        # 初始化社区划分结果
        self.community_labels = {}
        
        # 初始化关系数据
        self.relationship_data = {
            "call_relationships": [],
            "similarity_relationships": []
        }
        
        self.log("社会Agent初始化完成")
        
    def load_api_data(self, api_data_path: str) -> bool:
        """
        从文件加载API数据
        
        Args:
            api_data_path: API数据文件路径
        
        Returns:
            bool: 加载是否成功
        """
        try:
            # 加载API数据
            with open(api_data_path, 'r', encoding='utf-8') as f:
                self.api_data = json.load(f)
            
            self.log(f"API数据加载完成，包含{len(self.api_data)}个API")
            return True
        except Exception as e:
            self.log(f"加载API数据失败: {str(e)}", level="error")
            import traceback
            self.log(traceback.format_exc(), level="error")
            return False
    
    def load_relationship_data(self, relationship_data_path: str) -> bool:
        """
        从文件加载关系数据
        
        Args:
            relationship_data_path: 关系数据文件路径
        
        Returns:
            bool: 加载是否成功
        """
        try:
            # 加载关系数据
            with open(relationship_data_path, 'r', encoding='utf-8') as f:
                self.relationship_data = json.load(f)
            
            self.log(f"关系数据加载完成，包含{len(self.relationship_data['call_relationships'])}个调用关系和{len(self.relationship_data['similarity_relationships'])}个相似性关系")
            return True
        except Exception as e:
            self.log(f"加载关系数据失败: {str(e)}", level="error")
            import traceback
            self.log(traceback.format_exc(), level="error")
            return False
    
    def percept(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        感知环境
        
        Args:
            environment: 环境信息
            
        Returns:
            感知结果
        """
        perception = {
            "api_data": environment.get("api_data", self.api_data),
            "api_count": environment.get("api_count", len(self.api_data)),
            "api_categories": environment.get("api_categories", EXPERIMENT_CONFIG["api_categories"]),
            "relationship_data": environment.get("relationship_data", {})
        }
        
        self.log(f"感知环境: {perception}", level="debug")
        return perception
    
    def think(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """
        思考，构建协同网络
        
        Args:
            perception: 感知结果
            
        Returns:
            思考结果，包含社区划分、关键节点和跨社区连接
        """
        self.log("开始思考，构建协同网络")
        
        # 提取感知信息
        api_data = perception["api_data"]
        api_count = perception["api_count"]
        api_categories = perception["api_categories"]
        relationship_data = perception["relationship_data"]
        
        # 构建关系数据描述
        relationship_data_text = self._build_relationship_data_text(relationship_data)
        
        # 构建提示词
        prompt = SOCIAL_AGENT_PROMPT_TEMPLATE.substitute(
            api_count=api_count,
            api_categories=", ".join(api_categories),
            relationship_data=relationship_data_text
        )
        
        # 使用LLM构建协同网络
        try:
            response = self._llm.completion(
                prompt=prompt,
                system_message=SOCIAL_AGENT_SYSTEM_PROMPT,
                temperature=0.5  # 使用较低的温度以增加一致性
            )
            
            # 解析响应
            network = self._parse_network_response(response)
            
            # 更新社区划分、关键节点和跨社区连接
            self.communities = network["communities"]
            self.key_nodes = network["key_nodes"]
            self.cross_community_edges = network["cross_community_edges"]
            
            # 更新关系阈值
            if network["relationship_threshold"] is not None:
                self.relationship_threshold = network["relationship_threshold"]
            
            # 更新思维链
            self.thinking_chain.append({
                "type": "network_construction",
                "prompt": prompt,
                "response": response,
                "parsed_network": network
            })
            
            thought = {
                "communities": self.communities,
                "key_nodes": self.key_nodes,
                "cross_community_edges": self.cross_community_edges,
                "relationship_threshold": self.relationship_threshold,
                "reasoning": "基于API关系数据构建了协同网络"
            }
            
            self.log(f"构建协同网络完成，识别了{len(self.communities)}个社区，{len(self.key_nodes)}个关键节点")
            return thought
            
        except Exception as e:
            self.log(f"构建协同网络失败: {str(e)}", level="error")
            return {
                "error": str(e),
                "reasoning": "构建协同网络失败"
            }
    
    def act(self, thought: Dict[str, Any]) -> Dict[str, Any]:
        """
        行动，提取网络骨干结构
        
        Args:
            thought: 思考结果
            
        Returns:
            行动结果，包含提取的网络骨干结构
        """
        self.log("开始行动，提取网络骨干结构")
        
        if "error" in thought:
            return {
                "status": "error",
                "error": thought["error"]
            }
        
        # 提取网络骨干结构
        backbone_network = self._extract_backbone_network(
            thought["communities"],
            thought["key_nodes"],
            thought["cross_community_edges"],
            thought["relationship_threshold"]
        )
        
        return {
            "status": "success",
            "backbone_network": backbone_network,
            "communities": thought["communities"],
            "key_nodes": thought["key_nodes"],
            "cross_community_edges": thought["cross_community_edges"],
            "relationship_threshold": thought["relationship_threshold"]
        }
    
    def construct_collaborative_network(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建协同网络的完整流程
        
        Args:
            environment: 环境信息
            
        Returns:
            构建的协同网络和提取的骨干结构
        """
        self.log("开始构建协同网络")
        
        # 感知环境
        perception = self.percept(environment)
        
        # 思考，构建协同网络
        thought = self.think(perception)
        
        # 行动，提取网络骨干结构
        action = self.act(thought)
        
        return action
    
    def _build_relationship_data_text(self, relationship_data: Dict[str, Any]) -> str:
        """
        构建关系数据描述
        
        Args:
            relationship_data: 关系数据
            
        Returns:
            关系数据描述
        """
        # 如果没有关系数据，返回模拟的关系描述
        if not relationship_data:
            return """
            API之间的调用关系:
            - 云存储API被数据分析API调用，强度为0.85
            - 数据分析API被可视化API调用，强度为0.72
            - 认证授权API被支付处理API调用，强度为0.90
            - 消息推送API被电商API调用，强度为0.65
            - 地图API被出行服务API调用，强度为0.78
            
            API的功能相似性:
            - 云存储API与对象存储API相似度为0.88
            - 支付处理API与订阅管理API相似度为0.75
            - 数据分析API与机器学习API相似度为0.82
            - 消息推送API与通知服务API相似度为0.91
            """
        
        # TODO: 根据实际关系数据构建描述
        relationship_texts = []
        
        # 处理API之间的调用关系
        if "call_relationships" in relationship_data:
            relationship_texts.append("API之间的调用关系:")
            for call in relationship_data["call_relationships"][:10]:  # 限制数量
                relationship_texts.append(f"- {call['source']}被{call['target']}调用，强度为{call['weight']}")
        
        # 处理API的功能相似性
        if "similarity_relationships" in relationship_data:
            relationship_texts.append("\nAPI的功能相似性:")
            for sim in relationship_data["similarity_relationships"][:10]:  # 限制数量
                relationship_texts.append(f"- {sim['api1']}与{sim['api2']}相似度为{sim['similarity']}")
        
        return "\n".join(relationship_texts)
    
    def _parse_network_response(self, response: str) -> Dict[str, Any]:
        """
        解析LLM生成的网络响应
        
        Args:
            response: LLM响应
            
        Returns:
            解析后的网络
        """
        # 简单解析，实际应用中可能需要更复杂的解析逻辑
        lines = response.strip().split("\n")
        
        network = {
            "communities": {},
            "key_nodes": [],
            "cross_community_edges": [],
            "relationship_threshold": None
        }
        
        # 提取社区划分、关键节点和跨社区连接
        current_section = None
        current_community = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 检测章节
            if "1. 社区划分" in line:
                current_section = "communities"
                continue
            elif "2. 关键节点" in line:
                current_section = "key_nodes"
                continue
            elif "3. 跨社区连接" in line:
                current_section = "cross_community_edges"
                continue
            elif "4. 建议的关系阈值" in line:
                current_section = "relationship_threshold"
                continue
                
            # 根据当前章节处理内容
            if current_section == "communities":
                # 检测社区
                if "社区" in line and "：" in line:
                    community_parts = line.split("：", 1)
                    if len(community_parts) == 2:
                        community_name = community_parts[0].strip()
                        community_desc = community_parts[1].strip()
                        current_community = community_name
                        network["communities"][current_community] = {
                            "description": community_desc,
                            "apis": []
                        }
                elif current_community and line.startswith("-"):
                    # 提取API
                    api = line[1:].strip()
                    if api:
                        network["communities"][current_community]["apis"].append(api)
            elif current_section == "key_nodes":
                if line.startswith("-") or line.startswith("*"):
                    # 提取关键节点
                    if "：" in line or ":" in line:
                        line = line.replace("：", ":")  # 统一使用英文冒号
                        parts = line[1:].strip().split(":", 1)
                        if len(parts) == 2:
                            node_info = parts[1].strip()
                            # 提取节点名称和社区
                            node_parts = parts[0].strip().split("（", 1)
                            if len(node_parts) == 2:
                                node_name = node_parts[0].strip()
                                community = node_parts[1].rstrip("）").strip()
                                
                                # 提取中心性
                                import re
                                centrality_match = re.search(r"连接度中心性为([\d\.]+)", node_info)
                                centrality = float(centrality_match.group(1)) if centrality_match else None
                                
                                network["key_nodes"].append({
                                    "name": node_name,
                                    "centrality": centrality,
                                    "community": community
                                })
            elif current_section == "cross_community_edges":
                if line.startswith("-") or line.startswith("*"):
                    # 提取跨社区连接
                    if "与" in line and "：" in line:
                        line = line.replace("：", ":")  # 统一使用英文冒号
                        edge_parts = line[1:].strip().split(":", 1)
                        if len(edge_parts) == 2:
                            nodes = edge_parts[0].strip()
                            edge_info = edge_parts[1].strip()
                            
                            # 提取节点
                            node_parts = nodes.split("与")
                            if len(node_parts) == 2:
                                source = node_parts[0].strip()
                                target = node_parts[1].strip()
                                
                                # 提取权重
                                import re
                                weight_match = re.search(r"权重为([\d\.]+)", edge_info)
                                weight = float(weight_match.group(1)) if weight_match else None
                                
                                network["cross_community_edges"].append({
                                    "source": source,
                                    "target": target,
                                    "weight": weight
                                })
            elif current_section == "relationship_threshold":
                # 提取关系阈值
                import re
                threshold_match = re.search(r"阈值([\d\.]+)", line)
                if threshold_match:
                    network["relationship_threshold"] = float(threshold_match.group(1))
        
        self.log(f"解析网络响应完成，识别了{len(network['communities'])}个社区，{len(network['key_nodes'])}个关键节点，{len(network['cross_community_edges'])}个跨社区连接")
        
        # 如果没有解析出关系阈值，使用默认值
        if network["relationship_threshold"] is None:
            network["relationship_threshold"] = self.config.get("relationship_threshold", 0.3)
            self.log(f"未解析出关系阈值，使用默认值: {network['relationship_threshold']}")
        
        return network
    
    def _extract_backbone_network(
        self,
        communities: Dict[str, Any],
        key_nodes: List[Dict[str, Any]],
        cross_community_edges: List[Dict[str, Any]],
        relationship_threshold: float
    ) -> Dict[str, Any]:
        """
        提取网络骨干结构
        
        Args:
            communities: 社区划分
            key_nodes: 关键节点
            cross_community_edges: 跨社区连接
            relationship_threshold: 关系阈值
            
        Returns:
            网络骨干结构
        """
        # 构建骨干网络
        backbone = {
            "nodes": [],
            "edges": [],
            "communities": []
        }
        
        # 添加关键节点
        for node in key_nodes:
            backbone["nodes"].append({
                "id": node["name"],
                "name": node["name"],
                "centrality": node["centrality"],
                "community": node["community"]
            })
        
        # 添加跨社区连接
        for edge in cross_community_edges:
            if edge["weight"] >= relationship_threshold:
                backbone["edges"].append({
                    "source": edge["source"],
                    "target": edge["target"],
                    "weight": edge["weight"]
                })
        
        # 添加社区信息
        for community_id, community_data in communities.items():
            backbone["communities"].append({
                "id": community_id,
                "description": community_data["description"],
                "size": len(community_data["apis"])
            })
        
        return backbone
    
    def get_communities(self) -> Dict[str, Any]:
        """
        获取社区划分结果
        
        Returns:
            社区划分结果
        """
        return self.communities
    
    def get_key_nodes(self) -> List[Dict[str, Any]]:
        """
        获取关键节点
        
        Returns:
            关键节点列表
        """
        return self.key_nodes
    
    def get_cross_community_edges(self) -> List[Dict[str, Any]]:
        """
        获取跨社区连接
        
        Returns:
            跨社区连接列表
        """
        return self.cross_community_edges
    
    def get_relationship_threshold(self) -> float:
        """
        获取关系阈值
        
        Returns:
            关系阈值
        """
        return self.relationship_threshold
    
    def build_network(self) -> bool:
        """
        构建协作网络
        
        Returns:
            bool: 构建是否成功
        """
        try:
            # 清空现有网络
            self.network.clear()
            
            # 如果API数量超过最大节点数，则随机采样
            api_ids = list(self.api_data.keys())
            if len(api_ids) > self.max_nodes:
                self.log(f"API数量({len(api_ids)})超过最大节点数({self.max_nodes})，进行随机采样")
                api_ids = random.sample(api_ids, self.max_nodes)
            
            # 添加节点
            for api_id in api_ids:
                api_info = self.api_data[api_id]
                self.network.add_node(api_id, **api_info)
            
            # 添加边 - 调用关系
            edge_count = 0
            for relation in self.relationship_data.get('call_relationships', []):
                source = relation['source']
                target = relation['target']
                
                # 确保source和target在网络中
                if source not in self.network or target not in self.network:
                    continue
                    
                weight = relation.get('weight', 1.0)
                
                # 只添加权重超过阈值的边
                if weight >= self.relationship_threshold:
                    self.network.add_edge(source, target, 
                                         weight=weight, 
                                         type='call')
                    edge_count += 1
                    
                    # 如果边数超过最大边数，则停止添加
                    if edge_count >= self.max_edges:
                        self.log(f"边数已达到最大限制({self.max_edges})，停止添加边")
                        break
            
            # 添加边 - 相似性关系
            # 如果相似性关系太多，则根据权重排序，只添加权重最高的一部分
            similarity_relations = self.relationship_data.get('similarity_relationships', [])
            
            # 计算剩余可添加的边数
            remaining_edges = self.max_edges - edge_count
            
            if len(similarity_relations) > remaining_edges:
                self.log(f"相似性关系数量({len(similarity_relations)})超过剩余边数限制({remaining_edges})，根据权重排序")
                # 根据权重排序
                similarity_relations.sort(key=lambda x: x.get('weight', 0), reverse=True)
                # 只取前remaining_edges个
                similarity_relations = similarity_relations[:remaining_edges]
            
            for relation in similarity_relations:
                source = relation['source']
                target = relation['target']
                
                # 确保source和target在网络中
                if source not in self.network or target not in self.network:
                    continue
                    
                weight = relation.get('weight', 1.0)
                
                # 只添加权重超过阈值的边
                if weight >= self.relationship_threshold:
                    self.network.add_edge(source, target, 
                                         weight=weight, 
                                         type='similarity')
                    edge_count += 1
            
            self.log(f"成功构建协作网络，包含{self.network.number_of_nodes()}个节点和{self.network.number_of_edges()}条边")
            return True
        except Exception as e:
            self.log(f"构建网络失败: {str(e)}", level="error")
            import traceback
            self.log(traceback.format_exc(), level="error")
            return False
    
    def detect_communities(self) -> Dict:
        """
        检测API社区
        
        Returns:
            Dict: 社区划分结果，键为社区ID，值为该社区包含的节点列表
        """
        try:
            if self.community_detection_algorithm == 'louvain':
                # 使用Louvain算法进行社区检测
                self.log("开始使用Louvain算法进行社区检测")
                partition = community_louvain.best_partition(self.network)
                
                # 将结果转换为字典格式，键为社区ID，值为节点列表
                communities = {}
                for node, community_id in partition.items():
                    if community_id not in communities:
                        communities[community_id] = []
                    communities[community_id].append(node)
                
                # 限制社区数量，合并小社区
                if len(communities) > self.max_communities:
                    communities = self._merge_small_communities(communities)
                
                self.communities = communities
                self.community_labels = partition
                
                self.log(f"成功检测到{len(communities)}个社区")
                return communities
            else:
                self.log(f"不支持的社区检测算法: {self.community_detection_algorithm}", level="error")
                return {}
        except Exception as e:
            self.log(f"社区检测失败: {str(e)}", level="error")
            import traceback
            self.log(traceback.format_exc(), level="error")
            return {}
    
    def _merge_small_communities(self, communities: Dict) -> Dict:
        """
        合并小社区，确保社区总数不超过max_communities
        
        Args:
            communities: 原始社区划分结果
        
        Returns:
            Dict: 合并后的社区划分结果
        """
        # 按社区大小排序
        sorted_communities = sorted(communities.items(), 
                                   key=lambda x: len(x[1]), 
                                   reverse=True)
        
        # 保留前max_communities-1个大社区
        result = {k: v for k, v in sorted_communities[:self.max_communities-1]}
        
        # 将剩余社区合并为一个"其他"社区
        other_community = []
        for k, v in sorted_communities[self.max_communities-1:]:
            other_community.extend(v)
        
        if other_community:
            result[max(result.keys()) + 1] = other_community
        
        return result
    
    def identify_key_nodes(self, top_n: int = 10) -> List[Dict]:
        """
        识别网络中的关键节点
        
        Args:
            top_n: 返回的关键节点数量
        
        Returns:
            List[Dict]: 关键节点列表，每个节点包含ID、中心性指标等信息
        """
        try:
            self.log("开始识别关键节点")
            
            # 计算度中心性
            self.log("计算度中心性")
            degree_centrality = nx.degree_centrality(self.network)
            
            # 计算介数中心性（使用近似算法）
            self.log("计算介数中心性（使用近似算法）")
            # 如果网络规模大，则使用采样
            if self.network.number_of_nodes() > 1000:
                k = int(self.network.number_of_nodes() * self.betweenness_sample)
                betweenness_centrality = nx.betweenness_centrality(self.network, k=k, normalized=True)
            else:
                betweenness_centrality = nx.betweenness_centrality(self.network)
            
            # 计算接近中心性（使用近似算法）
            self.log("计算接近中心性")
            # 如果网络规模大，则使用采样
            if self.network.number_of_nodes() > 1000:
                closeness_centrality = {}
                # 随机选择一部分节点计算接近中心性
                sample_nodes = random.sample(list(self.network.nodes()), 
                                            int(self.network.number_of_nodes() * self.betweenness_sample))
                for node in self.network.nodes():
                    if node in sample_nodes:
                        try:
                            closeness_centrality[node] = nx.closeness_centrality(self.network, u=node)
                        except:
                            closeness_centrality[node] = 0.0
                    else:
                        closeness_centrality[node] = 0.0
            else:
                closeness_centrality = nx.closeness_centrality(self.network)
            
            # 计算特征向量中心性
            self.log("计算特征向量中心性")
            try:
                eigenvector_centrality = nx.eigenvector_centrality(self.network, max_iter=100, tol=1.0e-3)
            except:
                self.log("特征向量中心性计算失败，使用PageRank作为替代", level="warning")
                eigenvector_centrality = nx.pagerank(self.network)
            
            # 综合评分
            self.log("计算综合评分")
            key_nodes = []
            for node in self.network.nodes():
                score = (degree_centrality[node] + 
                         betweenness_centrality[node] + 
                         closeness_centrality[node] + 
                         eigenvector_centrality[node]) / 4.0
                
                community_id = self.community_labels.get(node, -1)
                
                key_nodes.append({
                    'id': node,
                    'name': self.network.nodes[node].get('Name', '未知'),
                    'category': self.network.nodes[node].get('Category', '未知'),
                    'degree_centrality': degree_centrality[node],
                    'betweenness_centrality': betweenness_centrality[node],
                    'closeness_centrality': closeness_centrality[node],
                    'eigenvector_centrality': eigenvector_centrality[node],
                    'score': score,
                    'community_id': community_id
                })
            
            # 按综合评分排序
            key_nodes.sort(key=lambda x: x['score'], reverse=True)
            
            # 取前top_n个节点
            self.key_nodes = key_nodes[:top_n]
            
            self.log(f"成功识别{len(self.key_nodes)}个关键节点")
            return self.key_nodes
        except Exception as e:
            self.log(f"关键节点识别失败: {str(e)}", level="error")
            import traceback
            self.log(traceback.format_exc(), level="error")
            return []
    
    def analyze_cross_community_links(self) -> List[Dict]:
        """
        分析跨社区连接
        
        Returns:
            List[Dict]: 跨社区连接列表
        """
        try:
            self.log("开始分析跨社区连接")
            cross_links = []
            
            for u, v, data in self.network.edges(data=True):
                community_u = self.community_labels.get(u, -1)
                community_v = self.community_labels.get(v, -1)
                
                # 如果两个节点属于不同社区，则认为是跨社区连接
                if community_u != community_v:
                    cross_links.append({
                        'source': u,
                        'source_name': self.network.nodes[u].get('Name', '未知'),
                        'source_community': community_u,
                        'target': v,
                        'target_name': self.network.nodes[v].get('Name', '未知'),
                        'target_community': community_v,
                        'weight': data.get('weight', 1.0),
                        'type': data.get('type', 'unknown')
                    })
            
            # 按权重排序
            cross_links.sort(key=lambda x: x['weight'], reverse=True)
            
            self.cross_community_edges = cross_links
            
            self.log(f"成功分析{len(cross_links)}个跨社区连接")
            # 即使没有跨社区连接，也返回空列表，不视为失败
            return cross_links
        except Exception as e:
            self.log(f"跨社区连接分析失败: {str(e)}", level="error")
            import traceback
            self.log(traceback.format_exc(), level="error")
            return []
    
    def extract_backbone_network(self) -> nx.Graph:
        """
        提取骨干网络
        
        Returns:
            nx.Graph: 骨干网络
        """
        try:
            self.log("开始提取骨干网络")
            if self.backbone_extraction_method == 'custom':
                # 自定义方法提取骨干网络
                backbone = nx.Graph()
                
                # 添加所有节点
                for node in self.network.nodes(data=True):
                    backbone.add_node(node[0], **node[1])
                
                # 添加关键节点之间的边
                key_node_ids = [node['id'] for node in self.key_nodes]
                for u, v, data in self.network.edges(data=True):
                    if u in key_node_ids or v in key_node_ids:
                        backbone.add_edge(u, v, **data)
                
                # 添加跨社区连接
                max_cross_edges = min(len(self.cross_community_edges), 1000)  # 最多添加1000条跨社区边
                for link in self.cross_community_edges[:int(max_cross_edges * 0.2)]:  # 只添加权重最高的20%
                    backbone.add_edge(link['source'], link['target'], 
                                     weight=link['weight'], 
                                     type=link['type'])
                
                self.log(f"成功提取骨干网络，包含{backbone.number_of_nodes()}个节点和{backbone.number_of_edges()}条边")
                return backbone
            else:
                self.log(f"不支持的骨干网络提取方法: {self.backbone_extraction_method}", level="error")
                return nx.Graph()
        except Exception as e:
            self.log(f"骨干网络提取失败: {str(e)}", level="error")
            import traceback
            self.log(traceback.format_exc(), level="error")
            return nx.Graph()
    
    def save_results(self, output_dir: str) -> bool:
        """
        保存分析结果
        
        Args:
            output_dir: 输出目录
        
        Returns:
            bool: 保存是否成功
        """
        try:
            self.log(f"开始保存分析结果到目录: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存社区划分结果
            with open(os.path.join(output_dir, 'communities.json'), 'w', encoding='utf-8') as f:
                json.dump(self.communities, f, ensure_ascii=False, indent=2)
            
            # 保存关键节点
            with open(os.path.join(output_dir, 'key_nodes.json'), 'w', encoding='utf-8') as f:
                json.dump(self.key_nodes, f, ensure_ascii=False, indent=2)
            
            # 保存跨社区连接
            with open(os.path.join(output_dir, 'cross_community_edges.json'), 'w', encoding='utf-8') as f:
                json.dump(self.cross_community_edges, f, ensure_ascii=False, indent=2)
            
            # 保存网络统计信息
            stats = {
                'node_count': self.network.number_of_nodes(),
                'edge_count': self.network.number_of_edges(),
                'community_count': len(self.communities),
                'key_node_count': len(self.key_nodes),
                'cross_community_link_count': len(self.cross_community_edges),
                'average_degree': sum(dict(self.network.degree()).values()) / self.network.number_of_nodes()
            }
            
            with open(os.path.join(output_dir, 'network_stats.json'), 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            self.log(f"成功保存分析结果到目录: {output_dir}")
            return True
        except Exception as e:
            self.log(f"保存分析结果失败: {str(e)}", level="error")
            import traceback
            self.log(traceback.format_exc(), level="error")
            return False
    
    def visualize_network(self, output_path: str, show_communities: bool = True) -> bool:
        """
        可视化网络
        
        Args:
            output_path: 输出文件路径
            show_communities: 是否显示社区
        
        Returns:
            bool: 可视化是否成功
        """
        try:
            self.log("开始可视化网络")
            
            # 如果网络太大，则只可视化一个子图
            if self.network.number_of_nodes() > 500:
                self.log(f"网络节点数({self.network.number_of_nodes()})过多，只可视化关键节点和它们的邻居")
                # 创建一个子图，包含关键节点和它们的邻居
                subgraph_nodes = set()
                for node in self.key_nodes:
                    subgraph_nodes.add(node['id'])
                    # 添加邻居节点
                    for neighbor in self.network.neighbors(node['id']):
                        subgraph_nodes.add(neighbor)
                
                # 如果子图仍然太大，则限制节点数量
                if len(subgraph_nodes) > 500:
                    self.log(f"子图节点数({len(subgraph_nodes)})仍然过多，随机采样500个节点")
                    subgraph_nodes = set(random.sample(list(subgraph_nodes), 500))
                
                # 创建子图
                subgraph = self.network.subgraph(subgraph_nodes)
            else:
                subgraph = self.network
            
            plt.figure(figsize=(16, 12))
            
            # 设置节点位置
            pos = nx.spring_layout(subgraph, seed=42)
            
            # 设置节点颜色
            if show_communities and self.community_labels:
                colors = []
                for node in subgraph.nodes():
                    community_id = self.community_labels.get(node, 0)
                    colors.append(community_id)
            else:
                colors = 'skyblue'
            
            # 设置节点大小
            sizes = []
            for node in subgraph.nodes():
                # 如果是关键节点，则设置更大的尺寸
                is_key = any(kn['id'] == node for kn in self.key_nodes)
                sizes.append(300 if is_key else 100)
            
            # 绘制节点
            nx.draw_networkx_nodes(subgraph, pos, 
                                  node_color=colors, 
                                  node_size=sizes, 
                                  cmap=plt.cm.rainbow)
            
            # 绘制边
            nx.draw_networkx_edges(subgraph, pos, alpha=0.3)
            
            # 绘制标签
            nx.draw_networkx_labels(subgraph, pos, font_size=8)
            
            plt.title("API服务协作网络")
            plt.axis('off')
            
            # 保存图像
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.log(f"成功可视化网络并保存到: {output_path}")
            return True
        except Exception as e:
            self.log(f"网络可视化失败: {str(e)}", level="error")
            import traceback
            self.log(traceback.format_exc(), level="error")
            return False
    
    def run(self, api_data_path: str, relationship_data_path: str, output_dir: str) -> Dict:
        """
        运行社会Agent的完整流程
        
        Args:
            api_data_path: API数据文件路径
            relationship_data_path: 关系数据文件路径
            output_dir: 输出目录
        
        Returns:
            Dict: 运行结果
        """
        # 加载数据
        if not self.load_api_data(api_data_path):
            return {'success': False, 'error': 'API数据加载失败'}
        
        if not self.load_relationship_data(relationship_data_path):
            return {'success': False, 'error': '关系数据加载失败'}
        
        # 构建网络
        if not self.build_network():
            return {'success': False, 'error': '网络构建失败'}
        
        # 检测社区
        communities = self.detect_communities()
        if not communities:
            return {'success': False, 'error': '社区检测失败'}
        
        # 识别关键节点
        key_nodes = self.identify_key_nodes()
        if not key_nodes:
            return {'success': False, 'error': '关键节点识别失败'}
        
        # 分析跨社区连接
        cross_links = self.analyze_cross_community_links()
        # 即使没有跨社区连接也继续执行，不视为失败
        
        # 提取骨干网络
        backbone = self.extract_backbone_network()
        
        # 保存结果
        if not self.save_results(output_dir):
            return {'success': False, 'error': '结果保存失败'}
        
        # 可视化网络
        vis_path = os.path.join(output_dir, 'network_visualization.png')
        self.visualize_network(vis_path)
        
        return {
            'success': True,
            'communities': len(communities),
            'key_nodes': len(key_nodes),
            'cross_community_links': len(cross_links),
            'output_dir': output_dir
        } 