"""
API服务生态系统情景生成 - 思维链可视化工具
可视化智能体的思考过程，生成思维链图表
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches


class ThinkingChainVisualizer:
    """思维链可视化工具类"""
    
    def __init__(self, logger=None):
        """
        初始化思维链可视化器
        
        Args:
            logger: 日志记录器，如果为None则创建新的记录器
        """
        if logger is None:
            import logging
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
    
    def visualize_thinking_chain(
        self,
        thinking_chain: List[Dict[str, Any]],
        output_file: str = "thinking_chain.png",
        title: str = "智能体思维链可视化",
        figsize: Tuple[int, int] = (16, 10),
        dpi: int = 300,
        node_size: int = 2000,
        font_size: int = 10
    ) -> str:
        """
        可视化思维链
        
        Args:
            thinking_chain: 思维链数据
            output_file: 输出文件路径
            title: 图表标题
            figsize: 图表大小
            dpi: 图表分辨率
            node_size: 节点大小
            font_size: 字体大小
            
        Returns:
            输出文件路径
        """
        self.logger.info(f"开始可视化思维链: {title}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # 检查思维链数据有效性
        if not thinking_chain:
            self.logger.error("思维链数据为空")
            return None
        
        # 创建有向图
        G = nx.DiGraph()
        
        # 节点类型到颜色的映射
        node_colors = {
            "perception": "#3498db",  # 蓝色
            "thought": "#2ecc71",     # 绿色
            "action": "#e74c3c",      # 红色
            "reflection": "#9b59b6",  # 紫色
            "network_construction": "#f39c12",  # 橙色
            "scenario_generation": "#1abc9c",   # 青色
            "constraint_generation": "#d35400",  # 棕色
            "scenario_validation": "#27ae60",   # 深绿色
            "default": "#7f8c8d"      # 灰色
        }
        
        # 添加节点
        for i, step in enumerate(thinking_chain):
            node_id = f"step_{i}"
            step_type = step.get("type", "default")
            step_content = self._get_step_summary(step)
            
            G.add_node(
                node_id,
                label=f"{step_type}\n{step_content}",
                color=node_colors.get(step_type, node_colors["default"]),
                type=step_type,
                step=i
            )
            
            # 如果不是第一个节点，则添加边连接前一个节点
            if i > 0:
                G.add_edge(f"step_{i-1}", node_id)
        
        # 创建图表
        plt.figure(figsize=figsize)
        
        # 设置布局
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # 获取节点颜色
        node_colors_list = [data["color"] for _, data in G.nodes(data=True)]
        
        # 绘制节点
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_size,
            node_color=node_colors_list,
            alpha=0.8
        )
        
        # 绘制边
        nx.draw_networkx_edges(
            G, pos,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=20,
            edge_color="#2c3e50",
            width=2.0,
            alpha=0.7
        )
        
        # 绘制标签
        labels = {node: data["label"] for node, data in G.nodes(data=True)}
        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=font_size,
            font_family="sans-serif",
            font_weight="bold",
            font_color="black"
        )
        
        # 添加图例
        legend_elements = []
        for node_type, color in node_colors.items():
            if node_type != "default" and any(data["type"] == node_type for _, data in G.nodes(data=True)):
                legend_elements.append(
                    mpatches.Patch(color=color, label=node_type.replace("_", " ").title())
                )
        
        plt.legend(handles=legend_elements, loc="upper right", fontsize=12)
        
        # 设置标题
        plt.title(title, fontsize=16, pad=20)
        
        # 保存图表
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
        plt.close()
        
        self.logger.info(f"思维链可视化已保存至: {output_file}")
        return output_file
    
    def visualize_agent_thinking_chain(
        self,
        agent_state_file: str,
        output_file: str = None,
        title: str = None
    ) -> str:
        """
        从智能体状态文件可视化思维链
        
        Args:
            agent_state_file: 智能体状态文件路径
            output_file: 输出文件路径，如果为None则自动生成
            title: 图表标题，如果为None则自动生成
            
        Returns:
            输出文件路径
        """
        try:
            # 加载智能体状态
            with open(agent_state_file, "r", encoding="utf-8") as f:
                agent_state = json.load(f)
            
            # 提取思维链
            thinking_chain = agent_state.get("thinking_chain", [])
            
            if not thinking_chain:
                self.logger.error(f"智能体状态文件中没有思维链数据: {agent_state_file}")
                return None
            
            # 设置默认输出文件路径
            if output_file is None:
                output_dir = os.path.dirname(agent_state_file)
                agent_name = agent_state.get("name", "unknown")
                output_file = os.path.join(output_dir, f"{agent_name}_thinking_chain.png")
            
            # 设置默认标题
            if title is None:
                agent_name = agent_state.get("name", "Unknown Agent")
                title = f"{agent_name}思维链可视化"
            
            # 可视化思维链
            return self.visualize_thinking_chain(
                thinking_chain=thinking_chain,
                output_file=output_file,
                title=title
            )
            
        except Exception as e:
            self.logger.error(f"可视化智能体思维链失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def visualize_multi_agent_thinking_chains(
        self,
        agent_state_files: List[str],
        output_dir: str,
        combined_output_file: str = "combined_thinking_chain.png",
        figsize: Tuple[int, int] = (20, 12),
        dpi: int = 300
    ) -> List[str]:
        """
        可视化多个智能体的思维链
        
        Args:
            agent_state_files: 智能体状态文件路径列表
            output_dir: 输出目录
            combined_output_file: 组合思维链输出文件名
            figsize: 图表大小
            dpi: 图表分辨率
            
        Returns:
            输出文件路径列表
        """
        self.logger.info(f"开始可视化{len(agent_state_files)}个智能体的思维链")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        output_files = []
        agent_states = []
        
        # 为每个智能体可视化思维链
        for agent_state_file in agent_state_files:
            try:
                # 加载智能体状态
                with open(agent_state_file, "r", encoding="utf-8") as f:
                    agent_state = json.load(f)
                
                agent_states.append(agent_state)
                
                # 提取思维链
                thinking_chain = agent_state.get("thinking_chain", [])
                
                if not thinking_chain:
                    self.logger.warning(f"智能体状态文件中没有思维链数据: {agent_state_file}")
                    continue
                
                # 设置输出文件路径
                agent_name = agent_state.get("name", "unknown")
                output_file = os.path.join(output_dir, f"{agent_name}_thinking_chain.png")
                
                # 设置标题
                title = f"{agent_name}思维链可视化"
                
                # 可视化思维链
                result = self.visualize_thinking_chain(
                    thinking_chain=thinking_chain,
                    output_file=output_file,
                    title=title
                )
                
                if result:
                    output_files.append(result)
                
            except Exception as e:
                self.logger.error(f"可视化智能体思维链失败: {agent_state_file}, {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
        
        # 创建组合思维链可视化
        if agent_states:
            try:
                # 创建图表
                fig, axes = plt.subplots(len(agent_states), 1, figsize=figsize)
                if len(agent_states) == 1:
                    axes = [axes]
                
                for i, agent_state in enumerate(agent_states):
                    agent_name = agent_state.get("name", "Unknown Agent")
                    thinking_chain = agent_state.get("thinking_chain", [])
                    
                    # 创建简化的思维链图
                    G = nx.DiGraph()
                    
                    # 节点类型到颜色的映射
                    node_colors = {
                        "perception": "#3498db",  # 蓝色
                        "thought": "#2ecc71",     # 绿色
                        "action": "#e74c3c",      # 红色
                        "reflection": "#9b59b6",  # 紫色
                        "network_construction": "#f39c12",  # 橙色
                        "scenario_generation": "#1abc9c",   # 青色
                        "constraint_generation": "#d35400",  # 棕色
                        "scenario_validation": "#27ae60",   # 深绿色
                        "default": "#7f8c8d"      # 灰色
                    }
                    
                    # 添加节点
                    for j, step in enumerate(thinking_chain):
                        node_id = f"agent{i}_step_{j}"
                        step_type = step.get("type", "default")
                        
                        G.add_node(
                            node_id,
                            label=step_type,
                            color=node_colors.get(step_type, node_colors["default"]),
                            type=step_type,
                            step=j
                        )
                        
                        # 如果不是第一个节点，则添加边连接前一个节点
                        if j > 0:
                            G.add_edge(f"agent{i}_step_{j-1}", node_id)
                    
                    # 设置布局
                    pos = nx.spring_layout(G, k=0.3, iterations=50)
                    
                    # 获取节点颜色
                    node_colors_list = [data["color"] for _, data in G.nodes(data=True)]
                    
                    # 在对应的子图上绘制
                    ax = axes[i]
                    ax.set_title(f"{agent_name}思维链", fontsize=14)
                    ax.axis("off")
                    
                    # 绘制节点
                    nx.draw_networkx_nodes(
                        G, pos,
                        ax=ax,
                        node_size=1500,
                        node_color=node_colors_list,
                        alpha=0.8
                    )
                    
                    # 绘制边
                    nx.draw_networkx_edges(
                        G, pos,
                        ax=ax,
                        arrows=True,
                        arrowstyle="-|>",
                        arrowsize=15,
                        edge_color="#2c3e50",
                        width=1.5,
                        alpha=0.7
                    )
                    
                    # 绘制标签
                    labels = {node: data["label"] for node, data in G.nodes(data=True)}
                    nx.draw_networkx_labels(
                        G, pos,
                        ax=ax,
                        labels=labels,
                        font_size=9,
                        font_family="sans-serif",
                        font_weight="bold",
                        font_color="black"
                    )
                
                # 添加图例
                legend_elements = []
                for node_type, color in node_colors.items():
                    if node_type != "default":
                        legend_elements.append(
                            mpatches.Patch(color=color, label=node_type.replace("_", " ").title())
                        )
                
                fig.legend(handles=legend_elements, loc="upper right", fontsize=12)
                
                # 设置总标题
                fig.suptitle("多智能体思维链可视化", fontsize=16)
                
                # 保存图表
                combined_output_path = os.path.join(output_dir, combined_output_file)
                plt.tight_layout()
                plt.savefig(combined_output_path, dpi=dpi, bbox_inches="tight")
                plt.close()
                
                output_files.append(combined_output_path)
                self.logger.info(f"组合思维链可视化已保存至: {combined_output_path}")
                
            except Exception as e:
                self.logger.error(f"创建组合思维链可视化失败: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
        
        return output_files
    
    def _get_step_summary(self, step: Dict[str, Any]) -> str:
        """
        获取思维步骤的摘要
        
        Args:
            step: 思维步骤数据
            
        Returns:
            摘要文本
        """
        step_type = step.get("type", "")
        
        if step_type == "perception":
            content = step.get("content", {})
            if isinstance(content, dict):
                keys = list(content.keys())
                return f"感知: {', '.join(keys[:3])}..." if keys else "感知环境"
            return "感知环境"
            
        elif step_type == "thought":
            content = step.get("content", {})
            if isinstance(content, dict) and "reasoning" in content:
                reasoning = content["reasoning"]
                return reasoning[:50] + "..." if len(reasoning) > 50 else reasoning
            return "思考"
            
        elif step_type == "action":
            content = step.get("content", {})
            if isinstance(content, dict) and "status" in content:
                return f"行动: {content['status']}"
            return "行动"
            
        elif step_type == "reflection":
            return "反思总结"
            
        elif step_type == "network_construction":
            parsed_network = step.get("parsed_network", {})
            communities = len(parsed_network.get("communities", {}))
            key_nodes = len(parsed_network.get("key_nodes", []))
            return f"构建网络: {communities}个社区, {key_nodes}个关键节点"
            
        elif step_type == "scenario_generation":
            scenario = step.get("scenario", {})
            return f"生成情景: {scenario.get('title', '未命名')}"
            
        elif step_type == "constraint_generation":
            constraints = step.get("constraints", [])
            return f"生成约束: {len(constraints)}个约束条件"
            
        elif step_type == "scenario_validation":
            parsed_result = step.get("parsed_result", {})
            assessment = parsed_result.get("overall_assessment", "")
            score = parsed_result.get("overall_score", "")
            return f"验证情景: {assessment}, 评分{score}"
            
        else:
            return step_type or "未知步骤"


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='思维链可视化工具')
    parser.add_argument('--agent_state', type=str, required=True,
                        help='智能体状态文件路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出文件路径')
    parser.add_argument('--title', type=str, default=None,
                        help='图表标题')
    
    args = parser.parse_args()
    
    visualizer = ThinkingChainVisualizer()
    visualizer.visualize_agent_thinking_chain(
        agent_state_file=args.agent_state,
        output_file=args.output,
        title=args.title
    )


if __name__ == "__main__":
    main() 