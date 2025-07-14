"""
API服务生态系统情景生成 - 主入口
"""

import os
import sys
import argparse
from pathlib import Path
import json
import numpy as np
from typing import Dict, Tuple
import logging
import datetime

# 添加项目根目录到Python路径
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(ROOT_DIR))

from utils.logger import get_default_logger
from utils.llm_interface import LLMInterface
from agents.environment_agent import EnvironmentAgent
from agents.social_agent import SocialAgent
from agents.planner_agent import PlannerAgent
from experiments.config import (
    ENVIRONMENT_AGENT_CONFIG,
    SOCIAL_AGENT_CONFIG,
    PLANNER_AGENT_CONFIG,
    EXPERIMENT_CONFIG
)
from utils.visualization import (
    load_historical_data,
    plot_environment_boundaries,
    plot_environment_boundaries_from_files,
    plot_multi_charts,
    plot_scenario_comparison
)
from utils.radar_visualizer import RadarVisualizer
from utils.network_visualizer import NetworkVisualizer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='API服务生态系统情景生成')
    parser.add_argument('--mode', type=str, default='run',
                        choices=['run', 'test', 'test_env_agent', 'test_social_agent',
                                'test_planner_agent', 'visualize', 'network_vis', 'radar_vis',
                                'visualization', 'all_vis'],
                        help='运行模式')
    parser.add_argument('--config', type=str, default='../llm_config.json',
                        help='配置文件路径')
    parser.add_argument('--llm_config', type=str, default='../llm_config.json',
                        help='LLM配置文件路径')
    parser.add_argument('--mock', action='store_true',
                        help='是否使用模拟模式')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--scenarios_file', type=str, default=None,
                        help='情景数据文件路径，用于可视化模式')
    parser.add_argument('--scenario_file', type=str, default=None,
                        help='单个情景文件路径，用于可视化模式')
    parser.add_argument('--historical_data_file', type=str, default=None,
                        help='历史数据文件路径，用于可视化模式')
    parser.add_argument('--network_file', type=str, default=None,
                        help='网络数据文件路径，用于可视化模式')
    parser.add_argument('--radar_file', type=str, default=None,
                        help='雷达图数据文件路径，用于雷达图可视化模式')
    parser.add_argument('--output_file', type=str, default=None,
                        help='输出文件路径')

    # 可视化模式参数
    parser.add_argument('--vis_title', type=str, default='API服务生态系统环境边界',
                        help='可视化图表标题')
    parser.add_argument('--fig_dpi', type=int, default=300,
                        help='图表DPI')

    # 雷达图可视化模式参数
    parser.add_argument('--radar_title', type=str, default='多维情景向量对比',
                        help='雷达图标题')
    parser.add_argument('--highlight_methods', type=str, default='Original,Ours',
                        help='高亮显示的方法，用逗号分隔')
    parser.add_argument('--fig_width', type=int, default=16,
                        help='图表宽度')
    parser.add_argument('--fig_height', type=int, default=12,
                        help='图表高度')

    # 网络可视化模式参数
    parser.add_argument('--network_years', type=str, default='2006,2010,2015,2020',
                        help='网络可视化年份列表，用逗号分隔')
    parser.add_argument('--network_methods', type=str, default='Original,Cluster,GT,HSS,PLA,Ours',
                        help='网络可视化方法列表，用逗号分隔')
    parser.add_argument('--network_title', type=str, default='API关系骨架计算实验设计框架',
                        help='网络可视化图表标题')

    return parser.parse_args()


def test_mode(args):
    """测试模式，用于测试各个组件是否正常工作"""
    logger.info("启动测试模式")

    # 测试LLM接口
    try:
        # 初始化LLM接口，根据参数决定是否使用模拟模式
        if args.llm_config:
            logger.info(f"使用配置文件: {args.llm_config}")
            llm = LLMInterface.from_config_file(args.llm_config, logger=logger, mock_mode=args.mock)
        else:
            llm = LLMInterface(mock_mode=args.mock, logger=logger)

        logger.info(f"LLM接口初始化成功: {llm}")

        # 测试不同类型的模拟响应
        logger.info("测试环境Agent模拟响应:")
        env_response = llm.completion(
            prompt="请生成一个关于API服务生态系统的极端情景",
            system_message="你是环境Agent"
        )
        logger.info(f"环境Agent模拟响应: {env_response[:100]}...")

        logger.info("测试社会Agent模拟响应:")
        social_response = llm.completion(
            prompt="请分析API服务之间的协同网络",
            system_message="你是社会Agent"
        )
        logger.info(f"社会Agent模拟响应: {social_response[:100]}...")

        logger.info("测试规划Agent模拟响应:")
        planning_response = llm.completion(
            prompt="请生成API服务生态系统的实验规则",
            system_message="你是规划Agent"
        )
        logger.info(f"规划Agent模拟响应: {planning_response[:100]}...")

        logger.info("测试验证Agent模拟响应:")
        validation_response = llm.completion(
            prompt="请评估以下情景的合理性",
            system_message="你是验证Agent"
        )
        logger.info(f"验证Agent模拟响应: {validation_response[:100]}...")

    except Exception as e:
        logger.error(f"LLM接口测试失败: {str(e)}")

    # 测试配置加载
    logger.info(f"实验配置: {EXPERIMENT_CONFIG}")

    # 测试环境Agent
    try:
        logger.info("测试环境Agent:")
        env_agent = EnvironmentAgent(llm_interface=llm, logger=logger)

        # 测试生成极端情景
        environment = {
            "historical_data": {},  # 使用默认的模拟数据
            "domain_knowledge": [],  # 使用默认的模拟知识
            "time_range": "2006-2022",
            "api_categories": EXPERIMENT_CONFIG["api_categories"]
        }

        result = env_agent.generate_extreme_scenario(environment)

        if result["status"] == "success":
            scenario = result["scenario"]
            logger.info(f"环境Agent生成极端情景成功: {scenario['title']}")
        else:
            logger.error(f"环境Agent生成极端情景失败: {result.get('error', '未知错误')}")

    except Exception as e:
        logger.error(f"环境Agent测试失败: {str(e)}")

    logger.info("测试模式完成")


def test_env_agent(args):
    """测试环境Agent"""
    logger.info("启动环境Agent测试模式")

    # 创建环境Agent，使用模拟模式
    try:
        # 创建LLM接口，根据参数决定是否使用模拟模式
        if args.llm_config:
            logger.info(f"使用配置文件: {args.llm_config}")
            llm = LLMInterface.from_config_file(args.llm_config, logger=logger, mock_mode=args.mock)
        else:
            llm = LLMInterface(mock_mode=args.mock, logger=logger)

        logger.info(f"LLM接口初始化成功: {llm}")

        # 使用LLM接口创建环境Agent
        env_agent = EnvironmentAgent(llm_interface=llm, logger=logger)
        logger.info(f"环境Agent创建成功: {env_agent}")

        # 测试生成极端情景
        environment = {
            "historical_data": {},  # 使用默认的模拟数据
            "domain_knowledge": [],  # 使用默认的模拟知识
            "time_range": "2006-2022",
            "api_categories": EXPERIMENT_CONFIG["api_categories"]
        }

        # 生成极端情景
        result = env_agent.generate_extreme_scenario(environment)

        if result["status"] == "success":
            scenario = result["scenario"]
            logger.info(f"生成极端情景成功:")
            logger.info(f"标题: {scenario['title']}")
            logger.info(f"描述: {scenario['description'][:100]}...")
            logger.info(f"需求变化: {scenario['demand_changes']}")
            logger.info(f"概率: {scenario['probability']}")
            logger.info(f"持续时间: {scenario['duration']}个月")
            logger.info(f"长期影响: {scenario['long_term_impact'][:100]}...")

            # 获取边界
            upper, lower = env_agent.get_boundaries()
            logger.info(f"上限边界: {upper}")
            logger.info(f"下限边界: {lower}")
        else:
            logger.error(f"生成极端情景失败: {result.get('error', '未知错误')}")

    except Exception as e:
        logger.error(f"环境Agent测试失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

    logger.info("环境Agent测试模式完成")


def test_social_agent(args):
    """测试社会Agent"""
    logger.info("启动社会Agent测试模式")

    # 创建社会Agent，使用模拟模式
    try:
        # 创建LLM接口，根据参数决定是否使用模拟模式
        if args.llm_config:
            logger.info(f"使用配置文件: {args.llm_config}")
            llm = LLMInterface.from_config_file(args.llm_config, logger=logger, mock=args.mock)
        else:
            llm = LLMInterface(mock_mode=args.mock, logger=logger)

        logger.info(f"LLM接口初始化成功: {llm}")

        # 使用LLM接口创建社会Agent
        social_agent = SocialAgent(llm_interface=llm, logger=logger)
        logger.info(f"社会Agent创建成功: {social_agent}")

        # 测试构建协同网络
        environment = {
            "api_data": {},  # 使用默认的模拟数据
            "api_count": 100,  # 假设有100个API
            "api_categories": EXPERIMENT_CONFIG["api_categories"],
            "relationship_data": {}  # 使用默认的模拟关系数据
        }

        # 构建协同网络
        result = social_agent.construct_collaborative_network(environment)

        if result["status"] == "success":
            communities = result["communities"]
            key_nodes = result["key_nodes"]
            cross_community_edges = result["cross_community_edges"]
            relationship_threshold = result["relationship_threshold"]
            backbone_network = result["backbone_network"]

            logger.info(f"构建协同网络成功:")
            logger.info(f"社区数量: {len(communities)}")
            logger.info(f"关键节点数量: {len(key_nodes)}")
            logger.info(f"跨社区连接数量: {len(cross_community_edges)}")
            logger.info(f"关系阈值: {relationship_threshold}")
            logger.info(f"骨干网络节点数: {len(backbone_network['nodes'])}")
            logger.info(f"骨干网络边数: {len(backbone_network['edges'])}")

            # 输出部分社区信息
            if communities:
                logger.info("社区信息:")
                for community_id, community_data in list(communities.items())[:2]:  # 只展示前两个社区
                    logger.info(f"  - {community_id}: {community_data['description']}")
                    logger.info(f"    包含API数量: {len(community_data['apis'])}")

            # 输出部分关键节点信息
            if key_nodes:
                logger.info("关键节点信息:")
                for node in key_nodes[:3]:  # 只展示前三个关键节点
                    logger.info(f"  - {node['name']}")
                    logger.info(f"    中心性: {node['centrality']}")
                    logger.info(f"    所属社区: {node['community']}")

            # 输出部分跨社区连接信息
            if cross_community_edges:
                logger.info("跨社区连接信息:")
                for edge in cross_community_edges[:3]:  # 只展示前三个跨社区连接
                    logger.info(f"  - {edge['source']} 与 {edge['target']}")
                    logger.info(f"    权重: {edge['weight']}")
        else:
            logger.error(f"构建协同网络失败: {result.get('error', '未知错误')}")

    except Exception as e:
        logger.error(f"社会Agent测试失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

    logger.info("社会Agent测试模式完成")


def test_planner_agent(args):
    """测试规划Agent"""
    logger.info("启动规划Agent测试模式")

    # 创建规划Agent
    try:
        # 创建LLM接口，根据参数决定是否使用模拟模式
        if args.llm_config:
            logger.info(f"使用配置文件: {args.llm_config}")
            llm = LLMInterface.from_config_file(args.llm_config, logger=logger, mock=args.mock)
        else:
            llm = LLMInterface(mock_mode=args.mock, logger=logger)

        logger.info(f"LLM接口初始化成功: {llm}")

        # 导入规划Agent
        from agents.planner_agent import PlannerAgent

        # 使用LLM接口创建规划Agent
        planner_agent = PlannerAgent(llm_interface=llm, logger=logger)
        logger.info(f"规划Agent创建成功: {planner_agent}")

        # 设置默认数据路径
        standards_data = args.standards_data
        if standards_data is None:
            standards_data = os.path.join(ROOT_DIR, "data", "processed", "industry_standards.json")

        # 检查数据文件是否存在
        if not os.path.exists(standards_data) and not args.mock:
            logger.error(f"行业规范数据文件不存在: {standards_data}")
            return

        # 准备测试数据
        if args.mock:
            # 使用模拟数据
            standards = [
                {
                    "title": "API服务质量标准",
                    "description": "API服务应保证99.9%的可用性，平均响应时间不超过200ms。",
                    "category": "quality",
                    "importance": "high"
                },
                {
                    "title": "API安全标准",
                    "description": "所有API调用必须经过身份验证和授权，敏感数据必须加密传输。",
                    "category": "security",
                    "importance": "high"
                },
                {
                    "title": "API定价标准",
                    "description": "API定价应基于调用次数和数据量，并提供免费的基础套餐。",
                    "category": "business",
                    "importance": "medium"
                }
            ]

            environment_boundaries = {
                "upper_boundary": {
                    "基础设施类": 50.0,
                    "生活服务类": 80.0,
                    "企业管理类": 60.0,
                    "社交娱乐类": 70.0
                },
                "lower_boundary": {
                    "基础设施类": -30.0,
                    "生活服务类": -20.0,
                    "企业管理类": -25.0,
                    "社交娱乐类": -40.0
                }
            }

            social_relationships = {
                "communities": {
                    "0": ["API1", "API2", "API3"],
                    "1": ["API4", "API5", "API6"],
                    "2": ["API7", "API8", "API9"]
                },
                "key_nodes": [
                    {"name": "API1", "centrality": 0.8},
                    {"name": "API4", "centrality": 0.7},
                    {"name": "API7", "centrality": 0.6}
                ],
                "cross_community_edges": [
                    {"source": "API1", "target": "API4", "weight": 0.5},
                    {"source": "API4", "target": "API7", "weight": 0.4}
                ]
            }
        else:
            # 加载行业规范数据
            with open(standards_data, "r", encoding="utf-8") as f:
                standards = json.load(f)

            # 从环境Agent获取边界数据
            from agents.environment_agent import EnvironmentAgent
            env_agent = EnvironmentAgent(llm_interface=llm, logger=logger)

            # 生成极端情景
            result = env_agent.generate_extreme_scenario({
                "historical_data": {},
                "domain_knowledge": [],
                "time_range": "2006-2022",
                "api_categories": EXPERIMENT_CONFIG["api_categories"]
            })

            if result["status"] == "success":
                scenario = result["scenario"]
                # 提取上下限边界
                upper_boundary = {}
                lower_boundary = {}
                for category, value in scenario["demand_changes"].items():
                    # 使用需求变化的1.5倍作为上限，0.5倍作为下限
                    upper_boundary[category] = value * 1.5 if value > 0 else value * 0.5
                    lower_boundary[category] = value * 0.5 if value > 0 else value * 1.5

                environment_boundaries = {
                    "upper_boundary": upper_boundary,
                    "lower_boundary": lower_boundary
                }
            else:
                logger.error(f"生成极端情景失败: {result.get('error', '未知错误')}")
                environment_boundaries = {
                    "upper_boundary": {},
                    "lower_boundary": {}
                }

            # 从社会Agent获取骨干关系数据
            social_agent_output_dir = os.path.join(ROOT_DIR, "output", "test_social_agent")
            backbone_relationships_file = os.path.join(social_agent_output_dir, "backbone_network.json")

            if os.path.exists(backbone_relationships_file):
                with open(backbone_relationships_file, "r", encoding="utf-8") as f:
                    social_relationships = json.load(f)
            else:
                # 使用模拟数据
                social_relationships = {
                    "communities": {
                        "0": ["API1", "API2", "API3"],
                        "1": ["API4", "API5", "API6"],
                        "2": ["API7", "API8", "API9"]
                    },
                    "key_nodes": [
                        {"name": "API1", "centrality": 0.8},
                        {"name": "API4", "centrality": 0.7},
                        {"name": "API7", "centrality": 0.6}
                    ],
                    "cross_community_edges": [
                        {"source": "API1", "target": "API4", "weight": 0.5},
                        {"source": "API4", "target": "API7", "weight": 0.4}
                    ]
                }

        # 加载数据到规划Agent
        planner_agent.standards = standards
        planner_agent.set_environment_boundaries(environment_boundaries)
        planner_agent.set_social_relationships(social_relationships)

        # 生成约束条件
        logger.info("生成约束条件")
        constraints = planner_agent.generate_constraints()

        # 如果成功生成约束条件，则编译第一个约束条件
        compiled_expression = None
        if constraints:
            logger.info(f"生成了{len(constraints)}个约束条件")
            logger.info(f"编译第一个约束条件")
            compiled_expression = planner_agent.compile_constraint(constraints[0])
        else:
            logger.warning("未生成任何约束条件")

        # 输出结果
        logger.info(f"生成的约束条件数量: {len(constraints)}")
        if constraints:
            logger.info(f"示例约束条件: {constraints[0].get('description', '无描述')}")

        if compiled_expression:
            logger.info(f"编译后的表达式: {compiled_expression.get('description', '无表达式')}")

        # 保存结果
        output_dir = os.path.join(ROOT_DIR, "output", "test_planner_agent")
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "constraints.json"), "w", encoding="utf-8") as f:
            json.dump(constraints, f, ensure_ascii=False, indent=2)

        if compiled_expression:
            with open(os.path.join(output_dir, "compiled_expression.json"), "w", encoding="utf-8") as f:
                json.dump(compiled_expression, f, ensure_ascii=False, indent=2)

        logger.info(f"测试结果已保存到: {output_dir}")

    except Exception as e:
        logger.error(f"规划Agent测试失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

    logger.info("规划Agent测试模式完成")


def run_mode(args):
    """运行模式，执行完整的情景生成流程"""
    logger.info("启动运行模式")

    # 创建输出目录
    output_dir = args.output_dir or os.path.join(ROOT_DIR, "output", "full_run")
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 创建LLM接口
        if args.llm_config:
            logger.info(f"使用配置文件: {args.llm_config}")
            llm = LLMInterface.from_config_file(args.llm_config, logger=logger, mock=args.mock)
        else:
            llm = LLMInterface(mock_mode=args.mock, logger=logger)

        logger.info(f"LLM接口初始化成功: {llm}")

        # 1. 创建环境Agent并生成极端情景
        logger.info("步骤1: 生成极端情景")
        env_agent = EnvironmentAgent(
            llm_interface=llm,
            logger=logger
        )

        env_result = env_agent.generate_extreme_scenario({
            "historical_data": {},
            "domain_knowledge": [],
            "time_range": "2006-2022",
            "api_categories": EXPERIMENT_CONFIG["api_categories"]
        })

        if env_result["status"] != "success":
            logger.error(f"生成极端情景失败: {env_result.get('error', '未知错误')}")
            return

        scenario = env_result["scenario"]
        logger.info(f"生成极端情景成功: {scenario['title']}")

        # 保存极端情景
        with open(os.path.join(output_dir, "extreme_scenario.json"), "w", encoding="utf-8") as f:
            json.dump(scenario, f, ensure_ascii=False, indent=2)

        # 2. 创建社会Agent并构建协同网络
        logger.info("步骤2: 构建协同网络")
        social_agent = SocialAgent(
            llm_interface=llm,
            logger=logger
        )

        # 设置默认数据路径
        api_data = os.path.join(ROOT_DIR, "data", "processed", "api_data.json")
        relationship_data = os.path.join(ROOT_DIR, "data", "processed", "relationship_data.json")

        # 检查数据文件是否存在
        if not os.path.exists(api_data) or not os.path.exists(relationship_data):
            logger.warning("API数据文件或关系数据文件不存在，使用模拟数据")
            social_result = social_agent.construct_collaborative_network({
                "api_data": {},
                "api_count": 100,
                "api_categories": EXPERIMENT_CONFIG["api_categories"],
                "relationship_data": {}
            })
        else:
            # 运行社会Agent
            social_result = social_agent.run(api_data, relationship_data, output_dir)

        if isinstance(social_result, dict) and social_result.get("status") == "failed":
            logger.error(f"构建协同网络失败: {social_result.get('error', '未知错误')}")
            return

        logger.info(f"构建协同网络成功")

        # 3. 创建规划Agent并生成约束条件
        logger.info("步骤3: 生成约束条件和实验规则")
        planner_agent = PlannerAgent(
            llm_interface=llm,
            logger=logger
        )

        # 设置默认数据路径
        standards_data = args.standards_data
        if standards_data is None:
            standards_data = os.path.join(ROOT_DIR, "data", "processed", "industry_standards.json")

        # 检查数据文件是否存在
        if not os.path.exists(standards_data) and not args.mock:
            logger.error(f"行业规范数据文件不存在: {standards_data}")
            return

        # 加载行业规范数据
        if os.path.exists(standards_data):
            with open(standards_data, "r", encoding="utf-8") as f:
                standards = json.load(f)
        else:
            # 使用模拟数据
            standards = [
                {
                    "title": "API服务质量标准",
                    "description": "API服务应保证99.9%的可用性，平均响应时间不超过200ms。",
                    "category": "quality",
                    "importance": "high"
                },
                {
                    "title": "API安全标准",
                    "description": "所有API调用必须经过身份验证和授权，敏感数据必须加密传输。",
                    "category": "security",
                    "importance": "high"
                },
                {
                    "title": "API定价标准",
                    "description": "API定价应基于调用次数和数据量，并提供免费的基础套餐。",
                    "category": "business",
                    "importance": "medium"
                }
            ]

        # 提取上下限边界
        upper_boundary = {}
        lower_boundary = {}
        for category, value in scenario["demand_changes"].items():
            # 使用需求变化的1.5倍作为上限，0.5倍作为下限
            upper_boundary[category] = value * 1.5 if value > 0 else value * 0.5
            lower_boundary[category] = value * 0.5 if value > 0 else value * 1.5

        environment_boundaries = {
            "upper_boundary": upper_boundary,
            "lower_boundary": lower_boundary
        }

        # 从社会Agent获取骨干关系数据
        if isinstance(social_result, dict) and "backbone_network" in social_result:
            social_relationships = social_result["backbone_network"]
        else:
            # 尝试从输出目录加载骨干网络数据
            backbone_network_file = os.path.join(output_dir, "backbone_network.json")
            if os.path.exists(backbone_network_file):
                with open(backbone_network_file, "r", encoding="utf-8") as f:
                    social_relationships = json.load(f)
            else:
                # 使用模拟数据
                social_relationships = {
                    "communities": {
                        "0": ["API1", "API2", "API3"],
                        "1": ["API4", "API5", "API6"],
                        "2": ["API7", "API8", "API9"]
                    },
                    "key_nodes": [
                        {"name": "API1", "centrality": 0.8},
                        {"name": "API4", "centrality": 0.7},
                        {"name": "API7", "centrality": 0.6}
                    ],
                    "cross_community_edges": [
                        {"source": "API1", "target": "API4", "weight": 0.5},
                        {"source": "API4", "target": "API7", "weight": 0.4}
                    ]
                }

        # 加载数据到规划Agent
        planner_agent.standards = standards
        planner_agent.set_environment_boundaries(environment_boundaries)
        planner_agent.set_social_relationships(social_relationships)

        # 生成约束条件
        logger.info("生成约束条件")
        constraints = planner_agent.generate_constraints()

        if not constraints:
            logger.warning("未生成任何约束条件")
        else:
            logger.info(f"生成了{len(constraints)}个约束条件")

            # 编译约束条件
            logger.info("编译约束条件")
            compiled_expressions = []
            for constraint in constraints[:3]:  # 只编译前3个约束条件
                logger.info(f"编译约束条件: {constraint['id']}")
                expression = planner_agent.compile_constraint(constraint)
                if expression:
                    compiled_expressions.append(expression)

            logger.info(f"编译了{len(compiled_expressions)}个约束条件")

            # 保存结果
            with open(os.path.join(output_dir, "constraints.json"), "w", encoding="utf-8") as f:
                json.dump(constraints, f, ensure_ascii=False, indent=2)

            with open(os.path.join(output_dir, "compiled_expressions.json"), "w", encoding="utf-8") as f:
                json.dump(compiled_expressions, f, ensure_ascii=False, indent=2)

        logger.info("API服务生态系统情景生成完成")
        logger.info(f"所有结果已保存到: {output_dir}")

    except Exception as e:
        logger.error(f"运行失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


def visualize_mode(args):
    """可视化模式，生成环境边界图表"""
    logger.info("启动可视化模式")

    try:
        # 导入可视化模块
        from utils.visualization import (
            load_historical_data,
            plot_environment_boundaries,
            plot_environment_boundaries_from_files,
            plot_multi_charts,
            plot_scenario_comparison
        )

        # 设置默认输出目录
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = os.path.join(ROOT_DIR, "output", "visualization")
        os.makedirs(output_dir, exist_ok=True)

        # 设置默认输出文件
        output_file = args.output_file
        if output_file is None:
            output_file = os.path.join(output_dir, "environment_boundaries.png")

        # 设置历史数据文件
        historical_data_file = args.historical_data_file
        if historical_data_file is None:
            historical_data_file = os.path.join(ROOT_DIR, "data", "processed", "historical_data.json")

        # 设置情景文件
        scenario_file = args.scenario_file
        if scenario_file is None:
            scenario_file = os.path.join(ROOT_DIR, "data", "scenarios", "extreme_scenario.json")

        # 检查历史数据文件是否存在
        if not os.path.exists(historical_data_file):
            logger.error(f"历史数据文件不存在: {historical_data_file}")
            logger.error("请确保已从programmableweb_2022(1).sql和programmableweb_dataset(1).sql导入真实的历史数据。")
            return

        # 加载历史数据
        try:
            historical_data = load_historical_data(historical_data_file)
            logger.info("成功加载历史数据")
        except Exception as e:
            logger.error(f"加载历史数据失败: {str(e)}")
            return

        # 检查情景文件是否存在
        if not os.path.exists(scenario_file):
            logger.error(f"情景文件不存在: {scenario_file}")
            logger.error("请确保已从programmableweb_2022(1).sql和programmableweb_dataset(1).sql导入真实的历史数据。")
            return

        # 加载情景数据
        try:
            # 尝试多种编码加载数据
            scenario_data = None
            encodings = ['utf-8-sig', 'utf-8', 'utf-16', 'gbk', 'latin1', None]

            for encoding in encodings:
                try:
                    if encoding:
                        with open(scenario_file, 'r', encoding=encoding) as f:
                            scenario_data = json.load(f)
                    else:
                        # 二进制读取，尝试自动检测编码
                        with open(scenario_file, 'rb') as f:
                            content = f.read()
                            import chardet
                            detected = chardet.detect(content)
                            encoding = detected['encoding']
                            if encoding:
                                scenario_data = json.loads(content.decode(encoding))

                    if scenario_data:
                        logger.info(f"成功使用 {encoding} 编码加载情景数据")
                        break
                except Exception as e:
                    logger.warning(f"使用 {encoding} 编码加载情景数据失败: {str(e)}")
                    continue

            if not scenario_data:
                logger.error("无法加载情景数据文件，尝试了多种编码都失败")
                return

            logger.info("成功加载情景数据")
        except Exception as e:
            logger.error(f"加载情景数据失败: {str(e)}")
            return

        # 绘制环境边界图表
        logger.info(f"绘制环境边界图表: {output_file}")

        # 从情景数据中提取上限和下限边界
        upper_boundary = {}
        lower_boundary = {}

        # 检查情景数据中是否包含边界信息
        if "boundaries" in scenario_data:
            if "upper" in scenario_data["boundaries"]:
                upper_boundary = scenario_data["boundaries"]["upper"]
            if "lower" in scenario_data["boundaries"]:
                lower_boundary = scenario_data["boundaries"]["lower"]
        else:
            # 如果没有边界信息，则使用默认值
            for category in historical_data.keys():
                # 使用历史数据的最大值和最小值的±20%作为边界
                values = list(historical_data[category].values())
                max_value = max(values) * 1.2
                min_value = min(values) * 0.8
                upper_boundary[category] = max_value
                lower_boundary[category] = min_value

        output_path = plot_environment_boundaries(
            historical_data=historical_data,
            upper_boundary=upper_boundary,
            lower_boundary=lower_boundary,
            output_path=output_file,
            title=args.vis_title,
            figsize=(12, 8),
            dpi=args.fig_dpi
        )

        # 绘制多图表
        multi_chart_path = os.path.join(output_dir, "multi_charts.png")
        logger.info(f"绘制多图表: {multi_chart_path}")

        # 使用加载的数据绘制多图表
        multi_chart_path = plot_multi_charts(
            scenarios=[scenario_data],  # 将单个情景放入列表中
            historical_data=historical_data,
            output_path=multi_chart_path,
            title=f"{args.vis_title} - 多图表",
            figsize=(16, 12),
            dpi=args.fig_dpi
        )

        # 绘制情景对比图表
        comparison_chart_path = os.path.join(output_dir, "scenario_comparison.png")
        logger.info(f"绘制情景对比图表: {comparison_chart_path}")

        # 使用加载的数据绘制情景对比图表
        comparison_chart_path = plot_scenario_comparison(
            scenarios=[scenario_data],  # 将单个情景放入列表中
            output_path=comparison_chart_path,
            title=f"{args.vis_title} - 情景对比",
            figsize=(14, 10),
            dpi=args.fig_dpi
        )

        logger.info(f"环境边界图表已生成: {output_path}")
        logger.info(f"多图表已生成: {multi_chart_path}")
        logger.info(f"情景对比图表已生成: {comparison_chart_path}")

        # 自动打开图表文件
        try:
            import platform
            import subprocess

            logger.info("尝试打开生成的文件...")

            # 根据操作系统选择打开方式
            if platform.system() == "Windows":
                os.startfile(output_path)
                os.startfile(multi_chart_path)
                os.startfile(comparison_chart_path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.call(["open", output_path])
                subprocess.call(["open", multi_chart_path])
                subprocess.call(["open", comparison_chart_path])
            else:  # Linux
                subprocess.call(["xdg-open", output_path])
                subprocess.call(["xdg-open", multi_chart_path])
                subprocess.call(["xdg-open", comparison_chart_path])

            logger.info("文件已打开")
        except Exception as e:
            logger.warning(f"无法自动打开文件: {str(e)}")
            logger.info(f"请手动打开图表文件: {output_path}")

    except Exception as e:
        logger.error(f"可视化失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

    logger.info("可视化模式完成")


def network_vis_mode(args, auto_open=True):
    """网络可视化模式，生成计算实验设计框架图表

    Args:
        args: 命令行参数
        auto_open: 是否自动打开生成的图表文件
    """
    logger.info("启动网络可视化模式")

    try:
        # 导入网络可视化模块
        from utils.network_visualizer import load_network_data, plot_network_metrics_comparison
        from utils.network_visualization import plot_network_comparison
        # 设置默认输出目录
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = os.path.join(ROOT_DIR, "output", "network_vis")
        os.makedirs(output_dir, exist_ok=True)

        # 设置默认路径
        network_data_file = args.network_file
        if network_data_file is None:
            network_data_file = os.path.join(output_dir, "network_data.json")
            logger.info(f"使用默认网络数据文件路径: {network_data_file}")

        # 检查网络数据文件是否存在
        if not os.path.exists(network_data_file):
            logger.error(f"网络数据文件不存在: {network_data_file}")
            logger.error("请确保已从programmableweb_2022(1).sql和programmableweb_dataset(1).sql导入真实的历史数据。")
            return

        # 加载网络数据
        try:
            network_data = load_network_data(network_data_file)
            logger.info("成功加载网络数据")
        except Exception as e:
            logger.error(f"加载网络数据失败: {str(e)}")
            return

        # 1. 绘制网络可视化图表
        logger.info(f"绘制网络可视化图表: {args.network_title}")

        # 设置输出文件
        network_output = os.path.join(output_dir, "network_visualization.png")

        # 解析年份和方法参数
        years = args.network_years.split(',') if isinstance(args.network_years, str) else args.network_years
        methods = args.network_methods.split(',') if isinstance(args.network_methods, str) else args.network_methods

        # 提取网络数据
        networks = network_data.get('networks', {})

        # 调用网络可视化函数 - 直接使用plot_network_comparison而不是plot_experiment_framework
        network_output = plot_network_comparison(
            networks=networks,
            methods=methods,
            years=years,
            output_path=network_output,
            title=args.network_title,
            figsize=(args.fig_width, args.fig_height),
            dpi=args.fig_dpi,
            node_size=150,  # 增加节点大小
            edge_width=1.0   # 增加边宽度
        )

        logger.info(f"图表已生成: {network_output}")

        # 2. 绘制网络指标比较图表
        logger.info(f"绘制网络指标比较图表")
        metrics_output = os.path.join(output_dir, "network_metrics_comparison.png")

        # 提取网络数据中的指标
        metrics_data = {}
        for year, year_data in networks.items():
            metrics_data[year] = {}
            for method, method_data in year_data.items():
                if 'metrics' in method_data:
                    metrics_data[year][method] = method_data['metrics']

        # 创建网络指标对比图
        try:
            # 直接调用plot_network_metrics_comparison函数创建网络指标对比图
            plot_network_metrics_comparison(
                networks=networks,
                methods=methods,
                years=years,
                output_path=metrics_output,
                title="API网络指标对比分析",
                figsize=(16, 12),
                dpi=args.fig_dpi
            )

            logger.info(f"指标比较图表已生成: {metrics_output}")
        except Exception as e:
            logger.error(f"生成指标比较图表失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

        # 自动打开图表文件
        if auto_open:
            try:
                import platform
                import subprocess

                logger.info("尝试打开生成的文件...")

                # 根据操作系统选择打开方式
                if platform.system() == "Windows":
                    os.startfile(network_output)
                    os.startfile(metrics_output)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.call(["open", network_output])
                    subprocess.call(["open", metrics_output])
                else:  # Linux
                    subprocess.call(["xdg-open", network_output])
                    subprocess.call(["xdg-open", metrics_output])

                logger.info("文件已打开")
            except Exception as e:
                logger.warning(f"无法自动打开文件: {str(e)}")
                logger.info(f"请手动打开图表文件: {network_output}")

    except Exception as e:
        logger.error(f"网络可视化失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

    logger.info("网络可视化模式完成")


def radar_vis_mode(args):
    """雷达图可视化模式，生成多维情景向量对比雷达图"""
    logger.info("启动雷达图可视化模式")

    try:
        # 导入雷达图可视化模块
        from utils.radar_visualizer import load_metrics_data, plot_radar_comparison

        # 设置默认输出目录
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = os.path.join(ROOT_DIR, "output", "radar_vis")
        os.makedirs(output_dir, exist_ok=True)

        # 设置默认路径
        radar_data_file = args.radar_file
        if radar_data_file is None:
            radar_data_file = os.path.join(output_dir, "radar_data.json")
            logger.info(f"使用默认雷达图数据文件路径: {radar_data_file}")

        # 检查雷达图数据是否存在
        if not os.path.exists(radar_data_file):
            logger.error(f"雷达图数据文件不存在: {radar_data_file}")
            logger.error("请确保已从programmableweb_2022(1).sql和programmableweb_dataset(1).sql导入真实的历史数据。")
            return

        # 加载雷达图数据
        try:
            radar_data = load_metrics_data(radar_data_file)
            logger.info("成功加载雷达图数据")

            # 验证并补全数据
            years = radar_data.get("years", [2006, 2010, 2015, 2020])
            methods = radar_data.get("methods", ["Original", "Cluster", "GT", "HSS", "PLA", "Ours"])
            dimensions = radar_data.get("dimensions", ["Weight Fraction", "Weight Entropy", "Node Fraction",
                                                      "Value Entropy", "Similarity", "Sum Activity",
                                                      "Reachability", "LCC Size"])

            # 确保数据字典中包含所有必要的字段
            if "years" not in radar_data:
                radar_data["years"] = years
            if "methods" not in radar_data:
                radar_data["methods"] = methods
            if "dimensions" not in radar_data:
                radar_data["dimensions"] = dimensions
            if "metrics" not in radar_data:
                radar_data["metrics"] = {}

            # 确保metrics中包含所有年份的数据
            metrics = radar_data["metrics"]
            for year in years:
                year_str = str(year)
                if year_str not in metrics:
                    logger.warning(f"雷达图数据中缺少{year}年的数据，将使用默认值")
                    metrics[year_str] = {}

                # 确保每个年份都包含所有方法的数据
                for method in methods:
                    if method not in metrics[year_str]:
                        logger.warning(f"雷达图数据中{year}年缺少{method}的数据，将使用默认值")
                        metrics[year_str][method] = [0.5] * len(dimensions)  # 使用0.5作为默认值，使图形更明显
                    elif not metrics[year_str][method]:  # 如果数据为空列表
                        metrics[year_str][method] = [0.5] * len(dimensions)
                    else:
                        # 确保数据维度与dimensions一致
                        current_len = len(metrics[year_str][method])
                        if current_len < len(dimensions):
                            logger.warning(f"雷达图数据中{year}年{method}的维度数量不足，将补充默认值")
                            metrics[year_str][method].extend([0.5] * (len(dimensions) - current_len))
                        elif current_len > len(dimensions):
                            logger.warning(f"雷达图数据中{year}年{method}的维度数量过多，将截断")
                            metrics[year_str][method] = metrics[year_str][method][:len(dimensions)]

            logger.info(f"数据验证和补全完成，包含{len(years)}个年份，{len(methods)}种方法")

        except Exception as e:
            logger.error(f"加载雷达图数据失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return

        # 设置输出文件
        radar_output = os.path.join(output_dir, "radar_comparison.png")

        # 解析高亮方法参数
        highlight_methods = args.highlight_methods.split(',') if isinstance(args.highlight_methods, str) else args.highlight_methods

        # 绘制雷达图
        radar_output = plot_radar_comparison(
            data=radar_data,
            output_path=radar_output,
            title=args.radar_title,
            figsize=(args.fig_width, args.fig_height),
            dpi=args.fig_dpi,
            highlight_methods=highlight_methods
        )

        logger.info(f"雷达图已生成: {radar_output}")

        # 自动打开图表文件
        try:
            import platform
            import subprocess

            logger.info("尝试打开生成的文件...")

            # 根据操作系统选择打开方式
            if platform.system() == "Windows":
                os.startfile(radar_output)
            elif platform.system() == "Darwin":  # macOS
                subprocess.call(["open", radar_output])
            else:  # Linux
                subprocess.call(["xdg-open", radar_output])

            logger.info("文件已打开")
        except Exception as e:
            logger.warning(f"无法自动打开文件: {str(e)}")
            logger.info(f"请手动打开图表文件: {radar_output}")

    except Exception as e:
        logger.error(f"雷达图可视化失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

    logger.info("雷达图可视化模式完成")


def visualize_scenarios(scenarios_file: str, output_dir: str) -> None:
    """
    可视化生成的情景

    Args:
        scenarios_file: 情景数据文件路径
        output_dir: 输出目录
    """
    logger.info(f"开始可视化情景数据: {scenarios_file}")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 加载情景数据
        with open(scenarios_file, 'r', encoding='utf-8') as f:
            scenarios_data = json.load(f)

        # 提取类别和情景数据
        categories = scenarios_data.get('categories', [])
        scenarios = scenarios_data.get('scenarios', [])

        if not categories or not scenarios:
            logger.error("情景数据格式不正确或为空")
            return

        # 初始化雷达图可视化器
        radar_viz = RadarVisualizer(logger=logger)

        # 生成所有情景的对比雷达图
        all_scenarios_data = []
        for scenario in scenarios:
            scenario_name = scenario.get('name', '未命名情景')
            scenario_values = scenario.get('demand_changes', {})

            # 确保值的顺序与类别顺序一致
            values = [scenario_values.get(cat, 0) for cat in categories]
            all_scenarios_data.append({
                'name': scenario_name,
                'values': values
            })

        # 生成雷达图
        output_file = os.path.join(output_dir, 'scenarios_comparison.png')
        radar_viz.create_radar_comparison(
            scenario_data=all_scenarios_data,
            categories=categories,
            title='API需求变化情景对比',
            output_file=output_file
        )
        logger.info(f"情景对比雷达图已保存至: {output_file}")

        # 为每个情景单独生成雷达图
        for scenario in scenarios:
            scenario_name = scenario.get('name', '未命名情景')
            scenario_values = scenario.get('demand_changes', {})

            # 确保值的顺序与类别顺序一致
            values = [scenario_values.get(cat, 0) for cat in categories]

            # 生成单个情景雷达图
            scenario_file = os.path.join(output_dir, f'scenario_{scenario_name}.png')
            radar_viz.create_radar_chart(
                data=[(scenario_name, values)],
                categories=categories,
                title=f'情景: {scenario_name}',
                output_file=scenario_file
            )
            logger.info(f"情景 '{scenario_name}' 雷达图已保存至: {scenario_file}")

    except Exception as e:
        logger.error(f"可视化情景数据失败: {e}")
        import traceback
        traceback.print_exc()

def visualize_network(network_file: str, output_dir: str) -> None:
    """
    可视化API协作网络

    Args:
        network_file: 网络数据文件路径
        output_dir: 输出目录
    """
    logger.info(f"开始可视化网络数据: {network_file}")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 加载网络数据
        with open(network_file, 'r', encoding='utf-8') as f:
            network_data = json.load(f)

        # 提取节点和边数据
        nodes = network_data.get('nodes', [])
        edges = network_data.get('edges', [])

        if not nodes or not edges:
            logger.error("网络数据格式不正确或为空")
            return

        # 初始化网络可视化器
        network_viz = NetworkVisualizer(logger=logger)

        # 生成网络图
        output_file = os.path.join(output_dir, 'api_network.png')
        network_viz.create_network_graph(
            nodes=nodes,
            edges=edges,
            title='API协作网络',
            output_file=output_file,
            show_labels=True
        )
        logger.info(f"API协作网络图已保存至: {output_file}")

        # 生成社区结构图
        community_file = os.path.join(output_dir, 'api_communities.png')
        network_viz.create_community_graph(
            nodes=nodes,
            edges=edges,
            title='API社区结构',
            output_file=community_file,
            community_detection='louvain'
        )
        logger.info(f"API社区结构图已保存至: {community_file}")

    except Exception as e:
        logger.error(f"可视化网络数据失败: {e}")
        import traceback
        traceback.print_exc()

def run_visualization(args):
    """
    运行可视化模式

    Args:
        args: 命令行参数
    """
    logger.info("开始运行可视化模式")

    # 设置输出目录
    output_dir = args.output_dir or os.path.join(os.path.dirname(__file__), '../output/visualization')
    os.makedirs(output_dir, exist_ok=True)

    # 可视化情景数据
    if args.scenarios_file:
        visualize_scenarios(args.scenarios_file, output_dir)

    # 可视化网络数据
    if args.network_file:
        visualize_network(args.network_file, output_dir)

    logger.info("可视化模式完成")

def all_vis_mode(args):
    """一次性生成所有可视化图表模式"""
    logger.info("启动一次性生成所有可视化图表模式")

    try:
        # 依次执行三种可视化模式，但不自动打开文件
        logger.info("1. 执行网络可视化")
        network_vis_mode(args, auto_open=False)

        logger.info("2. 执行雷达图可视化")
        # 修改雷达图可视化函数调用，添加auto_open参数
        radar_vis_mode_with_auto_open(args, auto_open=False)

        logger.info("3. 执行情景可视化")
        visualize_mode(args)

        logger.info("所有可视化图表已生成完成")
    except Exception as e:
        logger.error(f"一次性生成所有可视化图表失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

    logger.info("一次性生成所有可视化图表模式完成")


def radar_vis_mode_with_auto_open(args, auto_open=True):
    """雷达图可视化模式，生成多维情景向量对比雷达图，并可选择是否自动打开文件

    Args:
        args: 命令行参数
        auto_open: 是否自动打开生成的图表文件
    """
    logger.info("启动雷达图可视化模式")

    try:
        # 导入雷达图可视化模块
        from utils.radar_visualizer import load_metrics_data, plot_radar_comparison

        # 设置默认输出目录
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = os.path.join(ROOT_DIR, "output", "radar_vis")
        os.makedirs(output_dir, exist_ok=True)

        # 设置默认路径
        radar_data_file = args.radar_file
        if radar_data_file is None:
            radar_data_file = os.path.join(output_dir, "radar_data.json")
            logger.info(f"使用默认雷达图数据文件路径: {radar_data_file}")

        # 检查雷达图数据是否存在
        if not os.path.exists(radar_data_file):
            logger.error(f"雷达图数据文件不存在: {radar_data_file}")
            logger.error("请确保已从programmableweb_2022(1).sql和programmableweb_dataset(1).sql导入真实的历史数据。")
            return

        # 加载雷达图数据
        try:
            radar_data = load_metrics_data(radar_data_file)
            logger.info("成功加载雷达图数据")

            # 验证并补全数据
            years = radar_data.get("years", [2006, 2010, 2015, 2020])
            methods = radar_data.get("methods", ["Original", "Cluster", "GT", "HSS", "PLA", "Ours"])
            dimensions = radar_data.get("dimensions", ["Weight Fraction", "Weight Entropy", "Node Fraction",
                                                      "Value Entropy", "Similarity", "Sum Activity",
                                                      "Reachability", "LCC Size"])

            # 确保数据字典中包含所有必要的字段
            if "years" not in radar_data:
                radar_data["years"] = years
            if "methods" not in radar_data:
                radar_data["methods"] = methods
            if "dimensions" not in radar_data:
                radar_data["dimensions"] = dimensions
            if "metrics" not in radar_data:
                radar_data["metrics"] = {}

            # 确保metrics中包含所有年份的数据
            metrics = radar_data["metrics"]
            for year in years:
                year_str = str(year)
                if year_str not in metrics:
                    logger.warning(f"雷达图数据中缺少{year}年的数据，将使用默认值")
                    metrics[year_str] = {}

                # 确保每个年份都包含所有方法的数据
                for method in methods:
                    if method not in metrics[year_str]:
                        logger.warning(f"雷达图数据中{year}年缺少{method}的数据，将使用默认值")
                        metrics[year_str][method] = [0.5] * len(dimensions)  # 使用0.5作为默认值，使图形更明显
                    elif not metrics[year_str][method]:  # 如果数据为空列表
                        metrics[year_str][method] = [0.5] * len(dimensions)
                    else:
                        # 确保数据维度与dimensions一致
                        current_len = len(metrics[year_str][method])
                        if current_len < len(dimensions):
                            logger.warning(f"雷达图数据中{year}年{method}的维度数量不足，将补充默认值")
                            metrics[year_str][method].extend([0.5] * (len(dimensions) - current_len))
                        elif current_len > len(dimensions):
                            logger.warning(f"雷达图数据中{year}年{method}的维度数量过多，将截断")
                            metrics[year_str][method] = metrics[year_str][method][:len(dimensions)]

            logger.info(f"数据验证和补全完成，包含{len(years)}个年份，{len(methods)}种方法")

        except Exception as e:
            logger.error(f"加载雷达图数据失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return

        # 设置输出文件
        radar_output = os.path.join(output_dir, "radar_comparison.png")

        # 解析高亮方法参数
        highlight_methods = args.highlight_methods.split(',') if isinstance(args.highlight_methods, str) else args.highlight_methods

        # 绘制雷达图
        radar_output = plot_radar_comparison(
            data=radar_data,
            output_path=radar_output,
            title=args.radar_title,
            figsize=(args.fig_width, args.fig_height),
            dpi=args.fig_dpi,
            highlight_methods=highlight_methods
        )

        logger.info(f"雷达图已生成: {radar_output}")

        # 自动打开图表文件
        if auto_open:
            try:
                import platform
                import subprocess

                logger.info("尝试打开生成的文件...")

                # 根据操作系统选择打开方式
                if platform.system() == "Windows":
                    os.startfile(radar_output)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.call(["open", radar_output])
                else:  # Linux
                    subprocess.call(["xdg-open", radar_output])

                logger.info("文件已打开")
            except Exception as e:
                logger.warning(f"无法自动打开文件: {str(e)}")
                logger.info(f"请手动打开图表文件: {radar_output}")

    except Exception as e:
        logger.error(f"雷达图可视化失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

    logger.info("雷达图可视化模式完成")


def main():
    """主函数"""
    args = parse_args()

    if args.mode == 'test':
        test_mode(args)
    elif args.mode == 'run':
        run_mode(args)
    elif args.mode == 'test_env_agent':
        test_env_agent(args)
    elif args.mode == 'test_social_agent':
        test_social_agent(args)
    elif args.mode == 'test_planner_agent':
        test_planner_agent(args)
    elif args.mode == 'visualize':
        visualize_mode(args)
    elif args.mode == 'network_vis':
        network_vis_mode(args)
    elif args.mode == 'radar_vis':
        radar_vis_mode(args)
    elif args.mode == 'visualization':
        run_visualization(args)
    elif args.mode == 'all_vis':
        all_vis_mode(args)
    else:
        print(f"未知的运行模式: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()