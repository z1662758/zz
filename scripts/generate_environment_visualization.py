"""
API服务生态系统情景生成 - 极限环境边界可视化脚本
用于生成极限环境边界生成情况分析图表
"""

import os
import sys
import json
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(str(ROOT_DIR))

from utils.logger import get_default_logger
from utils.visualization import (
    generate_sample_historical_data,
    plot_environment_boundaries,
    plot_environment_boundaries_from_files
)
from experiments.config import EXPERIMENT_CONFIG


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='生成极限环境边界分析图表')
    parser.add_argument('--scenario_file', type=str, default=None,
                        help='情景文件路径，如果为None，则使用默认路径')
    parser.add_argument('--historical_data_file', type=str, required=True,
                        help='历史数据文件路径，必须提供')
    parser.add_argument('--output_file', type=str, default=None,
                        help='输出文件路径，如果为None，则使用默认路径')
    parser.add_argument('--title', type=str, default='极限环境边界生成情况分析',
                        help='图表标题')
    parser.add_argument('--start_year', type=int, default=2006,
                        help='起始年份')
    parser.add_argument('--end_year', type=int, default=2022,
                        help='结束年份')
    parser.add_argument('--width', type=int, default=12,
                        help='图表宽度')
    parser.add_argument('--height', type=int, default=8,
                        help='图表高度')
    parser.add_argument('--dpi', type=int, default=300,
                        help='图表分辨率')
    parser.add_argument('--generate_data_only', action='store_true',
                        help='仅检查历史数据，不绘制图表')
    return parser.parse_args()


def plot_from_files(scenario_file, historical_data_file, output_path, title, figsize, dpi):
    pass
def main():
    """主函数"""
    args = parse_args()
    logger = get_default_logger()
    logger.info("启动极限环境边界可视化脚本")
    
    # 设置默认路径
    if args.scenario_file is None:
        args.scenario_file = os.path.join(ROOT_DIR, "output", "full_run", "extreme_scenario.json")
    
    if args.output_file is None:
        output_dir = os.path.join(ROOT_DIR, "output", "visualizations")
        os.makedirs(output_dir, exist_ok=True)
        args.output_file = os.path.join(output_dir, "environment_boundaries.png")
    
    # 检查历史数据文件
    if args.historical_data_file is None:
        logger.error("未提供历史数据文件")
        raise ValueError("必须提供历史数据文件。模拟数据生成功能已被禁用。请使用从programmableweb_2022(1).sql和programmableweb_dataset(1).sql导入的真实数据。")
    
    # 如果只生成数据，则退出
    if args.generate_data_only:
        logger.info("已生成历史数据，退出")
        return
    
    # 检查情景文件是否存在
    if not os.path.exists(args.scenario_file):
        logger.error(f"情景文件不存在: {args.scenario_file}")
        
        # 创建一个模拟的情景文件
        logger.info("创建模拟情景文件")
        scenario = {
            "title": "模拟极端情景",
            "description": "这是一个模拟的极端情景，用于测试可视化功能",
            "demand_changes": {
                "基础设施类": 1200,
                "生活服务类": 2500,
                "企业管理类": 500,
                "社交娱乐类": 3500
            },
            "probability": 0.05,
            "duration": 12,
            "long_term_impact": "这是一个长期影响描述"
        }
        
        # 保存模拟情景
        os.makedirs(os.path.dirname(args.scenario_file), exist_ok=True)
        with open(args.scenario_file, 'w', encoding='utf-8') as f:
            json.dump(scenario, f, ensure_ascii=False, indent=2)
    
    # 绘制图表
    logger.info(f"绘制环境边界图表: {args.output_file}")
    output_path = plot_from_files(
        scenario_file=args.scenario_file,
        historical_data_file=args.historical_data_file,
        output_path=args.output_file,
        title=args.title,
        figsize=(args.width, args.height),
        dpi=args.dpi
    )
    
    logger.info(f"图表已生成: {output_path}")


if __name__ == "__main__":
    main() 