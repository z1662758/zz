"""
API服务生态系统情景生成 - 数据导入脚本
从SQL文件中导入数据，并将其转换为项目可用的格式，独立运行可直接调用
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(str(ROOT_DIR))

from utils.logger import get_default_logger
from utils.data_loader import DataLoader         #调用data_loader


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='API服务生态系统数据导入工具')
    parser.add_argument('--api_sql', type=str, required=True,
                        help='API数据SQL文件路径')
    parser.add_argument('--dataset_sql', type=str, required=True,
                        help='数据集SQL文件路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录，如果为None，则使用默认输出目录')
    parser.add_argument('--log_level', type=str, default='info',
                        choices=['debug', 'info', 'warning', 'error'],
                        help='日志级别')
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    logger = get_default_logger(log_level=args.log_level)
    
    # 设置输出目录
    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join(ROOT_DIR, "data2", "processed")
    
    logger.info(f"开始导入数据")
    logger.info(f"API数据SQL文件: {args.api_sql}")
    logger.info(f"数据集SQL文件: {args.dataset_sql}")
    logger.info(f"输出目录: {output_dir}")
    
    # 创建数据加载器
    loader = DataLoader(logger=logger)
    
    # 加载数据
    success = loader.load_from_sql(args.api_sql, args.dataset_sql)
    if not success:
        logger.error("数据加载失败")
        return
    
    # 保存处理后的数据
    success = loader.save_processed_data(output_dir)
    if not success:
        logger.error("数据保存失败")
        return
    
    # 输出统计信息
    api_count = loader.get_api_count()
    categories = loader.get_api_categories()
    relationship_data = loader.get_relationship_data()
    
    logger.info(f"数据导入成功")
    logger.info(f"API数量: {api_count}")
    logger.info(f"API类别: {categories}")
    logger.info(f"调用关系数量: {len(relationship_data['call_relationships'])}")
    logger.info(f"相似性关系数量: {len(relationship_data['similarity_relationships'])}")


if __name__ == "__main__":
    main() 