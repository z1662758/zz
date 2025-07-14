"""
API服务生态系统情景生成 - 日志记录器
"""

import os
import logging
from datetime import datetime
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.config import ROOT_DIR


def setup_logger(name, log_file=None, level=logging.INFO):
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径，如果为None，则不记录到文件
        level: 日志级别
        
    Returns:
        日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除现有的处理器
    if logger.handlers:
        logger.handlers = []
    
    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 如果指定了日志文件，添加文件处理器
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_default_logger(log_level=None):
    """
    获取默认日志记录器
    
    Args:
        log_level: 日志级别，可以是'debug', 'info', 'warning', 'error'或None
        
    Returns:
        默认日志记录器
    """
    # 设置日志级别
    level = logging.INFO
    if log_level:
        if log_level.lower() == 'debug':
            level = logging.DEBUG
        elif log_level.lower() == 'info':
            level = logging.INFO
        elif log_level.lower() == 'warning':
            level = logging.WARNING
        elif log_level.lower() == 'error':
            level = logging.ERROR
    
    # 创建日志目录
    log_dir = ROOT_DIR / "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # 创建日志文件名，包含当前日期时间
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"api_ecosystem_{timestamp}.log"
    
    return setup_logger("api_ecosystem", log_file=str(log_file), level=level) 