"""
API服务生态系统情景生成 - 配置文件
"""

import os
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 数据目录
DATA_DIR = ROOT_DIR / "data2"
RAW_DATA_DIR = DATA_DIR / "raw"                             #原数据（未加）
PROCESSED_DATA_DIR = DATA_DIR / "processed"                 #处理后的数据
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"            #知识库

# 实验配置，"random_seed"随机种子（多次运行结果保持一致，确保实验可复现）；
# "num_scenarios"生成的情景数量；"time_periods"时间跨度（2006-2022年）；
EXPERIMENT_CONFIG = {
    "random_seed": 42,
    "num_scenarios": 100,
    "time_periods": 17,  # 2006-2022年
    "api_categories": [
        "基础设施类",
        "生活服务类",
        "企业管理类",
        "社交娱乐类"
    ]
}

# 环境Agent配置，生成器和判别器通过对抗训练不断优化，最终生成器能产生以假乱真的数据。
#并定义了GAN模型的训练参数：learning_rate控制参数更新步长，batch_size设置每次训练的样本量，epochs决定训练轮次，gan_latent_dim指定生成器的输入噪声维度。这些参数共同影响模型训练的稳定性和生成数据的质量，较小的学习率(0.001)和大潜在空间维度(128)有助于生成更复杂的API市场场景。
ENVIRONMENT_AGENT_CONFIG = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "gan_latent_dim": 128,
    "adversarial_prompts": [           #提供4种极端场景的文本描述，指导GAN生成对应的模拟数据。
        "生成在经济衰退期间API市场的极端需求场景",
        "生成在技术革新爆发期API市场的极端需求场景",
        "生成在监管政策突变期API市场的极端需求场景",
        "生成在市场竞争白热化阶段API市场的极端需求场景"
    ]
}

# 社会Agent配置
SOCIAL_AGENT_CONFIG = {
    "relationship_threshold": 0.3,
    "community_detection_algorithm": "louvain",
    "backbone_extraction_method": "custom",  # 自定义骨干网络提取方法
    "max_communities": 10
}

# 规划Agent配置
PLANNER_AGENT_CONFIG = {
    "rule_generation_temperature": 0.7,
    "optimization_iterations": 50,
    "convergence_threshold": 0.01,
    "rag_top_k": 5
}

# 验证Agent配置
VERIFICATION_AGENT_CONFIG = {
    "hallucination_threshold": 0.8,
    "consistency_check_samples": 10,
    "validation_methods": ["knowledge_consistency", "historical_consistency", "logical_consistency"]
}

# LLM配置
LLM_CONFIG = {
    "model_name": "text-embedding-v3",  # 可替换为其他模型
    "temperature": 0.7,
    "max_tokens": 2000
}

# 可视化配置
VISUALIZATION_CONFIG = {
    "figure_size": (12, 8),
    "dpi": 300,
    "save_format": "png",
    "color_map": "viridis"
} 