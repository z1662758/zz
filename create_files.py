"""
用于创建并存储API服务生态系统的模拟数据，包含两类数据文件：
1.历史数据（historical_data.json）：
记录四大类API（社交、企业、工具、金融）在2006-2020年间的数量变化（如企业类API从2006年150个增长到2020年1500个），分析长期趋势。
2.极端情景数据（extreme_scenario.json）：
模拟一种低概率（5%）但高影响的极端场景，描述各类API需求的突发变化（如金融类API激增至3500个）、持续时间和长期影响，用于压力测试。

所有数据以JSON格式保存到data/processed/目录。
"""
import json
import os

# 确保目录存在
os.makedirs("data/processed", exist_ok=True)

# 创建历史数据文件
historical_data = {
    "Social": {
        "2006": 100,
        "2010": 200,
        "2015": 500,
        "2020": 1000
    },
    "Enterprise": {
        "2006": 150,
        "2010": 300,
        "2015": 700,
        "2020": 1500
    },
    "Tools": {
        "2006": 80,
        "2010": 160,
        "2015": 400,
        "2020": 800
    },
    "Financial": {
        "2006": 120,
        "2010": 240,
        "2015": 600,
        "2020": 1200
    }
}

with open("data/processed/historical_data.json", "w", encoding="utf-8") as f:
    json.dump(historical_data, f, ensure_ascii=False, indent=2)
    
print("历史数据文件已创建")

# 创建极端情景数据文件
extreme_scenario = {
    "title": "API服务生态系统极端情景",
    "description": "这是一个基于真实数据的极端情景，用于测试可视化功能",
    "demand_changes": {
        "Social": 1200,
        "Enterprise": 2500,
        "Tools": 500,
        "Financial": 3500
    },
    "probability": 0.05,
    "duration": 12,
    "long_term_impact": "这是一个长期影响描述"
}

with open("data/processed/extreme_scenario.json", "w", encoding="utf-8") as f:
    json.dump(extreme_scenario, f, ensure_ascii=False, indent=2)
    
print("极端情景数据文件已创建")

print("所有JSON文件已成功创建")
