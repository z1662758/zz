"""
API服务生态系统情景生成 - HTML报告生成工具
生成包含可视化结果的HTML报告
scenarios_comparison.png 由之前的 radar_comparison.png 相关代码生成
"""

import os
import json
import datetime
from typing import Dict, List, Any, Optional, Tuple

def generate_html_report(
    scenarios_file: Optional[str] = None,
    network_file: Optional[str] = None,
    visualization_dir: Optional[str] = None,
    output_file: Optional[str] = None,
    title: str = "API服务生态系统情景分析报告"
) -> str:
    """
    生成HTML报告
    
    Args:
        scenarios_file: 情景数据文件路径
        network_file: 网络数据文件路径
        visualization_dir: 可视化结果目录
        output_file: 输出文件路径
        title: 报告标题
        
    Returns:
        输出文件路径
    """
    # 设置默认输出文件
    if output_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"report_{timestamp}.html"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # 加载情景数据
    scenarios_data = None
    if scenarios_file and os.path.exists(scenarios_file):
        try:
            with open(scenarios_file, 'r', encoding='utf-8') as f:
                scenarios_data = json.load(f)
        except Exception as e:
            print(f"加载情景数据失败: {e}")
    
    # 加载网络数据
    network_data = None
    if network_file and os.path.exists(network_file):
        try:
            with open(network_file, 'r', encoding='utf-8') as f:
                network_data = json.load(f)
        except Exception as e:
            print(f"加载网络数据失败: {e}")
    
    # 查找可视化图片
    visualization_images = {}
    if visualization_dir and os.path.exists(visualization_dir):
        # 查找情景对比雷达图
        scenarios_comparison_path = os.path.join(visualization_dir, 'scenarios_comparison.png')
        if os.path.exists(scenarios_comparison_path):
            visualization_images['scenarios_comparison'] = os.path.relpath(
                scenarios_comparison_path, os.path.dirname(output_file))
        
        # 查找单个情景雷达图
        scenario_images = []
        for file in os.listdir(visualization_dir):
            if file.startswith('scenario_') and file.endswith('.png'):
                scenario_images.append({
                    'name': file[9:-4],  # 提取情景名称
                    'path': os.path.relpath(
                        os.path.join(visualization_dir, file), os.path.dirname(output_file))
                })
        visualization_images['scenario_images'] = scenario_images
        
        # 查找网络图
        network_path = os.path.join(visualization_dir, 'api_network.png')
        if os.path.exists(network_path):
            visualization_images['network'] = os.path.relpath(
                network_path, os.path.dirname(output_file))
        
        # 查找社区结构图
        communities_path = os.path.join(visualization_dir, 'api_communities.png')
        if os.path.exists(communities_path):
            visualization_images['communities'] = os.path.relpath(
                communities_path, os.path.dirname(output_file))
    
    # 生成HTML内容
    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <style>
            body {{
                font-family: 'Microsoft YaHei', Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f8f9fa;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            h1 {{
                text-align: center;
                padding: 20px 0;
                border-bottom: 1px solid #eee;
            }}
            .section {{
                margin: 30px 0;
                padding: 20px;
                background-color: #fff;
                border-radius: 5px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            .image-container {{
                margin: 20px 0;
                text-align: center;
            }}
            .image-container img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 5px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding: 20px;
                color: #777;
                font-size: 0.9em;
                border-top: 1px solid #eee;
            }}
            .scenario-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .scenario-card {{
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 15px;
                background-color: #fff;
            }}
            .scenario-card h3 {{
                margin-top: 0;
                color: #3498db;
            }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        
        <div class="section">
            <h2>报告概述</h2>
            <p>本报告生成于 {datetime.datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}</p>
            <p>本报告展示了API服务生态系统的情景分析结果，包括极端情景的需求变化和API协作网络结构。</p>
        </div>
    """
    
    # 添加情景分析部分
    if scenarios_data:
        html_content += f"""
        <div class="section">
            <h2>情景分析</h2>
            
            <h3>情景对比</h3>
        """
        
        # 添加情景对比雷达图
        if 'scenarios_comparison' in visualization_images:
            html_content += f"""
            <div class="image-container">
                <img src="{visualization_images['scenarios_comparison']}" alt="情景对比雷达图">
                <p>图1: API需求变化情景对比雷达图</p>
            </div>
            """
        
        # 添加情景表格
        html_content += """
            <h3>情景详情</h3>
            <table>
                <tr>
                    <th>情景名称</th>
                    <th>描述</th>
                    <th>概率</th>
                    <th>持续时间(月)</th>
                </tr>
        """
        
        for scenario in scenarios_data.get('scenarios', []):
            html_content += f"""
                <tr>
                    <td>{scenario.get('name', '未命名')}</td>
                    <td>{scenario.get('description', '无描述')}</td>
                    <td>{scenario.get('probability', 0)}</td>
                    <td>{scenario.get('duration', 0)}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h3>单个情景详情</h3>
            <div class="scenario-grid">
        """
        
        # 添加单个情景卡片
        for scenario in scenarios_data.get('scenarios', []):
            scenario_name = scenario.get('name', '未命名')
            scenario_image = next((img for img in visualization_images.get('scenario_images', []) 
                                  if img['name'] == scenario_name), None)
            
            html_content += f"""
                <div class="scenario-card">
                    <h3>{scenario_name}</h3>
                    <p>{scenario.get('description', '无描述')}</p>
                    <p>概率: {scenario.get('probability', 0)}</p>
                    <p>持续时间: {scenario.get('duration', 0)}个月</p>
                    <p>长期影响: {scenario.get('long_term_impacts', '无长期影响')}</p>
            """
            
            if scenario_image:
                html_content += f"""
                    <div class="image-container">
                        <img src="{scenario_image['path']}" alt="{scenario_name}雷达图">
                    </div>
                """
            
            html_content += """
                </div>
            """
        
        html_content += """
            </div>
        </div>
        """
    
    # 添加网络分析部分
    if network_data:
        html_content += """
        <div class="section">
            <h2>API协作网络分析</h2>
        """
        
        # 添加网络图
        if 'network' in visualization_images:
            html_content += f"""
            <div class="image-container">
                <img src="{visualization_images['network']}" alt="API协作网络图">
                <p>图2: API协作网络结构</p>
            </div>
            """
        
        # 添加社区结构图
        if 'communities' in visualization_images:
            html_content += f"""
            <div class="image-container">
                <img src="{visualization_images['communities']}" alt="API社区结构图">
                <p>图3: API社区结构分析</p>
            </div>
            """
        
        # 添加网络统计信息
        node_count = len(network_data.get('nodes', []))
        edge_count = len(network_data.get('edges', []))
        
        # 计算类别统计
        categories = {}
        for node in network_data.get('nodes', []):
            category = node.get('category')
            if category:
                categories[category] = categories.get(category, 0) + 1
        
        html_content += f"""
            <h3>网络统计信息</h3>
            <p>节点数量: {node_count}</p>
            <p>边数量: {edge_count}</p>
            
            <h3>API类别分布</h3>
            <table>
                <tr>
                    <th>类别</th>
                    <th>数量</th>
                    <th>占比</th>
                </tr>
        """
        
        for category, count in categories.items():
            percentage = round(count / node_count * 100, 2) if node_count > 0 else 0
            html_content += f"""
                <tr>
                    <td>{category}</td>
                    <td>{count}</td>
                    <td>{percentage}%</td>
                </tr>
            """
        
        html_content += """
            </table>
        </div>
        """
    
    # 添加页脚
    html_content += f"""
        <div class="footer">
            <p>API服务生态系统情景生成系统 &copy; {datetime.datetime.now().year}</p>
        </div>
    </body>
    </html>
    """
    
    # 写入HTML文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML报告已生成: {output_file}")
    return output_file

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='生成HTML报告')
    parser.add_argument('--scenarios_file', type=str, default=None,
                        help='情景数据文件路径')
    parser.add_argument('--network_file', type=str, default=None,
                        help='网络数据文件路径')
    parser.add_argument('--visualization_dir', type=str, default=None,
                        help='可视化结果目录')
    parser.add_argument('--output_file', type=str, default=None,
                        help='输出文件路径')
    parser.add_argument('--title', type=str, default="API服务生态系统情景分析报告",
                        help='报告标题')
    
    args = parser.parse_args()
    
    generate_html_report(
        scenarios_file=args.scenarios_file,
        network_file=args.network_file,
        visualization_dir=args.visualization_dir,
        output_file=args.output_file,
        title=args.title
    )

if __name__ == "__main__":
    main() 