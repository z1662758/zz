"""
API服务生态系统情景生成 - 数据加载工具
从SQL文件中提取数据，并将其转换为适合项目使用的格式（json）
"""

import os
import re
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_default_logger


class DataLoader:
    """数据加载工具，从SQL文件中提取数据并转换格式"""
    
    def __init__(self, logger=None):
        """
        初始化数据加载工具
        
        Args:
            logger: 日志记录器
        """
        self.logger = logger or get_default_logger()
        self.api_data = {}
        self.mashup_data = {}
        self.relationship_data = {
            "call_relationships": [],
            "similarity_relationships": []
        }
        
    def load_from_sql(self, api_sql_file: str, dataset_sql_file: str) -> bool:
        """
        从SQL文件中加载数据
        
        Args:
            api_sql_file: API数据SQL文件路径
            dataset_sql_file: 数据集SQL文件路径
            
        Returns:
            是否成功加载
        """
        try:
            self.logger.info(f"开始从SQL文件加载数据: {api_sql_file}, {dataset_sql_file}")
            
            # 加载API数据
            success_api = self._extract_api_data(api_sql_file)
            if not success_api:
                self.logger.error("提取API数据失败")
                return False
                
            # 加载关系数据
            success_rel = self._extract_relationship_data(dataset_sql_file)
            if not success_rel:
                self.logger.error("提取关系数据失败")
                return False
                
            self.logger.info(f"数据加载成功，共加载{len(self.api_data)}个API和{len(self.relationship_data['call_relationships'])}个调用关系")
            return True
            
        except Exception as e:
            self.logger.error(f"加载数据失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
            
    def _extract_api_data(self, sql_file: str) -> bool:
        """
        从SQL文件中提取API数据
        
        Args:
            sql_file: SQL文件路径
            
        Returns:
            是否成功提取
        """
        try:
            self.logger.info(f"开始提取API数据: {sql_file}")
            
            # 读取SQL文件内容
            with open(sql_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # 提取INSERT语句
            pattern = r"INSERT INTO `apidata` VALUES \((.*?)\);"
            matches = re.findall(pattern, content, re.DOTALL)
            
            if not matches:
                self.logger.warning("未找到API数据插入语句")
                return False
                
            # 解析INSERT语句
            for i, match in enumerate(matches):
                try:
                    # 分割字段值，考虑引号内的逗号
                    values = []
                    in_quote = False
                    current_value = ""
                    
                    for char in match:
                        if char == "'" and (len(current_value) == 0 or current_value[-1] != '\\'):
                            in_quote = not in_quote
                        elif char == ',' and not in_quote:
                            values.append(current_value.strip("'"))
                            current_value = ""
                        else:
                            current_value += char
                            
                    if current_value:
                        values.append(current_value.strip("'"))
                    
                    # 提取字段值
                    if len(values) >= 6:
                        api_id = f"api_{i}"
                        name = values[0]
                        description = values[1]
                        follower_num = values[2]
                        endpoint = values[3]
                        homepage = values[4]
                        category = values[5]
                        
                        # 存储API数据
                        self.api_data[api_id] = {
                            "Name": name,  # 使用大写首字母，与SocialAgent保持一致
                            "Description": description,
                            "FollowerNum": follower_num,
                            "Endpoint": endpoint,
                            "Homepage": homepage,
                            "Category": category
                        }
                except Exception as e:
                    self.logger.warning(f"解析API数据失败: {str(e)}")
                    continue
                    
            self.logger.info(f"API数据提取完成，共提取{len(self.api_data)}个API")
            return True
            
        except Exception as e:
            self.logger.error(f"提取API数据失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
            
    def _extract_relationship_data(self, sql_file: str) -> bool:
        """
        从SQL文件中提取关系数据
        
        Args:
            sql_file: SQL文件路径
            
        Returns:
            是否成功提取
        """
        try:
            self.logger.info(f"开始提取关系数据: {sql_file}")
            
            # 读取SQL文件内容
            try:
                with open(sql_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except FileNotFoundError:
                self.logger.error(f"文件不存在: {sql_file}")
                return False
                
            # 提取m-a_edges表的INSERT语句
            pattern = r"INSERT INTO `m-a_edges` VALUES \((.*?)\);"
            matches = re.findall(pattern, content, re.DOTALL)
            
            if not matches:
                # 尝试其他可能的表名或格式
                pattern = r"INSERT INTO `m-a_edges`.*? VALUES \((.*?)\);"
                matches = re.findall(pattern, content, re.DOTALL)
                
                if not matches:
                    self.logger.warning("未找到关系数据插入语句，尝试使用替代方法")
                    
                    # 如果无法从SQL文件中提取关系数据，则生成一些模拟的关系数据
                    self._generate_mock_relationships()
                    return True
                
            # 解析INSERT语句
            for match in matches:
                try:
                    # 分割字段值，考虑引号内的逗号
                    values = []
                    in_quote = False
                    current_value = ""
                    
                    for char in match:
                        if char == "'" and (len(current_value) == 0 or current_value[-1] != '\\'):
                            in_quote = not in_quote
                        elif char == ',' and not in_quote:
                            values.append(current_value.strip("'"))
                            current_value = ""
                        else:
                            current_value += char
                            
                    if current_value:
                        values.append(current_value.strip("'"))
                    
                    # 提取字段值
                    if len(values) >= 3:
                        source = f"api_{int(values[0])}" if values[0].isdigit() else values[0]
                        target = f"api_{int(values[1])}" if values[1].isdigit() else values[1]
                        weight = float(values[2]) if values[2] else 0.5  # 默认权重0.5
                        
                        # 确保source和target在api_data中存在
                        if source in self.api_data and target in self.api_data:
                            # 存储调用关系
                            self.relationship_data["call_relationships"].append({
                                "source": source,
                                "target": target,
                                "weight": weight
                            })
                except Exception as e:
                    self.logger.warning(f"解析关系数据失败: {str(e)}")
                    continue
                    
            # 计算API之间的相似性关系
            self._calculate_similarity_relationships()
                    
            self.logger.info(f"关系数据提取完成，共提取{len(self.relationship_data['call_relationships'])}个调用关系，{len(self.relationship_data['similarity_relationships'])}个相似性关系")
            return True
            
        except Exception as e:
            self.logger.error(f"提取关系数据失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _generate_mock_relationships(self) -> None:
        """
        生成模拟关系数据的方法已被禁用
        抛出异常提示用户必须提供有效的SQL数据文件
        """
        self.logger.error("无法从SQL文件中提取关系数据")
        raise ValueError("必须提供有效的SQL数据文件，模拟数据生成功能已被禁用。请确保提供了programmableweb_2022(1).sql和programmableweb_dataset(1).sql文件。")
            
    def _calculate_similarity_relationships(self) -> None:
        """
        计算API之间的相似性关系
        基于类别和描述的相似性
        """
        try:
            self.logger.info("开始计算API之间的相似性关系")
            
            # 按类别分组API
            category_apis = {}
            for api_id, api_info in self.api_data.items():
                category = api_info.get("Category", "")
                if category not in category_apis:
                    category_apis[category] = []
                category_apis[category].append(api_id)
                
            # 对于同一类别的API，计算相似性
            for category, apis in category_apis.items():
                if len(apis) <= 1:
                    continue
                    
                for i in range(len(apis)):
                    for j in range(i+1, len(apis)):
                        api1_id = apis[i]
                        api2_id = apis[j]
                        
                        # 提取两个API的描述和名称
                        api1_desc = self.api_data.get(api1_id, {}).get("Description", "")
                        api2_desc = self.api_data.get(api2_id, {}).get("Description", "")
                        api1_name = self.api_data.get(api1_id, {}).get("Name", "")
                        api2_name = self.api_data.get(api2_id, {}).get("Name", "")
                        
                        # 计算描述相似度 (基于共同词汇的简单相似度)
                        api1_words = set(api1_desc.lower().split())
                        api2_words = set(api2_desc.lower().split())
                        
                        # 避免零分母
                        if len(api1_words) == 0 or len(api2_words) == 0:
                            desc_similarity = 0.0
                        else:
                            common_words = api1_words.intersection(api2_words)
                            desc_similarity = len(common_words) / max(1, min(len(api1_words), len(api2_words)))
                        
                        # 计算名称相似度
                        api1_name_words = set(api1_name.lower().split())
                        api2_name_words = set(api2_name.lower().split())
                        
                        # 避免零分母
                        if len(api1_name_words) == 0 or len(api2_name_words) == 0:
                            name_similarity = 0.0
                        else:
                            common_name_words = api1_name_words.intersection(api2_name_words)
                            name_similarity = len(common_name_words) / max(1, min(len(api1_name_words), len(api2_name_words)))
                        
                        # 综合相似度 (名称相似度权重更高)
                        similarity = 0.3 + (0.4 * desc_similarity) + (0.3 * name_similarity)
                        
                        # 同类别加成
                        similarity = min(0.9, similarity)
                        
                        # 添加到相似性关系
                        self.relationship_data["similarity_relationships"].append({
                            "source": api1_id,
                            "target": api2_id,
                            "weight": similarity
                        })
                        
            self.logger.info(f"相似性关系计算完成，共计算{len(self.relationship_data['similarity_relationships'])}个相似性关系")
            
        except Exception as e:
            self.logger.error(f"计算相似性关系失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
    def save_processed_data(self, output_dir: str) -> bool:
        """
        保存处理后的数据为JSON文件
        
        Args:
            output_dir: 输出目录
            
        Returns:
            是否成功保存
        """
        try:
            self.logger.info(f"开始保存处理后的数据: {output_dir}")
            
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存API数据，将处理后的 API 数据以 JSON 格式保存到指定文件
            api_data_file = os.path.join(output_dir, "api_data.json")
            with open(api_data_file, "w", encoding="utf-8") as f:
                json.dump(self.api_data, f, ensure_ascii=False, indent=2)
                
            # 保存关系数据
            relationship_data_file = os.path.join(output_dir, "relationship_data.json")
            with open(relationship_data_file, "w", encoding="utf-8") as f:
                json.dump(self.relationship_data, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"处理后的数据已保存到: {output_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存处理后的数据失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
            
    def get_api_data(self) -> Dict[str, Any]:
        """
        获取API数据
        
        Returns:
            API数据字典
        """
        return self.api_data
        
    def get_relationship_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        获取关系数据
        
        Returns:
            关系数据字典
        """
        return self.relationship_data
        
    def get_api_count(self) -> int:
        """
        获取API数量
        
        Returns:
            API数量
        """
        return len(self.api_data)
        
    def get_api_categories(self) -> List[str]:
        """
        获取API类别列表
        
        Returns:
            API类别列表
        """
        categories = set()
        for api_info in self.api_data.values():
            category = api_info.get("Category", "")
            if category:
                categories.add(category)
        return list(categories)


if __name__ == "__main__":
    # 示例用法
    logger = get_default_logger()
    loader = DataLoader(logger=logger)
    
    # 设置SQL文件路径
    api_sql_file = "../programmableweb_2022.sql"
    dataset_sql_file = "../programmableweb_dataset.sql"
    
    # 加载数据
    success = loader.load_from_sql(api_sql_file, dataset_sql_file)
    if success:
        # 保存处理后的数据
        output_dir = "../data/processed"
        loader.save_processed_data(output_dir)
        
        # 输出统计信息
        api_count = loader.get_api_count()
        categories = loader.get_api_categories()
        relationship_data = loader.get_relationship_data()
        
        logger.info(f"API数量: {api_count}")
        logger.info(f"API类别: {categories}")
        logger.info(f"调用关系数量: {len(relationship_data['call_relationships'])}")
        logger.info(f"相似性关系数量: {len(relationship_data['similarity_relationships'])}")
    else:
        logger.error("数据加载失败") 