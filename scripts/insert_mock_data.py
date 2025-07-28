#!/usr/bin/env python 
# -*- coding: utf-8 -*-
import sys
import os
import csv
import datetime
import random
from typing import List, Dict

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dao.mongo_db import MongoDB


class MockDataInserter(object):
    def __init__(self):
        """
        初始化MongoDB连接和各个collection
        """
        self.mongo = MongoDB(db='recommendation')
        self.db_client = self.mongo.db_client
        self.read_collection = self.db_client['read']
        self.likes_collection = self.db_client['likes']
        self.collection_collection = self.db_client['collection']
        self.content_labels_collection = self.db_client['content_labels']

    def clear_collections(self):
        """
        清空所有相关的collection，方便重新插入数据
        """
        print("清空现有数据...")
        self.read_collection.delete_many({})
        self.likes_collection.delete_many({})
        self.collection_collection.delete_many({})
        self.content_labels_collection.delete_many({})
        print("数据清空完成")

    def read_csv_data(self, csv_file_path: str) -> List[Dict]:
        """
        读取CSV文件并解析数据
        """
        data = []
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if len(row) == 3:
                    user_id, score, content_id = row
                    data.append({
                        'user_id': int(user_id),
                        'score': int(score),
                        'content_id': content_id
                    })
        return data

    def generate_behavior_records(self, data: List[Dict]):
        """
        根据评分生成不同的用户行为记录
        评分规则：
        - 阅读: 1分
        - 点赞: 2分  
        - 收藏: 2分
        - 如果同时存在2项: 额外加1分
        - 如果同时存在3项: 额外加2分
        """
        read_records = []
        likes_records = []
        collection_records = []
        
        for item in data:
            user_id = item['user_id']
            content_id = item['content_id']
            score = item['score']
            
            # 生成随机时间戳（最近30天内）
            base_time = datetime.datetime.now()
            random_days = random.randint(0, 30)
            random_hours = random.randint(0, 23)
            random_minutes = random.randint(0, 59)
            timestamp = base_time - datetime.timedelta(
                days=random_days, 
                hours=random_hours, 
                minutes=random_minutes
            )
            
            # 基础记录模板
            base_record = {
                'user_id': user_id,
                'content_id': content_id,
                'timestamp': timestamp,
                'create_time': timestamp
            }
            
            # 根据评分决定生成哪些行为记录
            if score >= 2:
                # 评分2分以上，至少有阅读记录
                read_records.append(base_record.copy())
                
                if score >= 3:
                    # 评分3分以上，增加点赞概率
                    if random.random() < 0.7:  # 70%概率点赞
                        likes_records.append(base_record.copy())
                
                if score >= 4:
                    # 评分4分以上，增加收藏概率
                    if random.random() < 0.5:  # 50%概率收藏
                        collection_records.append(base_record.copy())
                        
                if score == 5:
                    # 评分5分，高概率三种行为都有
                    if random.random() < 0.9:  # 90%概率点赞
                        likes_record = base_record.copy()
                        if likes_record not in likes_records:
                            likes_records.append(likes_record)
                    
                    if random.random() < 0.7:  # 70%概率收藏
                        collection_record = base_record.copy()
                        if collection_record not in collection_records:
                            collection_records.append(collection_record)
        
        return read_records, likes_records, collection_records

    def generate_content_labels(self, data: List[Dict]):
        """
        生成内容标签数据
        """
        content_labels = []
        content_ids = list(set([item['content_id'] for item in data]))
        
        # 预定义的新闻类别和标签
        categories = ['科技', '财经', '体育', '娱乐', '政治', '社会', '国际', '军事']
        tags_pool = [
            ['人工智能', '科技创新', '互联网'], 
            ['股市', '经济', '投资', '金融'],
            ['足球', '篮球', '奥运会', '体育赛事'],
            ['明星', '电影', '音乐', '综艺'],
            ['政策', '改革', '民生', '社会治理'],
            ['社会新闻', '民生关注', '突发事件'],
            ['国际关系', '外交', '全球化'],
            ['军事', '国防', '安全']
        ]
        
        for content_id in content_ids:
            category_idx = random.randint(0, len(categories) - 1)
            category = categories[category_idx]
            tags = random.sample(tags_pool[category_idx], random.randint(1, 3))
            
            content_label = {
                'content_id': content_id,
                'title': f'新闻标题_{content_id}',
                'category': category,
                'tags': tags,
                'create_time': datetime.datetime.now(),
                'status': 'published'
            }
            content_labels.append(content_label)
        
        return content_labels

    def insert_data_to_mongodb(self, csv_file_path: str):
        """
        主要的数据插入方法
        """
        print("开始读取CSV数据...")
        data = self.read_csv_data(csv_file_path)
        print(f"读取到 {len(data)} 条记录")
        
        print("生成用户行为记录...")
        read_records, likes_records, collection_records = self.generate_behavior_records(data)
        
        print("生成内容标签数据...")
        content_labels = self.generate_content_labels(data)
        
        # 插入数据到各个collection
        if read_records:
            print(f"插入 {len(read_records)} 条阅读记录...")
            self.read_collection.insert_many(read_records)
            
        if likes_records:
            print(f"插入 {len(likes_records)} 条点赞记录...")
            self.likes_collection.insert_many(likes_records)
            
        if collection_records:
            print(f"插入 {len(collection_records)} 条收藏记录...")
            self.collection_collection.insert_many(collection_records)
            
        if content_labels:
            print(f"插入 {len(content_labels)} 条内容标签...")
            self.content_labels_collection.insert_many(content_labels)
        
        print("数据插入完成！")

    def verify_data(self):
        """
        验证插入的数据
        """
        print("\n=== 数据验证 ===")
        print(f"阅读记录数量: {self.read_collection.count_documents({})}")
        print(f"点赞记录数量: {self.likes_collection.count_documents({})}")
        print(f"收藏记录数量: {self.collection_collection.count_documents({})}")
        print(f"内容标签数量: {self.content_labels_collection.count_documents({})}")
        
        # 显示一些示例数据
        print("\n=== 示例数据 ===")
        sample_like = self.likes_collection.find_one()
        if sample_like:
            print("点赞记录示例:", sample_like)
            
        sample_content = self.content_labels_collection.find_one()
        if sample_content:
            print("内容标签示例:", sample_content)


def main():
    inserter = MockDataInserter()
    
    # 询问是否清空现有数据
    clear_choice = input("是否清空现有数据？(y/n): ").lower().strip()
    if clear_choice == 'y':
        inserter.clear_collections()
    
    # 插入数据
    csv_file_path = 'data/news_score/news_log.csv'
    inserter.insert_data_to_mongodb(csv_file_path)
    
    # 验证数据
    inserter.verify_data()


if __name__ == '__main__':
    main() 