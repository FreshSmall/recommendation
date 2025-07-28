#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
简化版数据插入脚本
专门用于将news_log.csv的数据快速插入到MongoDB的likes collection中
"""
import sys
import os
import csv
import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dao.mongo_db import MongoDB


class LikesDataInserter(object):
    def __init__(self):
        """
        初始化MongoDB连接
        """
        self.mongo = MongoDB(db='recommendation')
        self.db_client = self.mongo.db_client
        self.likes_collection = self.db_client['likes']

    def clear_likes_collection(self):
        """
        清空likes collection
        """
        print("清空现有likes数据...")
        result = self.likes_collection.delete_many({})
        print(f"删除了 {result.deleted_count} 条记录")

    def insert_likes_from_csv(self, csv_file_path: str):
        """
        从CSV文件读取数据并插入到likes collection
        将评分大于等于3的记录作为likes记录插入
        """
        likes_records = []
        
        print("开始读取CSV文件...")
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if len(row) == 3:
                    user_id, score, content_id = row
                    score = int(score)
                    
                    # 只有评分大于等于3的才认为是"喜欢"
                    if score >= 3:
                        like_record = {
                            'user_id': int(user_id),
                            'content_id': content_id,
                            'score': score,
                            'timestamp': datetime.datetime.now(),
                            'create_time': datetime.datetime.now()
                        }
                        likes_records.append(like_record)
        
        print(f"准备插入 {len(likes_records)} 条likes记录...")
        
        if likes_records:
            self.likes_collection.insert_many(likes_records)
            print("数据插入完成！")
        else:
            print("没有符合条件的数据需要插入")

    def verify_data(self):
        """
        验证插入的数据
        """
        total_count = self.likes_collection.count_documents({})
        print(f"\n总likes记录数量: {total_count}")
        
        # 按用户统计
        pipeline = [
            {"$group": {"_id": "$user_id", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]
        user_stats = list(self.likes_collection.aggregate(pipeline))
        print("\n用户点赞统计（前10）:")
        for stat in user_stats:
            print(f"  用户 {stat['_id']}: {stat['count']} 次点赞")
        
        # 按内容统计
        pipeline = [
            {"$group": {"_id": "$content_id", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]
        content_stats = list(self.likes_collection.aggregate(pipeline))
        print("\n内容点赞统计（前10）:")
        for stat in content_stats:
            print(f"  内容 {stat['_id']}: {stat['count']} 次点赞")
        
        # 显示示例数据
        sample = self.likes_collection.find_one()
        if sample:
            print(f"\n示例记录: {sample}")


def main():
    inserter = LikesDataInserter()
    
    # 询问是否清空现有数据
    clear_choice = input("是否清空现有likes数据？(y/n): ").lower().strip()
    if clear_choice == 'y':
        inserter.clear_likes_collection()
    
    # 插入数据
    csv_file_path = 'data/news_score/news_log.csv'
    if not os.path.exists(csv_file_path):
        print(f"错误：找不到文件 {csv_file_path}")
        return
    
    inserter.insert_likes_from_csv(csv_file_path)
    
    # 验证数据
    inserter.verify_data()


if __name__ == '__main__':
    main() 