#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
简化推荐模型 - 使用scikit-learn实现
作为YouTubeDNN的替代方案，避免TensorFlow版本兼容性问题
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import pickle

# 将项目根目录添加到Python路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class SimpleRecommendationModel:
    def __init__(self, embedding_dim=32):
        self.embedding_dim = embedding_dim
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.svd_model = TruncatedSVD(n_components=embedding_dim, random_state=42)
        self.user_item_matrix = None
        self.user_embeddings = None
        self.item_embeddings = None
        
    def load_and_preprocess_data(self):
        """加载和预处理数据"""
        # 加载数据
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        data_path = os.path.join(project_root, "data", "read_history.csv")
        data = pd.read_csv(data_path)
        
        print(f"加载数据：{len(data)} 条记录")
        
        # 处理timestamp列
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.sort_values('timestamp')
        
        # 编码用户和物品ID
        data['user_id_encoded'] = self.user_encoder.fit_transform(data['user_id'])
        data['item_id_encoded'] = self.item_encoder.fit_transform(data['item_id'])
        
        print(f"用户数：{data['user_id_encoded'].nunique()}")
        print(f"物品数：{data['item_id_encoded'].nunique()}")
        
        return data
    
    def create_user_item_matrix(self, data):
        """创建用户-物品交互矩阵"""
        # 使用rating作为交互强度，如果没有rating则使用1
        if 'rating' in data.columns:
            interaction_strength = data['rating']
        else:
            interaction_strength = 1
            
        # 创建用户-物品矩阵
        user_item_df = data.groupby(['user_id_encoded', 'item_id_encoded'])['rating'].mean().reset_index()
        
        # 转换为矩阵格式
        n_users = data['user_id_encoded'].nunique()
        n_items = data['item_id_encoded'].nunique()
        
        user_item_matrix = np.zeros((n_users, n_items))
        
        for _, row in user_item_df.iterrows():
            user_item_matrix[int(row['user_id_encoded']), int(row['item_id_encoded'])] = row['rating']
        
        self.user_item_matrix = user_item_matrix
        print(f"创建用户-物品矩阵：{user_item_matrix.shape}")
        return user_item_matrix
    
    def train_embeddings(self):
        """训练用户和物品的嵌入向量"""
        print("训练SVD模型...")
        
        # 使用SVD分解获取低维表示
        user_item_svd = self.svd_model.fit_transform(self.user_item_matrix)
        item_user_svd = self.svd_model.components_.T
        
        self.user_embeddings = user_item_svd
        self.item_embeddings = item_user_svd
        
        print(f"用户嵌入向量形状：{self.user_embeddings.shape}")
        print(f"物品嵌入向量形状：{self.item_embeddings.shape}")
        
    def get_recommendations(self, user_id, top_k=10):
        """为用户获取推荐"""
        try:
            # 编码用户ID
            user_encoded = self.user_encoder.transform([user_id])[0]
            
            # 获取用户嵌入向量
            user_embedding = self.user_embeddings[user_encoded].reshape(1, -1)
            
            # 计算与所有物品的相似度
            similarities = cosine_similarity(user_embedding, self.item_embeddings)[0]
            
            # 获取用户已交互的物品（排除已交互物品）
            interacted_items = np.where(self.user_item_matrix[user_encoded] > 0)[0]
            
            # 将已交互物品的相似度设为-1（排除）
            similarities[interacted_items] = -1
            
            # 获取top_k推荐
            top_items_encoded = np.argsort(similarities)[::-1][:top_k]
            
            # 解码物品ID
            top_items = self.item_encoder.inverse_transform(top_items_encoded)
            top_scores = similarities[top_items_encoded]
            
            return list(zip(top_items, top_scores))
            
        except Exception as e:
            print(f"推荐过程出错：{e}")
            return []
    
    def evaluate_model(self, data, test_ratio=0.2):
        """评估模型性能"""
        print("评估模型...")
        
        # 简单的时间分割
        data_sorted = data.sort_values('timestamp')
        split_idx = int(len(data_sorted) * (1 - test_ratio))
        
        train_data = data_sorted.iloc[:split_idx]
        test_data = data_sorted.iloc[split_idx:]
        
        # 评估（不重新训练，使用完整数据训练的模型）
        hits = 0
        total_users = 0
        
        test_users = test_data['user_id'].unique()[:100]  # 限制测试用户数量以提高速度
        
        for user_id in tqdm(test_users, desc="评估中"):
            # 获取用户在测试集中的真实交互
            user_test_items = test_data[test_data['user_id'] == user_id]['item_id'].values
            
            if len(user_test_items) == 0:
                continue
                
            # 获取推荐
            recommendations = self.get_recommendations(user_id, top_k=10)
            
            if recommendations:
                recommended_items = [item for item, score in recommendations]
                
                # 检查是否命中
                if any(item in recommended_items for item in user_test_items):
                    hits += 1
                    
                total_users += 1
        
        hit_rate = hits / total_users if total_users > 0 else 0
        print(f"命中率 (Hit Rate): {hit_rate:.3f}")
        
        return hit_rate
    
    def save_model(self, path="simple_rec_model.pkl"):
        """保存模型"""
        model_data = {
            'user_encoder': self.user_encoder,
            'item_encoder': self.item_encoder,
            'svd_model': self.svd_model,
            'user_embeddings': self.user_embeddings,
            'item_embeddings': self.item_embeddings,
            'user_item_matrix': self.user_item_matrix
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"模型已保存到：{path}")
    
    def load_model(self, path="simple_rec_model.pkl"):
        """加载模型"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.user_encoder = model_data['user_encoder']
        self.item_encoder = model_data['item_encoder']
        self.svd_model = model_data['svd_model']
        self.user_embeddings = model_data['user_embeddings']
        self.item_embeddings = model_data['item_embeddings']
        self.user_item_matrix = model_data['user_item_matrix']
        
        print(f"模型已从 {path} 加载")
    
    def run_training(self):
        """完整的训练流程"""
        print("=== 简化推荐模型训练开始 ===")
        
        # 1. 加载数据
        data = self.load_and_preprocess_data()
        
        # 2. 创建用户-物品矩阵
        self.create_user_item_matrix(data)
        
        # 3. 训练嵌入向量
        self.train_embeddings()
        
        # 4. 评估模型
        hit_rate = self.evaluate_model(data)
        
        # 5. 保存模型
        self.save_model()
        
        print("=== 训练完成 ===")
        print(f"最终命中率：{hit_rate:.3f}")
        
        # 6. 演示推荐
        sample_users = data['user_id'].unique()[:5]
        print("\n=== 推荐示例 ===")
        for user_id in sample_users:
            recommendations = self.get_recommendations(user_id, top_k=5)
            print(f"用户 {user_id} 的推荐：")
            for item, score in recommendations:
                print(f"  物品 {item}: 分数 {score:.3f}")
            print()

if __name__ == '__main__':
    model = SimpleRecommendationModel()
    model.run_training() 