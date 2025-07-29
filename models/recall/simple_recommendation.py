#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
简化推荐模型 - 使用scikit-learn实现
作为YouTubeDNN的替代方案，避免TensorFlow版本兼容性问题

核心思想：
1. 使用SVD（奇异值分解）将高维稀疏的用户-物品交互矩阵分解为低维稠密向量
2. 通过余弦相似度计算用户与物品的相似性
3. 基于相似性生成个性化推荐列表
"""

# 系统库导入
import sys  # 系统相关功能
import os   # 操作系统接口
import pandas as pd  # 数据处理和分析库
import numpy as np   # 数值计算库
from sklearn.preprocessing import LabelEncoder  # 标签编码器，将分类变量转换为数值
from sklearn.decomposition import TruncatedSVD  # 截断SVD，用于矩阵分解
from sklearn.metrics.pairwise import cosine_similarity  # 余弦相似度计算
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF向量化（当前未使用但保留）
from tqdm import tqdm  # 进度条显示库
import pickle  # 序列化库，用于模型保存和加载

# 将项目根目录添加到Python路径中，确保可以导入项目内的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class SimpleRecommendationModel:
    """
    简化推荐模型类
    基于矩阵分解的协同过滤推荐算法
    """
    def __init__(self, embedding_dim=32):
        """
        初始化推荐模型
        
        Args:
            embedding_dim (int): 嵌入向量的维度，决定特征压缩后的维度大小
        """
        self.embedding_dim = embedding_dim  # 嵌入向量维度
        self.user_encoder = LabelEncoder()  # 用户ID编码器，将用户ID转换为连续整数
        self.item_encoder = LabelEncoder()  # 物品ID编码器，将物品ID转换为连续整数
        self.svd_model = TruncatedSVD(n_components=embedding_dim, random_state=42)  # SVD模型，用于矩阵分解
        self.user_item_matrix = None  # 用户-物品交互矩阵
        self.user_embeddings = None   # 用户嵌入向量矩阵
        self.item_embeddings = None   # 物品嵌入向量矩阵
        
    def load_and_preprocess_data(self):
        """
        加载和预处理数据
        
        Returns:
            pd.DataFrame: 预处理后的数据
        """
        # 构建数据文件的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在目录
        project_root = os.path.dirname(os.path.dirname(current_dir))  # 获取项目根目录
        data_path = os.path.join(project_root, "data", "read_history.csv")  # 构建数据文件路径
        data = pd.read_csv(data_path)  # 读取CSV数据文件
        
        print(f"加载数据：{len(data)} 条记录")  # 输出数据记录数
        
        # 处理时间戳列，确保数据按时间顺序排列
        data['timestamp'] = pd.to_datetime(data['timestamp'])  # 将字符串时间转换为datetime对象
        data = data.sort_values('timestamp')  # 按时间戳排序，用于后续的时间分割
        
        # 对用户ID和物品ID进行标签编码，将字符串ID转换为连续的整数ID
        data['user_id_encoded'] = self.user_encoder.fit_transform(data['user_id'])  # 编码用户ID
        data['item_id_encoded'] = self.item_encoder.fit_transform(data['item_id'])  # 编码物品ID
        
        # 输出基本统计信息
        print(f"用户数：{data['user_id_encoded'].nunique()}")  # 唯一用户数
        print(f"物品数：{data['item_id_encoded'].nunique()}")  # 唯一物品数
        
        return data
    
    def create_user_item_matrix(self, data):
        """
        创建用户-物品交互矩阵
        矩阵的行表示用户，列表示物品，值表示交互强度（评分）
        
        Args:
            data (pd.DataFrame): 包含用户-物品交互数据的DataFrame
            
        Returns:
            np.ndarray: 用户-物品交互矩阵
        """
        # 检查是否存在评分列，如果没有则使用默认值1表示有交互
        if 'rating' in data.columns:
            interaction_strength = data['rating']  # 使用实际评分作为交互强度
        else:
            interaction_strength = 1  # 如果没有评分，使用1表示存在交互
            
        # 对同一用户-物品对的多次交互取平均值，避免重复交互的影响
        user_item_df = data.groupby(['user_id_encoded', 'item_id_encoded'])['rating'].mean().reset_index()
        
        # 获取用户数和物品数，确定矩阵维度
        n_users = data['user_id_encoded'].nunique()  # 用户总数
        n_items = data['item_id_encoded'].nunique()  # 物品总数
        
        # 初始化用户-物品交互矩阵，默认值为0（表示无交互）
        user_item_matrix = np.zeros((n_users, n_items))
        
        # 填充交互矩阵，将用户-物品的交互记录填入对应位置
        for _, row in user_item_df.iterrows():
            user_idx = int(row['user_id_encoded'])  # 用户索引
            item_idx = int(row['item_id_encoded'])  # 物品索引
            rating = row['rating']  # 交互强度（评分）
            user_item_matrix[user_idx, item_idx] = rating  # 填入矩阵
        
        self.user_item_matrix = user_item_matrix  # 保存交互矩阵
        print(f"创建用户-物品矩阵：{user_item_matrix.shape}")  # 输出矩阵形状
        return user_item_matrix
    
    def train_embeddings(self):
        """
        训练用户和物品的嵌入向量
        使用SVD（奇异值分解）将高维稀疏矩阵分解为低维稠密向量
        """
        print("训练SVD模型...")
        
        # 使用SVD分解用户-物品矩阵
        # fit_transform返回用户在低维空间的表示
        user_item_svd = self.svd_model.fit_transform(self.user_item_matrix)
        
        # components_是物品在低维空间的表示，需要转置以获得正确的形状
        item_user_svd = self.svd_model.components_.T  # 转置得到 (n_items, embedding_dim)
        
        # 保存用户和物品的嵌入向量
        self.user_embeddings = user_item_svd   # 用户嵌入向量 (n_users, embedding_dim)
        self.item_embeddings = item_user_svd   # 物品嵌入向量 (n_items, embedding_dim)
        
        # 输出嵌入向量的形状信息
        print(f"用户嵌入向量形状：{self.user_embeddings.shape}")
        print(f"物品嵌入向量形状：{self.item_embeddings.shape}")
        
    def get_recommendations(self, user_id, top_k=10):
        """
        为指定用户生成推荐列表
        
        Args:
            user_id: 原始用户ID
            top_k (int): 推荐物品的数量
            
        Returns:
            list: 推荐物品列表，每个元素为(物品ID, 相似度分数)的元组
        """
        try:
            # 将原始用户ID编码为内部索引
            user_encoded = self.user_encoder.transform([user_id])[0]
            
            # 获取该用户的嵌入向量，并调整形状为(1, embedding_dim)
            user_embedding = self.user_embeddings[user_encoded].reshape(1, -1)
            
            # 计算用户嵌入向量与所有物品嵌入向量的余弦相似度
            similarities = cosine_similarity(user_embedding, self.item_embeddings)[0]
            
            # 获取用户已经交互过的物品索引（评分大于0的物品）
            interacted_items = np.where(self.user_item_matrix[user_encoded] > 0)[0]
            
            # 将已交互物品的相似度设为-1，确保不会被推荐（排除已知喜好）
            similarities[interacted_items] = -1
            
            # 获取相似度最高的top_k个物品的索引
            # argsort()返回从小到大的索引，[::-1]反转为从大到小，[:top_k]取前k个
            top_items_encoded = np.argsort(similarities)[::-1][:top_k]
            
            # 将编码后的物品索引转换回原始物品ID
            top_items = self.item_encoder.inverse_transform(top_items_encoded)
            # 获取对应的相似度分数
            top_scores = similarities[top_items_encoded]
            
            # 返回(物品ID, 分数)的元组列表
            return list(zip(top_items, top_scores))
            
        except Exception as e:
            print(f"推荐过程出错：{e}")
            return []  # 出错时返回空列表
    
    def evaluate_model(self, data, test_ratio=0.2):
        """
        评估模型性能
        使用时间分割方法，将最新的20%数据作为测试集
        
        Args:
            data (pd.DataFrame): 完整数据集
            test_ratio (float): 测试集比例
            
        Returns:
            float: 命中率（Hit Rate）
        """
        print("评估模型...")
        
        # 按时间顺序分割数据，模拟真实的推荐场景
        data_sorted = data.sort_values('timestamp')  # 确保按时间排序
        split_idx = int(len(data_sorted) * (1 - test_ratio))  # 计算分割点
        
        train_data = data_sorted.iloc[:split_idx]    # 前80%作为训练数据
        test_data = data_sorted.iloc[split_idx:]     # 后20%作为测试数据
        
        # 使用完整数据训练的模型进行评估（不重新训练）
        hits = 0        # 命中次数
        total_users = 0 # 总测试用户数
        
        # 限制测试用户数量以提高评估速度
        test_users = test_data['user_id'].unique()[:100]
        
        # 为每个测试用户评估推荐效果
        for user_id in tqdm(test_users, desc="评估中"):
            # 获取该用户在测试期间实际交互的物品
            user_test_items = test_data[test_data['user_id'] == user_id]['item_id'].values
            
            # 如果用户在测试期间没有交互，跳过
            if len(user_test_items) == 0:
                continue
                
            # 为用户生成推荐列表
            recommendations = self.get_recommendations(user_id, top_k=10)
            
            if recommendations:
                # 提取推荐的物品ID
                recommended_items = [item for item, score in recommendations]
                
                # 检查是否命中：推荐列表中是否包含用户实际交互的物品
                if any(item in recommended_items for item in user_test_items):
                    hits += 1  # 命中计数
                    
                total_users += 1  # 总用户计数
        
        # 计算命中率：命中用户数 / 总测试用户数
        hit_rate = hits / total_users if total_users > 0 else 0
        print(f"命中率 (Hit Rate): {hit_rate:.3f}")
        
        return hit_rate
    
    def save_model(self, path="simple_rec_model.pkl"):
        """
        保存训练好的模型到文件
        
        Args:
            path (str): 模型保存路径
        """
        # 将模型的所有重要组件打包成字典
        model_data = {
            'user_encoder': self.user_encoder,      # 用户ID编码器
            'item_encoder': self.item_encoder,      # 物品ID编码器
            'svd_model': self.svd_model,            # SVD模型
            'user_embeddings': self.user_embeddings, # 用户嵌入向量
            'item_embeddings': self.item_embeddings, # 物品嵌入向量
            'user_item_matrix': self.user_item_matrix # 用户-物品交互矩阵
        }
        
        # 使用pickle序列化保存模型
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"模型已保存到：{path}")
    
    def load_model(self, path="simple_rec_model.pkl"):
        """
        从文件加载训练好的模型
        
        Args:
            path (str): 模型文件路径
        """
        # 使用pickle反序列化加载模型
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        # 恢复模型的各个组件
        self.user_encoder = model_data['user_encoder']
        self.item_encoder = model_data['item_encoder']
        self.svd_model = model_data['svd_model']
        self.user_embeddings = model_data['user_embeddings']
        self.item_embeddings = model_data['item_embeddings']
        self.user_item_matrix = model_data['user_item_matrix']
        
        print(f"模型已从 {path} 加载")
    
    def run_training(self):
        """
        完整的训练流程
        按顺序执行数据加载、模型训练、评估和保存
        """
        print("=== 简化推荐模型训练开始 ===")
        
        # 1. 加载和预处理数据
        data = self.load_and_preprocess_data()
        
        # 2. 创建用户-物品交互矩阵
        self.create_user_item_matrix(data)
        
        # 3. 训练嵌入向量（SVD分解）
        self.train_embeddings()
        
        # 4. 评估模型性能
        hit_rate = self.evaluate_model(data)
        
        # 5. 保存训练好的模型
        self.save_model()
        
        print("=== 训练完成 ===")
        print(f"最终命中率：{hit_rate:.3f}")
        
        # 6. 演示推荐功能：为前5个用户生成推荐
        sample_users = data['user_id'].unique()[:5]  # 获取前5个用户
        print("\n=== 推荐示例 ===")
        for user_id in sample_users:
            recommendations = self.get_recommendations(user_id, top_k=5)  # 为每个用户推荐5个物品
            print(f"用户 {user_id} 的推荐：")
            for item, score in recommendations:
                print(f"  物品 {item}: 分数 {score:.3f}")  # 输出推荐物品和相似度分数
            print()

# 主程序入口
if __name__ == '__main__':
    # 创建推荐模型实例并执行完整训练流程
    model = SimpleRecommendationModel()
    model.run_training() 