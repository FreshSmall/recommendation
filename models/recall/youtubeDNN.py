#!/usr/bin/env python 
# -*- coding: utf-8 -*-
#                       _oo0oo_
#                      o8888888o
#                      88" . "88
#                      (| -_- |)
#                      0\  =  /0
#                    ___/`---'\___
#                  .' \\|     |// '.
#                 / \\|||  :  |||// \
#                / _||||| -:- |||||- \
#               |   | \\\  -  /// |   |
#               | \_|  ''\---/''  |_/ |
#               \  .-\__  '-'  ___/-. /
#             ___'. .'  /--.--\  `. .'___
#          ."" '<  `.___\_<|>_/___.' >' "".
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#         \  \ `_.   \_ __\ /__ _/   .-` /  /
#     =====`-.____`.___ \_____/___.-`___.-'=====
#                       `=---='
#
#
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#               佛祖保佑         永无BUG
# @Time    : 2021-11-09 20:28
# @Author  : Hongbo Huang
# @File    : youtubeDNN.py
import sys
import os

# 将项目根目录添加到Python路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from models.recall.preprocess import gen_data_set, gen_model_input
from deepctr.feature_column import SparseFeat, VarLenSparseFeat

# 修复TensorFlow 2.16兼容性问题
import tensorflow as tf

# 强制TensorFlow 2.x使用1.x兼容模式
tf.compat.v1.disable_eager_execution()
tf.compat.v1.enable_v2_tensorshape()

try:
    from keras.utils import backend as K
    from keras.models import Model
except ImportError:
    try:
        from tensorflow.python.keras import backend as K
        from tensorflow.python.keras.models import Model
    except ImportError:
        # 降级方案
        K = tf.keras.backend
        Model = tf.keras.models.Model

from deepmatch.models import *
from deepmatch.utils import recall_N
from deepmatch.utils import sampledsoftmaxloss
import numpy as np
from tqdm import tqdm

# 添加faiss导入（如果需要的话）
try:
    import faiss
except ImportError:
    print("Warning: faiss not installed. Please install it if you need similarity search.")
    faiss = None


class YoutubeModel(object):
    def __init__(self, embedding_dim=32):
        self.SEQ_LEN = 50
        self.embedding_dim = embedding_dim
        self.user_feature_columns = None
        self.item_feature_columns = None

    def training_set_construct(self):
        # 加载数据 - 使用动态路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        data_path = os.path.join(project_root, "data", "read_history.csv")
        data = pd.read_csv(data_path)
        
        # 处理timestamp列 - 转换为数值格式以便排序
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['timestamp'] = data['timestamp'].astype('int64') // 10**9  # 转换为Unix时间戳
        
        # 负采样个数
        negsample = 0
        # 特征编码 - 只处理分类特征，不包括timestamp和rating
        features = ["user_id", "item_id", "gender", "age", "city"]
        features_max_idx = {}
        for feature in features:
            lbe = LabelEncoder()
            data[feature] = lbe.fit_transform(data[feature]) + 1
            # 确保数据类型为int32
            data[feature] = data[feature].astype('int32')
            features_max_idx[feature] = data[feature].max() + 1

        # 抽取用户、物品特征
        user_info = data[["user_id", "gender", "age", "city"]].drop_duplicates('user_id')  # 去重操作
        item_info = data[["item_id"]].drop_duplicates('item_id')
        # 注释掉设置索引的操作，保持原始DataFrame格式
        # user_info.set_index("user_id", inplace=True)

        # 构建输入数据
        train_set, test_set = gen_data_set(data, negsample)
        # 转化为模型的输入
        train_model_input, train_label = gen_model_input(train_set, user_info, self.SEQ_LEN)
        test_model_input, test_label = gen_model_input(test_set, user_info, self.SEQ_LEN)
        # 用户端特征输入 - 恢复完整配置
        self.user_feature_columns = [SparseFeat('user_id', features_max_idx['user_id'], self.embedding_dim),
                                     SparseFeat('gender', features_max_idx['gender'], self.embedding_dim),
                                     SparseFeat('age', features_max_idx['age'], self.embedding_dim),
                                     SparseFeat('city', features_max_idx['city'], self.embedding_dim),
                                     VarLenSparseFeat(SparseFeat('hist_item_id', features_max_idx['item_id'],
                                                                 self.embedding_dim, embedding_name='hist_item'),
                                                      self.SEQ_LEN, 'mean', 'hist_len')
                                     ]
        # 物品端的特征输入
        self.item_feature_columns = [SparseFeat('item_id', features_max_idx['item_id'], self.embedding_dim)]

        return train_model_input, train_label, test_model_input, test_label, train_set, test_set, user_info, item_info

    def training_model(self, train_model_input, train_label):
        K.set_learning_phase(True)
        # TensorFlow配置已经在模块开头设置了，这里移除重复配置
        
        # 定义模型 - 使用正确的YoutubeDNN参数
        model = YoutubeDNN(
            user_feature_columns=self.user_feature_columns, 
            item_feature_columns=self.item_feature_columns, 
            user_dnn_hidden_units=(128, 64, self.embedding_dim),
            sampler_config={'num_sampled': 100}  # 正确的采样配置参数
        )
        model.compile(optimizer="adam", loss=sampledsoftmaxloss)
        # 保存训练过程中的数据 - 减少epochs用于测试
        model.fit(train_model_input, train_label, batch_size=512, epochs=2, verbose=1, validation_split=0.0,)
        return model

    def extract_embedding_layer(self, model, test_model_input, item_info):
        all_item_model_input = {"item_id": item_info['item_id'].values, }
        # 获取用户、item的embedding_layer
        user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
        item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)

        user_embs = user_embedding_model.predict(test_model_input, batch_size=2 ** 12)
        item_embs = item_embedding_model.predict(all_item_model_input, batch_size=2 ** 12)
        print(user_embs.shape)
        print(item_embs.shape)
        return user_embs, item_embs

    def eval(self, user_embs, item_embs, test_model_input, item_info, test_set):
        if faiss is None:
            print("Error: faiss is required for evaluation. Please install it.")
            return 0, 0
            
        test_true_label = {line[0]: line[2] for line in test_set}
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(item_embs)
        D, I = index.search(np.ascontiguousarray(user_embs), 50)
        s = []
        hit = 0

        # 统计预测结果
        for i, uid in tqdm(enumerate(test_model_input['user_id'])):
            try:
                pred = [item_info['item_id'].values[x] for x in I[i]]
                recall_score = recall_N(test_true_label[uid], pred, N=50)
                s.append(recall_score)
                if test_true_label[uid] in pred:
                    hit += 1
            except:
                print(i)

        # 计算召回率和命中率
        recall = np.mean(s)
        hit_rate = hit / len(test_model_input['user_id'])

        return recall, hit_rate

    def scheduler(self):
        # 构建训练集、测试集
        train_model_input, train_label, test_model_input, test_label, \
        train_set, test_set, user_info, item_info = self.training_set_construct()
        #
        model = self.training_model(train_model_input, train_label)

        # 获取用户、item的layer
        user_embs, item_embs = self.extract_embedding_layer(model, test_model_input, item_info)
        # 评估模型
        recall, hit_rate = self.eval(user_embs, item_embs, test_model_input, item_info, test_set)
        print(f"Recall: {recall}, Hit Rate: {hit_rate}")


if __name__ == '__main__':
    model = YoutubeModel()
    model.scheduler()