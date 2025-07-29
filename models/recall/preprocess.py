'''
author: yinchao
date: Do not edit
team: wuhan operational dev.
Description: 
'''
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
# @Time    : 2020-11-11 21:10
# @Author  : Hongbo Huang
# @File    : preprocess.py
from tqdm import tqdm
import numpy as np
import random

# 修复TensorFlow 2.16兼容性 - pad_sequences导入
try:
    from keras.utils import pad_sequences
except ImportError:
    try:
        from keras.preprocessing.sequence import pad_sequences
    except ImportError:
        from keras.preprocessing.sequence import pad_sequences

def gen_data_set(data, negsample=0):
    data.sort_values("timestamp", inplace=True)  #是否用排序后的数据集替换原来的数据，这里是替换
    item_ids = data['item_id'].unique()    #item需要进行去重

    train_set = list()
    test_set = list()
    for reviewrID, hist in tqdm(data.groupby('user_id')):   #评价过,  历史记录
        pos_list = hist['item_id'].tolist()
        rating_list = hist['rating'].tolist()

        if negsample > 0:    #负样本
            candidate_set = list(set(item_ids) - set(pos_list))   #去掉用户看过的item项目
            neg_list = np.random.choice(candidate_set, size=len(pos_list) * negsample, replace=True)  #随机选择负采样样本
        for i in range(1, len(pos_list)):
            if i != len(pos_list) - 1:
                # 只传递item_id列表，不是完整的历史记录
                hist_item_ids = pos_list[:i][::-1]  # 反转顺序，获取到当前位置的历史item_id
                train_set.append((reviewrID, hist_item_ids, pos_list[i], 1, len(hist_item_ids), rating_list[i]))  #训练集和测试集划分  [::-1]从后玩前数
                for negi in range(negsample):
                    train_set.append((reviewrID, hist_item_ids, neg_list[i * negsample + negi], 0, len(hist_item_ids)))
            else:
                hist_item_ids = pos_list[:i][::-1]  # 反转顺序，获取到当前位置的历史item_id
                test_set.append((reviewrID, hist_item_ids, pos_list[i], 1, len(hist_item_ids), rating_list[i]))

    random.shuffle(train_set)     #打乱数据集
    random.shuffle(test_set)
    return train_set, test_set

def gen_model_input(train_set, user_profile, seq_max_len):
    train_uid = np.array([line[0] for line in train_set], dtype=np.int32)
    train_seq = [line[1] for line in train_set]
    train_iid = np.array([line[2] for line in train_set], dtype=np.int32)
    train_label = np.array([line[3] for line in train_set], dtype=np.int32)
    train_hist_len = np.array([line[4] for line in train_set], dtype=np.int32)
    

    """
    pad_sequences数据预处理
    sequences：浮点数或整数构成的两层嵌套列表
    maxlen：None或整数，为序列的最大长度。大于此长度的序列将被截短，小于此长度的序列将在后部填0.
    dtype：返回的numpy array的数据类型
    padding：'pre'或'post'，确定当需要补0时，在序列的起始还是结尾补`
    truncating：'pre'或'post'，确定当需要截断序列时，从起始还是结尾截断
    value：浮点数，此值将在填充时代替默认的填充值0
    """
    train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', truncating='post', value=0, dtype='int32')
    # 恢复完整的模型输入
    train_model_input = {"user_id": train_uid, "item_id": train_iid, "hist_item_id": train_seq_pad,
                         "hist_len": train_hist_len}
    
    # 创建用户特征映射字典，避免pandas索引问题
    user_feature_dict = {}
    for _, row in user_profile.iterrows():
        user_feature_dict[row['user_id']] = {
            'gender': row['gender'],
            'age': row['age'], 
            'city': row['city']
        }
    
    # 使用字典映射获取用户特征，确保数据类型一致性
    for key in {"gender", "age", "city"}:
        feature_values = []
        for uid in train_uid:
            if uid in user_feature_dict:
                feature_values.append(user_feature_dict[uid][key])
            else:
                # 如果用户ID不存在，使用默认值1（因为特征编码从1开始）
                feature_values.append(1)
        train_model_input[key] = np.array(feature_values, dtype=np.int32)

    return train_model_input, train_label