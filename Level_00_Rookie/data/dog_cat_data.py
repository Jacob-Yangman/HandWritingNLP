"""
encoding: utf-8
    @Author: Jacob
    @Date  : 2024/10/06
    @Desc  : 
"""
import numpy as np
import random

def generate_data(num_samples, bounds, label):

    data = np.array([[random.uniform(low, high) for low, high in bounds] for _ in range(num_samples)])
    labels = np.full((num_samples, 1), label)
    return np.hstack((data, labels))

def get_data(num_samples=50):
    """
    返回指定样本数量的猫狗数据 (特征, 标签)
    数据类型：ndarray
    数据维度：features: (num_samples, 3)   labels: (num_samples, 1)
    @param num_samples: 猫/狗各生成的样本数量，默认为50条
    @return: (features, labels)
    """

    # 定义范围
    dog_bounds = [(4.8, 5.5), (5.2, 6), (7, 8)]
    cat_bounds = [(3.1, 3.8), (3.2, 5.1), (5.6, 6)]

    # 生成数据
    dogs = generate_data(num_samples, dog_bounds, 1)
    cats = generate_data(num_samples, cat_bounds, 0)

    # 合并数据
    data = np.vstack((dogs, cats))
    features = data[:, :3]
    labels = data[:, -1].reshape(-1, 1)

    return features, labels
