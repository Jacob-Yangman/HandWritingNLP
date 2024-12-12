"""
encoding: utf-8
    @Author: Jacob
    @Date  : 2024/10/03
    @Desc  : 
"""
import numpy as np
import random

random.seed(10)
"""
np.random.seed(10)

def normalize(data):
    np_data = np.array(data)
    return (np_data - np_data.min()) / (np_data.max() - np_data.min())

def standardize(data):
    np_data = np.array(data)
    return (np_data - np_data.mean()) / np_data.std()
"""

# 1. 数据加载
years = [i for i in range(2000, 2023)]
prices = sorted([random.uniform(6000, 15000) for _ in range(2000, 2023)])

# 2. 数据处理

# 归一化
years_min = min(years)
years_max = max(years)

years_norm = [(y - years_min) / (years_max - years_min) for y in years]

prices_min = min(prices)
prices_max = max(prices)

prices_norm = [(p - prices_min) / (prices_max - prices_min) for p in prices]

# 3. 参数初始化
epoch = 2000
k = 1
b = 0.1
lr = 0.1

# 4. 模型训练
for e in range(epoch):

    for year, price in zip(years_norm, prices_norm):
        # 推理
        pred = k * year + b
        print("Current prediction >>>", pred)
        # 损失函数
        loss = 0.5 * (pred - price) ** 2

        # 梯度计算
        dl_k = (pred - price) * year
        dl_b = pred - price

        # 参数更新
        k -= dl_k * lr
        b -= dl_b * lr

        print(k, b, sep='\t')

# 5. 模型上线、预测

while 1:

    if not (year := input("请输入年份: ")):
        break
    year_norm = (int(year) - years_min) / (years_max - years_min)

    prediction = k * year_norm + b
    prediction_reverse_norm = prediction * (prices_max - prices_min) + prices_min
    print(f"{year}年的房价: {prediction_reverse_norm}")
