"""
encoding: utf-8
    @Author: Jacob
    @Date  : 2024/10/04
    @Desc  :
"""
import warnings

import numpy as np
from numpy import dtype
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", category=UserWarning)


# 1. 数据加载（主函数）
def parsing(data):
    features = data.iloc[:, 1:]
    labels = np.array(data.iloc[:, 0]).reshape(-1, 1)

    features_norm = feature_scaling(features, fit=True)
    labels_norm = label_scaler.fit_transform(labels)

    return features_norm, labels_norm


# 2. 归一化

label_scaler = MinMaxScaler()
feature_scaler = MinMaxScaler()


def feature_scaling(data, fit=False):
    if fit:
        feature_scaler.fit(data)
    return feature_scaler.transform(data)


# 3. 参数初始化(主函数)


# 4. 训练
def training(features, labels):
    """
    feature : shape: (270, 6)
    labels : shape: (270, 1)
    """
    global k, b

    for e in range(epoch):
        # 预测
        pred = features @ k + b  # (270, 1)

        # 求损失
        loss = np.mean((pred - labels) ** 2)  # (1, )

        # 求梯度
        delta_pred = (pred - labels) / len(features)  # (270, 1)
        delta_k = features.T @ delta_pred  # (6, 1)
        delta_b = np.sum(delta_pred)  # (1, )

        # 参数更新
        k -= delta_k * lr  # (6, 1)
        b -= delta_b * lr  # (1, )

        print(f"Loss >> {loss}, k >> {k.mean()}, b >> {b}")


def infering():
    while 1:
        try:
            bedrooms = int(input("室 >>> "))
        except:
            break

        livingrooms = int(input("厅 >>> "))
        lavatories = int(input("卫 >>> "))
        square = float(input("面积 >>> "))
        floor = int(input("楼层 >>> "))
        year = int(input("年份 >>> "))

        features_input = pd.DataFrame([[bedrooms, livingrooms, lavatories, square, floor, year]])

        features_input_norm = feature_scaling(features_input)  # (1, 6)

        pred = features_input_norm @ k + b  # (1, 1)

        pred = label_scaler.inverse_transform(pred)  # (1, 1)

        print(f"预测房价: {pred}")


if __name__ == '__main__':
    # 1. 数据加载
    path = "./data/上海二手房价.csv"
    data = pd.read_csv(path)  # (270, 7)

    X_train, Y_train = parsing(data)  # (270, 6)  (270, 1)

    epoch = 10000
    lr = 0.001
    k = np.ones((X_train.shape[-1], 1), dtype=np.float32)  # (6, 1)
    # b = np.zeros((X_train.shape[0], 1), dtype=np.float32)       # (270, 1)
    b = np.array([0.])  # (1, )

    training(X_train, Y_train)
    infering()
