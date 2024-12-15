# coding=utf-8
"""
    @Author: Jacob
    @Date  : 2024/10/04
    @Desc  :
"""
import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from data.dog_cat_data import get_data

warnings.filterwarnings("ignore", category=FutureWarning)

# 1. 数据加载

features, labels = get_data(num_samples=50)

# 2. 归一化
feature_scaler = MinMaxScaler()
label_scaler = MinMaxScaler()


def feature_scaling(data, fit=False):
    if fit:
        feature_scaler.fit(data)
    return feature_scaler.transform(data)


features_norm = feature_scaling(features, fit=True)
labels_norm = label_scaler.fit_transform(labels)

# 3. 参数初始化
epoch = 1000
lr = 0.1
k = np.ones((features.shape[-1], 1), dtype=np.float32)  # (6, 1)
# b = np.zeros((X_train.shape[0], 1), dtype=np.float32)       # (270, 1)
b = np.array([0.])  # (1, )

# 4. 训练
"""
features: (100, 3)
labels: (100, 1)
k: (3, 1)
b: (1, )
"""


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


for e in range(epoch):
    # 预测
    out = features_norm @ k + b  # (100, 1)
    sig_out = sigmoid(out)

    # 求损失
    loss = -np.mean(labels_norm * np.log(sig_out) + (1 - labels_norm) * np.log(1 - sig_out))  # (1, )

    # 求梯度
    delta_pred = (sig_out - labels_norm) / len(features)  # (100, 1)
    delta_k = features_norm.T @ delta_pred  # (6, 1)
    delta_b = np.sum(delta_pred)  # (1, )

    # 参数更新
    k -= delta_k * lr  # (6, 1)
    b -= delta_b * lr  # (1, )

    print(f"Loss >> {loss}, k >> {k.mean()}, b >> {b}")

# 5. 模型上线、推理


while 1:
    try:
        feature1 = float(input("特征1 >>> "))
    except:
        break

    feature2 = float(input("特征2 >>> "))
    feature3 = float(input("特征3 >>> "))

    features_input = pd.DataFrame([[feature1, feature2, feature3]])

    features_input_norm = feature_scaling(features_input)  # (1, 3)

    pred = features_input_norm @ k + b  # (1, 1)
    # sigmoid_pred = sigmoid(features_input_norm @ k + b)      # (1, 1)

    # pred = label_scaler.inverse_transform(pred)              # (1, 1)

    print(f"预测结果: {pred}")

    if pred > 0:
        print("🐶")
    else:
        print("🐱")
