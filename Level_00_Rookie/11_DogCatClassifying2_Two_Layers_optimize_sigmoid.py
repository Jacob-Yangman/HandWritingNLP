# coding=utf-8
"""
    @Author: Jacob
    @Date  : 2024/10/04
    @Desc  : 使用两层网络，优化Sigmoid运算
"""
import warnings
import numpy as np
import pandas as pd

from data.dog_cat_data import get_data

from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# 1. 数据加载
features, labels = get_data(50)

# 2. 归一化
feature_scaler = MinMaxScaler()
label_scaler = MinMaxScaler()


def feature_scaling(data, fit=False):
    if fit:
        feature_scaler.fit(data)
    return feature_scaler.transform(data)


features_norm = feature_scaling(features, fit=True)  # (100, 3)
labels_norm = label_scaler.fit_transform(labels)  # (100, 1)

# 3. 参数初始化
epoch = 1000
lr = 0.1
w1 = np.random.normal(size=(3, 50))
w2 = np.random.normal(size=(50, 1))
b1 = 0.
b2 = 0.


# 4. 训练
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def desigmoid(y):
    return -np.log(1 / y - 1)


for e in range(epoch):
    # 预测
    layer1_out = features_norm @ w1 + b1  # (100, 50)

    layer2_out = sigmoid(layer1_out @ w2 + b2)  # (100, 1)

    # 求损失
    loss = -np.mean(labels_norm * np.log(layer2_out) + (1 - labels_norm) * np.log(1 - layer2_out))  # (1, )

    # 求梯度
    delta_pred = (layer2_out - labels_norm) / len(features)  # (100, 1)
    delta_w2 = layer1_out.T @ delta_pred  # (50, 1)
    delta_b2 = np.sum(delta_pred)  # (1, )
    delta_layer1_out = delta_pred @ w2.T  # (100, 50)
    delta_w1 = features_norm.T @ delta_layer1_out  # (3, 50)
    delta_b1 = np.sum(delta_layer1_out)  # (1, )

    # 参数更新
    w2 -= delta_w2 * lr  # (50, 1)
    w1 -= delta_w1 * lr  # (3, 50)
    b2 -= delta_b2 * lr  # (1, )
    b1 -= delta_b1 * lr  # (1, )

    print(f"Loss >> {loss:.5f}\t\t"
          f"w1 >> {w1.mean():.3f}\t"
          f"w2 >> {w2.mean():.3f}\t"
          f"b1 >> {b1:.3f}\t"
          f"b2 >> {b2:.3f}")

# 5. 模型上线、推理
threshold = 0.5

while 1:
    try:
        feature1 = float(input("特征1 >>> "))
    except:
        break

    feature2 = float(input("特征2 >>> "))
    feature3 = float(input("特征3 >>> "))

    features_input = pd.DataFrame([[feature1, feature2, feature3]])

    features_input_norm = feature_scaling(features_input)  # (1, 3)

    infer_layer1_out = features_input_norm @ w1 + b1  # (1, 1)
    pred = infer_layer1_out @ w2 + b2  # (1, 1)
    sigmoid_pred = sigmoid(pred)  # (1, 1)

    # pred = label_scaler.inverse_transform(pred)              # (1, 1)

    print(f"预测结果: {pred}")
    print("Sigmoid：", sigmoid_pred)

    if pred > desigmoid(threshold):
        print("🐶")
    else:
        print("🐱")
