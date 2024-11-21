"""
encoding: utf-8
    @Author: Jacob
    @Date  : 2024/10/04
    @Desc  :
    参数矩阵求导规律:
    X * K = Pred
    Loss = (Pred - Label) ** 2
    G = ∂Loss/∂Pred = 2*(Pred - Label)
    ∂Loss/∂K = X.T() * G
"""
import numpy as np
from numpy import dtype

# 1. 数据加载
years = np.arange(2000, 2023)
squares = np.array(sorted(np.random.randint(80, 120, size=(23,))))
prices = np.array(sorted(np.random.uniform(6000, 15000, size=(23,))))

# 2. 归一化
years_max = years.max()
years_min = years.min()
years_norm = (years - years_min) / (years_max - years_min)

squares_max = squares.max()
squares_min = squares.min()
squares_norm = (squares - squares_min) / (squares_max - squares_min)

x_norm = np.stack((years_norm, squares_norm), axis=1)  # (23, 2)

prices_max = prices.max()
prices_min = prices.min()
prices_norm = (prices - prices_min) / (prices_max - prices_min)
y_norm = np.expand_dims(prices_norm, axis=1)  # (23, 1)

# 3. 参数初始化
epoch = 100
lr = 0.2
k = np.ones((2, 1), dtype=np.float32)
b = np.array(1.)

# 4. 训练
for e in range(epoch):
    # 预测
    pred = x_norm @ k + b  # (23, 1)

    # 求损失
    loss = (pred - y_norm) ** 2  # (23, 1)

    # 求梯度
    delta_pred = pred - y_norm  # 省略常数项          # (23, 1)
    ## 除以批次大小，避免批次对参数k的过度影响
    delta_pred = delta_pred / y_norm.shape[0]  # (23, 1)

    delta_k = x_norm.transpose() @ delta_pred  # (2, 1)
    delta_b = np.mean(pred - y_norm)  # (1, )

    # 参数更新
    k -= delta_k * lr
    b -= delta_b * lr

    print(f"Loss >> {loss.mean()}")

while 1:
    year = input("年份 >>> ")

    if not year:
        break

    square = input("面积 >>> ")
    year_norm = (int(year) - years_min) / (years_max - years_min)

    square_norm = (int(square) - squares_min) / (squares_max - squares_min)

    input_norm = np.array((year_norm, square_norm)).reshape(1, 2)  # (1, 2)

    pre_prediction = input_norm @ k + b  # (1, 1)

    prediction = pre_prediction * (prices_max - prices_min) + prices_min

    print(f"预测房价: {prediction}")
