# coding=utf-8
"""
@Author: Jacob
@Date  : 2024/10/10
@Desc  :
一、封装Linear、Sigmoid、SoftMax，提供forward()和backward()方法
二、统一forward()和backward()方法的输入和输出
三、封装所有层为ModuleList
"""
import numpy as np


class Linear:
    def __init__(self, in_features, out_features):
        self.weight = np.random.normal(size=(in_features, out_features))
        self.bias = np.random.normal(size=(1, out_features))

    def forward(self, x):
        self.x = x
        output = x @ self.weight + self.bias
        return output

    def backward(self, G):
        delta_w = self.x.T @ G
        delta_b = np.sum(G, axis=0, keepdims=True)
        # 参数更新
        self.weight -= delta_w * lr
        self.bias -= delta_b * lr
        return G @ self.weight.T


class ModuleList:
    def __init__(self, moduleLst):
        self.moduleLst = moduleLst

    def forward(self, x):
        for layer in self.moduleLst:
            x = layer.forward(x)
        return x

    def backward(self, y):
        for layer in reversed(self.moduleLst):
            y = layer.backward(y)


lr = 0.1

# 搭建网络
layers = [
    Linear(784, 1024),
    Linear(1024, 10),
]

model = ModuleList(layers)

