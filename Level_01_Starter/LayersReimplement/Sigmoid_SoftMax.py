# coding=utf-8
"""
@Author: Jacob Y
@Date  : 12/12/2024
@Desc  : 
"""
"""
encoding: utf-8
    @Author: Jacob
    @Date  : 2024/10/10
    @Desc  :
    使用np.clip()对Sigmoid输入范围进行裁剪
    SoftMax输入减去当前最大值，防止np.exp(x)中的x过大导致溢出
"""
import numpy as np


def makeOneHot(labels, classes):
    pass


def sigmoid(x):
    x = np.clip(x, -1e+2, 1e+3000)
    return 1 / (1 + np.exp(-x))


def softmax(data):
    # data = np.clip(data, -1e200, 1e+2)
    data = data - data.max(axis=-1, keepdims=True)
    e_ex = np.exp(data)
    e_ex_sum = np.sum(e_ex, axis=-1, keepdims=True)
    # print(e_ex_sum)
    return e_ex / e_ex_sum


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


class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        self.sig_out = sigmoid(x)
        return self.sig_out

    def backward(self, G):
        return G * (self.sig_out * (1 - self.sig_out))


class Softmax:
    def __init__(self):
        pass

    def forward(self, x):
        self.p = softmax(x)
        return self.p

    def backward(self, G):
        return (self.p - G) / len(G)


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
    Sigmoid(),
    Linear(1024, 10),
    Softmax(),
]

model = ModuleList(layers)
