# coding=utf-8
"""
@Author: Jacob Y
@Date  : 2024/10/12
@Desc  : 
"""
import numpy as np

lr = 0.001


class Module:
    def __init__(self):
        self.info = "Module: \n"
        self.params = []

class Parameter:
    def __init__(self, params):
        self.params = params
        self.grad = np.zeros_like(self.params)

class Linear(Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.info = f"Linear\t\t({in_features}, {out_features})"
        self.w = Parameter(np.random.normal(size=(in_features, out_features)))
        self.b = Parameter(np.random.normal(size=(1, out_features)))

    def forward(self, x):
        self.x = x
        output = x @ self.w.params + self.b.params
        return output

    def backward(self, G):
        self.w.grad = self.x.T @ G
        self.b.grad = np.sum(G, axis=0, keepdims=True)


        # 参数更新
        self.w.params -= (self.w.grad) * lr
        self.b.params -= (self.b.grad) * lr
        return G @ self.w.params.T