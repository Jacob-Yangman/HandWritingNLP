# coding=utf-8
"""
@Author: Jacob Y
@Date  : 2024/10/16
@Desc  : 
"""
import numpy as np


class Parameter:
    def __init__(self, params):
        self.params = params
        self.grad = np.zeros_like(self.params)

class Module:
    def __init__(self):
        self.info = "Module: \n"
        self.params = []


class Linear(Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.info = f"Linear\t\t({in_features}, {out_features})"

        self.w = Parameter(np.random.normal(size=(in_features, out_features)))
        self.b = Parameter(np.random.normal(size=(1, out_features)))

        self.params.append(self.w)
        self.params.append(self.b)


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


class Model:
    def __init__(self):
        self.model_list = ModuleList(
            [
                Linear(784, 256),
                # ReLU(),
                # Dropout(0.2),
                ...
            ])

    def forward(self, x, label=None):
        p = self.model_lst.forward(x)

        if label is not None:
            self.label = label
            loss = - np.mean(np.sum(label * np.log(p + epsilon), axis=1))
            return loss
        else:  # 测试模型时，不传入标签，返回预测结果中的最大值
            return np.argmax(p, axis=-1)

    def backward(self):
        self.model_lst.backward(self.label)

    def __str__(self):
        return self.model_lst.__str__()

    def parameters(self):
        all_params = list()
        for layer in self.model_lst.moduleLst:
            all_params.extend(layer.params)
        return all_params  # 此时返回的参数列表如果不为空，则包含的是Parameter类的实例对象


if __name__ == '__main__':
    epsilon = 1e-8   # 防止损失函数中出现log(0)
    model = Model()
    ...