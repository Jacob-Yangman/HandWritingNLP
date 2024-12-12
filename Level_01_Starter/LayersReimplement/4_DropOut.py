# coding=utf-8
"""
@Author: Jacob Y
@Date  : 2024/10/15
@Desc  : 
"""
import numpy as np


class Dropout:
    def __init__(self, rate=0.3):
        super().__init__()
        self.info = f"DropOut()\t\trate={rate}"
        self.rate = rate

    def forward(self, x):
        dropoutMatrix = np.random.rand(*x.shape)  # 生成一个mask矩阵，确定哪些位置的节点需要失活
        self.neg = dropoutMatrix < self.rate
        x[self.neg] = 0  # 将输入矩阵中相应位置的节点失活
        return x

    def backward(self, G):
        G[self.neg] = 0
        return G
