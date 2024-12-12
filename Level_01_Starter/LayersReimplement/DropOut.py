# coding=utf-8
"""
@Author: Jacob Y
@Date  : 12/12/2024
@Desc  : 
"""
import numpy as np


class Dropout:
    def __init__(self, rate=0.3):
        self.rate = rate

    def forward(self, x):
        dropoutMatrix = np.random.rand(*x.shape)
        self.neg = dropoutMatrix < self.rate
        x[self.neg] = 0
        return x

    def backward(self, G):
        G[self.neg] = 0
        return G
