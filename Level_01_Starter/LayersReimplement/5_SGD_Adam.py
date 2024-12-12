# coding=utf-8
"""
@Author: Jacob Y
@Date  : 2024/10/14
@Desc  : 
"""
import numpy as np

lr = 0.001
class Optim:
    def __init__(self, params_lst, lr=0.01):
        self.params = params_lst
        self.lr = lr

    def step(self):
        pass

    def zero_grad(self):
        for param in self.params:
            param.grad = 0


class SGD(Optim):
    def __init__(self, params_lst, lr=0.01):
        super().__init__(params_lst, lr=0.01)

    def step(self):
        for param in self.params:
            param.params -= param.grad * lr



class ADAM(Optim):
    def __init__(self, params_lst, lr=0.01, beta1=0.9, beta2=0.999, e=1e-8):
        super().__init__(params_lst, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.e = e

        self.t = 0

        for p in self.params:
            p.m = 0
            p.v = 0

    def step(self):
        self.t += 1
        for p in self.params:
            gt = p.grad
            p.m = self.beta1 * p.m + (1 - self.beta1) * gt
            p.v = self.beta2 * p.v + (1 - self.beta2) * gt ** 2
            mt_ = p.m / (1 - self.beta1**self.t)
            vt_ = p.v / (1 - self.beta2**self.t)

            p.params = p.params - self.lr * mt_ / (np.sqrt(vt_) + self.e)