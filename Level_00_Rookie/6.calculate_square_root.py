"""
encoding: utf-8
    @Author: Jacob
    @Date  : 2024/10/03
    @Desc  : 手写算法，求解x的平方根
"""

# 梯度下降法
def calSquareRootByGD(n):
    x = n // 2
    lr = 0.001
    for i in range(500):
        pred = x ** 2
        loss = (pred - n) ** 2
        grad = 2 * (pred - n) * 2 * x
        x = x - lr * grad
        print(f"Iteration >>> {i + 1}\tSquareRoot >>> {x}\tLoss >>> {loss}")
    return x

def calSquareRootByNewon(n, eps=1e-10):
    """
    $$x_{t+1}=x_t-\frac{x2-n}{2x}=\frac{1}{2}（x_t+\frac{n}{x_t})$$
    :param n: 求解n的平方根
    :param eps: 可接受的最小误差值
    """
    x = n // 2  # 参数x初始化
    err = 1  # 初始误差
    while err > eps:
        new_x = 0.5 * (x + n / x)
        err = abs(new_x - x)
        x = new_x
    return x

print(calSquareRootByGD(20))
print(calSquareRootByNewon(20))
