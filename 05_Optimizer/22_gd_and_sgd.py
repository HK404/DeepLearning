# 简单的梯度下降与随机梯度下降算法
import numpy as np
import torch
import math
import sys

sys.path.append("..")
import d2lzh_pytorch as d2l


# 一维梯度下降
# 原函数为fx = x**2，我们知道x=0有全局最小值
def gd(eta):
    x = 10
    results = [x]
    for i in range(10):
        x -= eta * 2 * x  # f(x) = x * x的导数为f'(x) = 2 * x
        results.append(x)
    print('epoch 10, x:', x)
    return results


res = gd(0.2)
print(res)


# 显示梯度下降路径
def show_trace(res):
    n = max(abs(min(res)), abs(max(res)), 10)
    f_line = np.arange(-n, n, 0.1)
    d2l.set_figsize()
    d2l.plt.plot(f_line, [x * x for x in f_line])
    d2l.plt.plot(res, [x * x for x in res], '-o')
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('f(x)')
    d2l.plt.show()


show_trace(res)

# 1、eta被称为学习率，eta过小时导致需要更多次迭代才能接近最佳值
show_trace(gd(0.05))

# 2、eta过大时会导致泰勒展开公式不在成立，此时无法保证迭代x会降低fx的值
show_trace(gd(1.1))


# 多维函数梯度下降
def train_2d(trainer):
    x1, x2, s1, s2 = -5, -2, 0, 0  # s1和s2是自变量状态，本章后续几节会使用
    results = [(x1, x2)]
    for i in range(20):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print('epoch %d, x1 %f, x2 %f' % (i + 1, x1, x2))
    return results


def show_trace_2d(f, results):
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
    d2l.plt.show()

eta = 0.1


def f_2d(x1, x2):  # 目标函数
    return x1 ** 2 + 2 * x2 ** 2


# 梯度下降
def gd_2d(x1, x2, s1, s2):
    return x1 - eta * 2 * x1, x2 - eta * 4 * x2, 0, 0


show_trace_2d(f_2d, train_2d(gd_2d))

# 随机梯度下降
def sgd_2d(x1, x2, s1, s2):
    return (x1 - eta * (2 * x1 + np.random.normal(0.1)),
            x2 - eta * (4 * x2 + np.random.normal(0.1)), 0, 0)

show_trace_2d(f_2d, train_2d(sgd_2d))
