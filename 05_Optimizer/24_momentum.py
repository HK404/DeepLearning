# 普通梯度下降会带来一些问题
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
import torch

eta = 0.4  # 学习率

# 目标函数
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

# 优化函数
def gd_2d(x1, x2, s1, s2):
    return x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0


# 在这个例子中
# 给定学习率，梯度下降迭代自变量时会使自变量在竖直方向比在水平方向移动幅度更大
# 那么，我们需要一个较小的学习率从而避免自变量在竖直方向上越过目标函数最优解
# 然而，这会造成自变量在水平方向上朝最优解移动变慢
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))

# 试着将学习率调得稍大一点，此时自变量在竖直方向不断越过最优解并逐渐发散
eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))

# 动量法（优化函数）
def momentum_2d(x1, x2, v1, v2):
    v1 = gamma * v1 + eta * 0.2 * x1
    v2 = gamma * v2 + eta * 4 * x2
    return x1 - v1, x2 - v2, v1, v2

eta, gamma = 0.4, 0.5

# 可以看到动量法在竖直方向上的移动更加平滑，且在水平方向上更快逼近最优解
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))

eta = 0.6
# 下面使用较大的学习率η=0.6η=0.6，此时自变量也不再发散
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))

# 从零开始实现动量法
features, labels = d2l.get_data_ch7()

def init_momentum_states():
    v_w = torch.zeros((features.shape[1], 1), dtype=torch.float32)
    v_b = torch.zeros(1, dtype=torch.float32)
    return (v_w, v_b)

def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v.data = hyperparams['momentum'] * v.data + hyperparams['lr'] * p.grad.data
        p.data -= v.data

d2l.train_ch7(sgd_momentum, init_momentum_states(),
              {'lr': 0.02, 'momentum': 0.5}, features, labels)

# PyTorch简洁实现，使用momentum即可指定动量法
d2l.train_pytorch_ch7(torch.optim.SGD, {'lr': 0.004, 'momentum': 0.9},
                    features, labels)
