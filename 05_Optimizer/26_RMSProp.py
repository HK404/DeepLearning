# RMSProp算法和AdaGrad算法的不同在于
# RMSProp算法使用了小批量随机梯度按元素平方的指数加权移动平均来调整学习率
import math
import torch
import sys

sys.path.append("..")
import d2lzh_pytorch as d2l

# 飞机噪音数据集
features, labels = d2l.get_data_ch7()


def init_rmsprop_states():
    s_w = torch.zeros((features.shape[1], 1), dtype=torch.float32)
    s_b = torch.zeros(1, dtype=torch.float32)
    return (s_w, s_b)


def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        s.data = gamma * s.data + (1 - gamma) * (p.grad.data) ** 2
        p.data -= hyperparams['lr'] * p.grad.data / torch.sqrt(s + eps)


# 我们将初始学习率设为0.01，并将超参数γγ设为0.9。此时，变量st
# 可看作是最近1/(1−0.9)=10个时间步的平方项gt⊙gtg的加权平均
d2l.train_ch7(rmsprop, init_rmsprop_states(), {'lr': 0.01, 'gamma': 0.9},
              features, labels)

# Pytorch实现
d2l.train_pytorch_ch7(torch.optim.RMSprop, {'lr': 0.01, 'alpha': 0.9},
                    features, labels)
