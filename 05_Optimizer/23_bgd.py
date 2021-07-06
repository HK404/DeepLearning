# 小批量随机梯度下降实现
import numpy as np
import time
import torch
from torch import nn, optim
import sys

sys.path.append("../..")
import d2lzh_pytorch as d2l


# 读取飞机噪音数据集
def get_data_ch7():
    data = np.genfromtxt('../data/airfoil_self_noise.dat', delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)  # 标准化？
    return torch.tensor(data[:1500, :-1], dtype=torch.float32), \
           torch.tensor(data[:1500, -1], dtype=torch.float32)  # 前1500个样本(每个样本5个特征) -1列表最后一列元素


features, labels = get_data_ch7()
features.shape  # torch.Size([1500, 5])


# 从零实现bgd
# 我们将在训练函数里对各个小批量样本的损失求平均，因此优化算法里的梯度不需要除以批量大小
def sgd(params, states, hyperparams):
    for p in params:
        p.data -= hyperparams['lr'] * p.grad.data


# 一个通用的训练函数，以方便本章后面介绍的其他优化算法使用
# 它初始化一个线性回归模型，然后可以使用小批量随机梯度下降以及后续小节介绍的其他算法来训练模型
def train_ch7(optimizer_fn, states, hyperparams, features, labels,
              batch_size=10, num_epochs=2):
    # 初始化模型
    net, loss = d2l.linreg, d2l.squared_loss
    w = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(features.shape[1], 1)), dtype=torch.float32),
                           requires_grad=True)
    b = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)

    def eval_loss():
        return loss(net(features, w, b), labels).mean().item()

    ls = [eval_loss()]
    data_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features, labels), batch_size, shuffle=True)

    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            l = loss(net(X, w, b), y).mean()  # 使用平均损失

            # 梯度清零
            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()

            l.backward()
            # 上面的sgd函数
            optimizer_fn([w, b], states, hyperparams)  # 迭代模型参数
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())  # 每100个样本记录下当前训练误差
    # 打印结果和作图
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    print('loss: %f' % (ls[0]))
    d2l.set_figsize()
    d2l.plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('loss')
    d2l.plt.show()


def train_sgd(lr, batch_size, num_epochs=2):
    train_ch7(sgd, None, {'lr': lr}, features, labels, batch_size, num_epochs)


# 1、批量大小为样本总量时，优化算法等价于梯度下降
# 梯度下降的1个迭代周期对模型参数只迭代1次。可以看到6次迭代后目标函数值（训练损失）的下降趋向了平稳
train_sgd(1, 1500, 6)

# 2、当批量大小为1时，优化使用的是随机梯度下降
# 为了简化实现，有关（小批量）随机梯度下降的实验中我们未对学习率进行自我衰减，而是直接采用较小的常数学习率
# 随机梯度下降中，每处理一个样本会更新一次自变量（模型参数），一个迭代周期里会对自变量进行1,500次更新
train_sgd(0.005, 1)

# 3、当批量大小为10时，优化使用的是小批量随机梯度下降
# 它在每个迭代周期的耗时介于梯度下降和随机梯度下降的耗时之间
train_sgd(0.05, 10)
