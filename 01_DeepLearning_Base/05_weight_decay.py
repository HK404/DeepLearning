# 解决过拟合问题常用方法，权重衰减
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("../..")
import d2lzh_pytorch as d2l

# 1.高维线性回归实验
# 其中噪声项ϵϵ服从均值为0、标准差为0.01的正态分布。
# 为了较容易地观察过拟合，我们考虑高维线性回归问题，如设维度p=200；
# 同时，我们特意把训练数据集的样本数设低，如20。
n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05

features = torch.randn((n_train + n_test, num_inputs))
labels = torch.matmul(features, true_w) + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]

# 2.从零开始实现权重衰减
# 初始化模型参数
def init_params():
    w = torch.randn((num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

# 3.定义L2范数惩罚项
def l2_penalty(w):
    return (w**2).sum() / 2

# 4.定义和训练模型
batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = d2l.linreg, d2l.squared_loss

dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

def fit_and_plot(lambd):
    w, b = init_params()
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            # 添加了L2范数惩罚项
            l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            l = l.sum()

            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()
            l.backward()
            d2l.sgd([w, b], lr, batch_size)
        train_ls.append(loss(net(train_features, w, b), train_labels).mean().item())
        test_ls.append(loss(net(test_features, w, b), test_labels).mean().item())
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                 range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', w.norm().item())

# 5.观察过拟合，当lambd设为0时，我们没有使用权重衰减。
fit_and_plot(lambd=0)

# 6.使用权重衰减
fit_and_plot(lambd=3)

