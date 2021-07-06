import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
import sys
from d2lzh_pytorch import *
sys.path.append("../..")


# 从零开始实现线性回归
# 1.生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs,
                       dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b

# 加上噪声epsilon
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                       dtype=torch.float32)
print(features[0], labels[0])

# 2.读取数据
batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break

# 3.初始化模型参数
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)
lr = 0.03  # 学习率
num_epochs = 3  # 迭代周期个数

# 4.定义模型
net = linreg

# 5.定义损失函数
loss = squared_loss

# 6.使用优化算法和训练模型
for epoch in range(num_epochs):  # 训练模型一共需要num_epochs个迭代周期
    # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。
    # X和y分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()  # l是有关小批量X和y的损失
        l.backward()  # 小批量的损失对模型参数求梯度
        sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数

        # 不要忘了梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('weight w', w)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))
