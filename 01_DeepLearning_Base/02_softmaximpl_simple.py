import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
import d2lzh_pytorch as d2l
from collections import OrderedDict
sys.path.append("../..")

# 使用pytorch简单实现softmax回归

# 1.获取和读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 2.定义和初始化模型
num_inputs = 784
num_outputs = 10


class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):  # x shape: (batch, 1, 28, 28)
        y = self.linear(x.view(x.shape[0], -1))
        return y


net = LinearNet(num_inputs, num_outputs)


net = nn.Sequential(
    # FlattenLayer(),
    # nn.Linear(num_inputs, num_outputs)
    OrderedDict([
        ('flatten', d2l.FlattenLayer()),
        ('linear', nn.Linear(num_inputs, num_outputs))
    ])
)

init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

# 3.softmax和损失函数
loss = nn.CrossEntropyLoss()

# 4.优化算法(sgd)小批量随机梯度下降
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# 5.训练模型
num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
