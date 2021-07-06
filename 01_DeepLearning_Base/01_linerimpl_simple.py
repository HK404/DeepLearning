import torch
import numpy as  np
import torch.nn as nn

# 使用pytorch的包简洁实现线性回归，样本数1000，特征数2，真实权重为2，-3.4，bias=4.2
# 1.生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b

# 2.读取数据
import torch.utils.data as Data

batch_size = 10
# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)
# 随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
for X, y in data_iter:
    print(X, y)
    break


# 3.定义模型
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    # forward 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y


net = LinearNet(num_inputs)
print(net)  # 使用print可以打印出网络的结构
for param in net.parameters():
    print(param)

# 4.初始化参数模型
from torch.nn import init

init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)  # 也可以直接修改bias的data: net[0].bias.data.fill_(0)

# 5.定义损失函数
loss = nn.MSELoss()

# 6.定义优化算法
import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)

# 7.训练模型
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad()  # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
    dense = net.linear
    print(true_w, dense.weight)
    print(true_b, dense.bias)
    print('epoch %d, loss: %f' % (epoch, l.item()))
    print()
