# 4.2 模型参数的访问、初始化和共享
import torch
from torch import nn
from torch.nn import init

net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))  # pytorch已进行默认初始化

print(net)
X = torch.rand(2, 4)
Y = net(X).sum()

# 访问上面网络的参数
print(type(net.named_parameters()))
for name, param in net.named_parameters():
    print(name, param.size())

# 使用init初始化模型参数
for name, param in net.named_parameters():
    if 'weight' in name:
        init.normal_(param, mean=0, std=0.01)
        print(name, param.data)

for name, param in net.named_parameters():
    if 'bias' in name:
        init.constant_(param, val=0)
        print(name, param.data)

# 4.2.3 自定义初始化方法
# 令权重有一半概率初始化为0
# 有另一半概率初始化为[−10,−5]和[5,10]两个区间里均匀分布的随机数
def init_weight_(tensor):
    with torch.no_grad():
        tensor.uniform_(-10, 10)
        tensor *= (tensor.abs() >= 5).float()

for name, param in net.named_parameters():
    if 'weight' in name:
        init_weight_(param)
        print(name, param.data)

# 4.2.4 共享模型参数
