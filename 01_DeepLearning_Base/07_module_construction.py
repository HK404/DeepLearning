# 4.1.1 继承Module类来构造模型
from collections import OrderedDict

import torch
from torch import nn

# 1.使用Module类来构造模型
class MLP(nn.Module):
    # 声明带有模型参数的层，这里声明了两个全连接层
    def __init__(self, **kwargs):
        # 调用MLP父类Module的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
        # 参数，如“模型参数的访问、初始化和共享”一节将介绍的模型参数params
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784, 256) # 隐藏层
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)  # 输出层


    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)
# 以上的MLP类中无须定义反向传播函数。系统将通过自动求梯度而自动生成反向传播所需的backward函数
X = torch.rand(2, 784)
net = MLP()
print(net)
net(X)

# 2.使用Sequential类
# 实现一个与Sequential类有相同功能的MySequential类
# 这或许可以帮助读者更加清晰地理解Sequential类的工作机制
class MySequential(nn.Module):
    from collections import OrderedDict
    def __init__(self, *args):
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict): # 如果传入的是一个OrderedDict
            for key, module in args[0].items():
                self.add_module(key, module)  # add_module方法会将module添加进self._modules(一个OrderedDict)
        else:  # 传入的是一些Module
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
    def forward(self, input):
        # self._modules返回一个 OrderedDict，保证会按照成员添加时的顺序遍历成员
        for module in self._modules.values():
            input = module(input)
        return input

net = MySequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
        )
print(net)
net(X)

# 4.1.3 构造复杂的模型
# 直接继承Module类可以极大地拓展模型构造的灵活性
# 下面我们构造一个稍微复杂点的网络FancyMLP
# 在这个网络中，我们通过get_constant函数创建训练中不被迭代的参数，即常数参数
# 在前向计算中，除了使用创建的常数参数外，我们还使用Tensor的函数和Python的控制流，并多次调用相同的层
class FancyMLP(nn.Module):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)

        self.rand_weight = torch.rand((20, 20), requires_grad=False) # 不可训练参数（常数参数）
        self.linear = nn.Linear(20, 20)

    def forward(self, x):
        x = self.linear(x)
        # 使用创建的常数参数，以及nn.functional中的relu函数和mm函数
        x = nn.functional.relu(torch.mm(x, self.rand_weight.data) + 1)

        # 复用全连接层。等价于两个全连接层共享参数
        x = self.linear(x)
        # 控制流，这里我们需要调用item函数来返回标量进行比较
        while x.norm().item() > 1:
            x /= 2
        if x.norm().item() < 0.8:
            x *= 10
        return x.sum()

X = torch.rand(2, 20)
net = FancyMLP()
print(net)
print(net(X))

# 因为FancyMLP和Sequential类都是Module类的子类，所以我们可以嵌套调用它们
class NestMLP(nn.Module):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential(nn.Linear(40, 30), nn.ReLU())

    def forward(self, x):
        return self.net(x)

net = nn.Sequential(NestMLP(), nn.Linear(30, 20), FancyMLP())

X = torch.rand(2, 40)
print(net)
print(net(X))

