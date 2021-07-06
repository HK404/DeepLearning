# torchvision.datasets: 一些加载数据的函数及常用的数据集接口；
# torchvision.models: 包含常用的模型结构（含预训练模型），例如AlexNet、VGG、ResNet等；
# torchvision.transforms: 常用的图片变换，例如裁剪、旋转等；
# torchvision.utils: 其他的一些有用的方法。

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys

sys.path.append("..")  # 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l
from d2lzh_pytorch import *

# 数据集和测试集
mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True,
                                                transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True,
                                               transform=transforms.ToTensor())
print(type(mnist_train))
print(len(mnist_train), len(mnist_test))

# 可以通过索引查看数据
feature, label = mnist_train[0]
print(feature.shape, label)  # Channel x Height x Width

# 查看数据集前十个样本,X为图像，y是图像标签0~9(十种类别)
X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
print(len(X[1]))
# print(y[0],y[1])
show_fashion_mnist(X, get_fashion_mnist_labels(y))

# 读取小批量数据
batch_size = 256
if sys.platform.startswith('win'):  # windows平台
    num_workers = 0  # 0表示不用额外的进程来加速读取数据
else:
    num_workers = 4
start = time.time()
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# 查看一下读取一遍训练数据需要的时间

for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))
