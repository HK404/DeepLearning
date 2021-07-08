# 测试模块
import torch
from torch import nn,optim
import torchvision
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
import torch.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1、mean()维度测试
x = torch.tensor([[[[1, 2],
                   [1, 2]],
                  [[3, 4],
                   [3, 4]]]], dtype=float)
print(x, "\n")
print(x.shape, "\n")
print(len(x.shape), "\n")
# print(torch.randn(3, 3, 3))
print(x.mean(dim=0))
print(x.mean(dim=1))
print(x.mean(dim=2))
print(x.mean(dim=3), "\n")

assert len(x.shape) in (2, 4)
if len(x.shape) == 2:
    # 使用全连接层的情况，计算特征维上的均值和方差
    mean = x.mean(dim=0)
    var = ((x - mean) ** 2).mean(dim=0)
else:
    # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。这里我们需要保持
    # X的形状以便后面可以做广播运算
    mean = x.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
print("mean is :", mean, mean.shape)

# mean = x.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
# print(mean)
# print(mean.shape)

import numpy as np
# 坐标向量
a = np.array([1,2,3])
# 坐标向量
b = np.array([7,8])
# 从坐标向量中返回坐标矩阵
# 返回list,有两个元素,第一个元素是X轴的取值,第二个元素是Y轴的取值
x, y= np.meshgrid(a,b)
print('x is :', x)
print('y is :', y)
print('stack :', np.stack((x, y), axis=2))