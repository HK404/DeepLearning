import torch
from torch import nn
import sys

sys.path.append("..")
import d2lzh_pytorch as d2l

# 1、多通道输入，卷积到一个输出通道
# X.size（[2,3,3]）
X = torch.tensor([[[0, 1, 2],
                   [3, 4, 5],
                   [6, 7, 8]],
                  [[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]]])
# K.size([2,2,2])
K = torch.tensor([[[0, 1],
                   [2, 3]],
                  [[1, 2],
                   [3, 4]]])

res = d2l.corr2d(X[0, :, :], K[0, :, :])
print(res)


def corr2d_multi_in(X, K):
    # 沿着X和K的第0维（通道维）分别计算再相加
    #    res = d2l.corr2d(X[0, :, :], K[0, :, :])
    res = d2l.corr2d(X[0, :, :], K[0, :, :])
    for i in range(1, X.shape[0]):
        res += d2l.corr2d(X[i, :, :], K[i, :, :])
    return res


print(corr2d_multi_in(X, K))

# 测试维度
Z = torch.tensor([[[1, 1, 1, 1],
                   [1, 1, 1, 1],
                   [1, 1, 1, 1]],
                  [[1, 1, 1, 1],
                   [1, 1, 1, 1],
                   [1, 1, 1, 1]]])
# 通道维、高度、宽度
print(Z.shape)

# 2、多通道输入与输出
# 了解一下torch.stack()扩维拼接
X = torch.tensor([[1, 2, 3, 4],
                  [1, 2, 3, 4],
                  [1, 2, 3, 4]])

Y = torch.tensor([[1, 2, 3, 4],
                  [1, 2, 3, 4],
                  [1, 2, 3, 4]])

print('dim0 the result is: \n', torch.stack((X, Y), dim=0), torch.stack((X, Y), dim=0).shape)
print('dim1 the result is: \n', torch.stack((X, Y), dim=1), torch.stack((X, Y), dim=1).shape)
print('dim2 the result is: \n', torch.stack((X, Y), dim=2), torch.stack((X, Y), dim=2).shape)


def corr2d_multi_in_out(X, K):
    # 对K的第0维遍历，每次同输入X做互相关计算。所有结果使用stack函数合并在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K])


# 将核数组K同K+1（K中每个元素加一）和K+2连结在一起来构造一个输出通道数为3的卷积核
K = torch.stack([K, K + 1, K + 2])
print(K.shape)  # torch.Size([3, 2, 2, 2])

corr2d_multi_in_out(X, K)

