import torch
import torchvision
import numpy as np
import sys

sys.path.append("../..")  # 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l

# 从零开始实现softmax回归
# 1.获取和读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 2.初始化模型参数
# 已知每个样本输入是高和宽均为28像素的图像。模型的输入向量的长度是 28×28=78428×28=784
# 该向量的每个元素对应图像中每个像素。由于图像有10个类别，单层神经网络输出层的输出个数为10，
# 因此softmax回归的权重和偏差参数分别为784×10和1×10的矩阵。
num_inputs = 784
num_outputs = 10

W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)

# 需要求梯度
W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


# 定义softmax函数，返回预测概率，矩阵X的行数是样本数，列数是输出个数。
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制


# 3.定义模型
def net(X):
    # -1表示一个不确定的数，即不知道几行
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)


# 4.定义损失函数(交叉熵损失函数)
def cross_entropy(y_hat, y):
    # gather以dim维度0为按列索引，1为按行索引，按矩阵y.view索引，例如：
    # y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
    # y = torch.LongTensor([0, 2])
    # y_hat.gather(1, y.view(-1, 1))
    # 输出
    # tensor([[0.1000],
    #         [0.5000]])
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))


# 5.定义准确率函数
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()


# y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# y = torch.LongTensor([0, 2])
# print(cross_entropy(y_hat,y))
# print(accuracy(y_hat,y))

# 看一下test_iter的结构
# for i,(X,y) in enumerate(test_iter):
#     if(i==1):
#         break
#     print(X,y)
#     print(y.shape)
#     print(y.shape[0])


# 6.训练模型
num_epochs, lr = 5, 0.1


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()  # “softmax回归的简洁实现”一节将用到

            # item()返回张量中的元素
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = d2l.evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)


# 7.预测
X, y = iter(test_iter).next()

true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[0:9], titles[0:9])
