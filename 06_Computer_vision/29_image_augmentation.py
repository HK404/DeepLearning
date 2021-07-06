# 使用图像增广来避免过拟合
import time
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image

import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# from matplotlib import pyplot as plt
# 读取一张图像
d2l.set_figsize()
img = Image.open('../data/img/cat1.jpg')
#d2l.plt.imshow(img)
#d2l.plt.show()

# 绘图函数
def show_images(imgs, num_rows, num_cols, scale=2 ,):
    figsize = (num_cols * scale, num_rows * scale)
    # subplots画多个子图
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    d2l.plt.show()
    return axes

# 为了方便观察图像增广的效果，接下来我们定义一个辅助函数
# 这个函数对输入图像img多次运行图像增广方法aug并展示所有的结果
def apply(img, aug, num_rows=2, num_cols=4, scale=3):
    # 随机水平翻转2x4次(得到8个图像)(Y:list)
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scale, )

# 1、翻转和裁剪
# 使用水平翻转策略
apply(img, torchvision.transforms.RandomHorizontalFlip())

# 上下翻转不如左右翻转通用。但是至少对于样例图像，上下翻转不会造成识别障碍
# 使用垂直翻转策略
apply(img, torchvision.transforms.RandomVerticalFlip())

# 通过对图像随机裁剪来让物体以不同的比例出现在图像的不同位置，这同样能够降低模型对目标位置的敏感性
# 下面代码每次随机裁剪出一块面积为原面积10%∼100%的区域，且该区域的宽和高之比随机取自0.5∼2，然后再将该区域的宽和高分别缩放到200像素
shape_aug = torchvision.transforms.RandomResizedCrop(200, scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)

# 2、改变颜色
# 我们可以从4个方面改变图像的颜色：亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue）
# 接下来我们将图像的亮度随机变化为原图亮度的50%（1−0.5）∼150%（1+0.5）
apply(img, torchvision.transforms.ColorJitter(brightness=0.5))
apply(img, torchvision.transforms.ColorJitter(hue=0.5))
apply(img, torchvision.transforms.ColorJitter(contrast=0.5))

# 也可以同时设置如何随机变化图像的亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue）
color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)

# 3、叠加多个图像增广方法
# 通过Compose实例将上面定义的多个图像增广方法叠加起来
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)

# 4、使用图像增广训练模型（CIFAR-10数据集）
all_imges = torchvision.datasets.CIFAR10(train=True, root="~/Datasets/CIFAR", download=False)
# all_imges的每一个元素都是(image, label)
show_images([all_imges[i][0] for i in range(32)], 4, 8, scale=1.6);

# 仅使用简单的水平翻转
flip_aug = torchvision.transforms.Compose([
     torchvision.transforms.Resize(96),
     torchvision.transforms.RandomHorizontalFlip(),
     # 使用ToTensor将小批量图像转成PyTorch需要的格式，即形状为(批量大小, 通道数, 高, 宽)、值域在0到1之间且类型为32位浮点数
     torchvision.transforms.ToTensor()])

no_aug = torchvision.transforms.Compose([
     torchvision.transforms.Resize(96),
     torchvision.transforms.ToTensor()])

# 读取数据函数
num_workers = 0 if sys.platform.startswith('win32') else 4
def load_cifar10(is_train, augs, batch_size, root="~/Datasets/CIFAR"):
    dataset = torchvision.datasets.CIFAR10(root=root, train=is_train, transform=augs, download=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers)

# 使用GPU训练模型
def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = d2l.evaluate_accuracy3(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

def train_with_data_aug(train_augs, test_augs, lr=0.001):
    batch_size, net = 256, d2l.resnet18()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss()
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    train(train_iter, test_iter, net, loss, optimizer, device, num_epochs=10)

train_with_data_aug(flip_aug, no_aug)
