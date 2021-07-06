import torch

print(torch.__version__)

print(torch.cuda.is_available())

a = torch.ones((3,1))
a = a.cuda(0)
b = torch.ones((3,1)).cuda(0)
print(a+b)
# 不能在不同平台使用
c = torch.ones((3,1))
print(a+c)