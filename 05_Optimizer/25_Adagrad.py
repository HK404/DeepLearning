import math
import torch
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

features, labels = d2l.get_data_ch7()

# 从零开始实现
def init_adagrad_states():
    s_w = torch.zeros((features.shape[1], 1), dtype=torch.float32)
    s_b = torch.zeros(1, dtype=torch.float32)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        s.data += (p.grad.data**2)
        p.data -= hyperparams['lr'] * p.grad.data / torch.sqrt(s + eps)

d2l.train_ch7(adagrad, init_adagrad_states(), {'lr': 0.1}, features, labels)

# PyTorch实现
d2l.train_pytorch_ch7(torch.optim.Adagrad, {'lr': 0.1}, features, labels)
