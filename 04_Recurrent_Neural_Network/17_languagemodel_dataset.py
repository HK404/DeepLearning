import torch
import random
import zipfile
import sys
sys.path.append("../..")

# 读取周杰伦歌词数据集前40字
with zipfile.ZipFile('../data/jaychou_lyrics.txt.zip') as zin:
# with zipfile.ZipFile('/home/xiaozhong/桌面/learnPython/data/jaychou_lyrics.txt.zip') as zin:
    with zin.open('jaychou_lyrics.txt') as f:
        corpus_chars = f.read().decode('utf-8')
print(corpus_chars[:40])

# 为了打印方便，我们把换行符替换成空格，然后仅使用前1万个字符来训练模型
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[0:10000]

# 建立字符索引
# set([iterable])建立一个无序无重复元素的集合
idx_to_char = list(set(corpus_chars))
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
vocab_size = len(char_to_idx)
vocab_size # 1027

# 将训练数据集中每个字符转化为索引，并打印前20个字符及其对应的索引
corpus_indices = [char_to_idx[char] for char in corpus_chars]
# print(corpus_indices)
sample = corpus_indices[:20]
# str.join(sequence)返回通过指定字符连接序列中元素后生成的新字符串
print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
print('indices:', sample)

# 时序数据采样有两种方式：随机采样，相邻采样
# 时序数据随机采样
def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    # 减1是因为输出的索引x是相应输入的索引y加1
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    # 返回从pos开始的长为num_steps的序列
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(Y, dtype=torch.float32, device=device)

# 时序数据相邻采样
def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus_indices = torch.tensor(corpus_indices, dtype=torch.float32, device=device)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size*batch_len].view(batch_size, batch_len)
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y
