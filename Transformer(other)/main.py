import torch
import os
from Transformer import *
from util import *
from train import *
# 1、导入相关的库
# Transformer里面是关于Transformer模型的函数
# util里面是相关的数据读取文件
# train内是相关的训练和测试函数

print(torch.__version__)
print(torch.cuda.is_available())
device = torch.device('cuda')

# 2、设置相关的参数
embedding_size = 32  # token的维度
num_layers = 2       # 编码器和解码器的层数，这里两者层数相同，也可以不同
dropout = 0.05       # 所有层的droprate都相同，也可以不同
batch_size = 64      # 批次
num_steps = 10       # 预测步长
factor = 1           # 学习率因子
warmup = 2000        # 学习率上升步长
lr, num_epochs, ctx = 0.005, 500, device  # 学习率；周期；设备
num_hiddens, num_heads = 64, 4            # 隐层单元的数目——表示FFN中间层的输出维度；attention的数目

# 3、导入文件
# 文件为fra.txt文件
src_vocab, tgt_vocab, train_iter = load_data_nmt(batch_size, num_steps)

# 4、加载模型
# TransformerEncoder为编码器模型
# TransformerDecoder为解码器模型
# transformer为编码器和解码器构成的最终模型
encoder = TransformerEncoder(vocab_size=len(src_vocab),
                             embedding_size=embedding_size,
                             n_layers=num_layers,
                             hidden_size=num_hiddens,
                             num_heads=num_heads,
                             dropout=dropout, )
decoder = TransformerDecoder(vocab_size=len(src_vocab),
                             embedding_size=embedding_size,
                             n_layers=num_layers,
                             hidden_size=num_hiddens,
                             num_heads=num_heads,
                             dropout=dropout, )


class transformer(nn.Module):
    def __init__(self, enc_net, dec_net):
        super(transformer, self).__init__()
        self.enc_net = enc_net  # TransformerEncoder的对象
        self.dec_net = dec_net  # TransformerDecoder的对象

    def forward(self, enc_X, dec_X, valid_length=None, max_seq_len=None):
        """
        enc_X: 编码器的输入
        dec_X: 解码器的输入
        valid_length: 编码器的输入对应的valid_length,主要用于编码器attention的masksoftmax中，
                      并且还用于解码器的第二个attention的masksoftmax中
        max_seq_len:  位置编码时调整sin和cos周期大小的，默认大小为enc_X的第一个维度seq_len
        """

        # 1、通过编码器得到编码器最后一层的输出enc_output
        enc_output = self.enc_net(enc_X, valid_length, max_seq_len)
        # 2、state为解码器的初始状态，state包含两个元素，分别为[enc_output, valid_length]
        state = self.dec_net.init_state(enc_output, valid_length)
        # 3、通过解码器得到编码器最后一层到线性层的输出output，这里的output不是解码器最后一层的输出，而是
        #    最后一层再连接线性层的输出
        output = self.dec_net(dec_X, state)
        return output

model = transformer(encoder, decoder)

# 5、训练模型
model.train()
train(model, train_iter, lr, factor, warmup, num_epochs, ctx)

# 6、测试模型
model.eval()
for sentence in ['Go .', 'Wow !', "I'm OK .", 'I won !','I like study']:
    print(sentence + ' => ' + translate(model, sentence, src_vocab, tgt_vocab, num_steps, ctx))