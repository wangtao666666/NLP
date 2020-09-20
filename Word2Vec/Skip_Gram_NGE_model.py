# -*- coding: utf-8 -*-

# @Time : 2020/9/19 下午8:48
# @Author : TaoWang
# @Description :

# ************ 构建模型部分 ************

import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGramModel(nn.Module):

    def __init__(self, vocab_size, embed_size):
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.w_embedding = nn.Embedding(vocab_size, embed_size, sparse=True)  # 初始化中心词矩阵
        self.v_embedding = nn.Embedding(vocab_size, embed_size, sparse=True)  # 初始化周围词矩阵
        self._init_emb()  # 对两个矩阵进行初始化

    def _init_emb(self):
        """
        : desc 初始化embedding
        :return:
        """
        init_range = 0.5 / self.embed_size
        self.w_embedding.weight.data.uniform_(-init_range, init_range)
        self.v_embedding.weight.data.uniform_(-0, 0)  # 服从均值为0的正态分布

    def forward(self, pos_w, pos_v, neg_v):
        """
        :param pos_w: pos_w = [0, 0, 1, 1, 1]  # pos_w 一系列中心词 batch_size * 1
        :param pos_v: pos_v = [1, 2, 0, 2, 3]  # pos_v 真正的周围词，一个pos_w和一个pos_v组合成一个正样本，也是一个batch_size * 1
        :param neg_v: neg_v = [[23, 42, 32], [32, 24, 53], [32, 24, 53], [32, 24, 53], [32, 24, 53]]
         neg_v 是 一个正样本对应三个随机采样的负样本，三是设置的超参数，batch_size * 3
        :return:
        """
        emb_w = self.w_embedding(torch.LongTensor(pos_w))  # batch_size 转化为 batch_size * embedd_dim
        emb_v = self.v_embedding(torch.LongTensor(pos_v))  # batch_size 转化为 batch_size * embedd_dim

        # batch_size * neg_sampling_number 转化为 batch_size * neg_sampling_number * embed_dim
        neg_emb_v = self.v_embedding(torch.LongTensor(neg_v))

        """ 正样本loss """
        score = torch.mul(emb_w, emb_v)  # 对应元素相乘 batch_size * embedd_dim
        score = torch.sum(score, dim=1)  # 对应行相加，转化为1列
        score = F.logsigmoid(score)

        # emb_w 是 batch_size * embedd_dim 首先转户为3维向量 batch_size * embedd_dim * 1
        # neg_emb_v 是 batch_size * neg_sampling_number * embed_dim
        # 两者相乘之后是 batch_size * neg_sampling_number * 1

        neg_score = torch.bmm(neg_emb_v, emb_w.unsqueeze(2))  # 两个三维向量相乘
        neg_score = F.logsigmoid(-1 * neg_score) # 负样本越小越好，-loss越大越好
        loss = -1 * (torch.sum(score) + torch.sum(neg_score))

        return loss

    def save_embedding(self, id2word, file_name):
        """
        : desc 模型训练完 保存向量
        :param id2word:
        :param file_name:
        :return:
        """
        embedding_1 = self.w_embedding.weight.data.cpu().numpy()
        embedding_2 = self.v_embedding.weight.data.cpu().numpy()
        embedding = (embedding_1 + embedding_2) / 2
        fout = open(file_name, 'w')
        fout.write("%d %d\n" % (len(id2word), self.embed_size))

        for wid, w in id2word.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e)) # 前面是一个词 后面是一个词对应的向量
            fout.write('%s %s\n' % (w, e))


model = SkipGramModel(100, 10)
id2word = dict()
for i in range(100):
    id2word[i] = str(i)

pos_w = [0, 0, 1, 1, 1]
pos_v = [1, 2, 0, 2, 3]
neg_v = [[23, 42, 32], [32, 34, 53], [32, 24, 53], [32, 24, 53], [32, 24, 53]]

print(model.forward(pos_w, pos_v, neg_v))

print(model.v_embedding.weight.data.cpu().numpy().shape)

