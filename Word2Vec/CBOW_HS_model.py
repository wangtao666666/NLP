# -*- coding: utf-8 -*-

# @Time : 2020/9/20 上午11:36
# @Author : TaoWang
# @Description :

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CBOWModel(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(CBOWModel, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.u_embeddings = nn.Embedding(2*self.vocab_size-1, self.emb_size, sparse=True)
        self.w_embeddings = nn.Embedding(2*self.vocab_size-1, self.emb_size, sparse=True)
        self._init_embedding()  # 初始化

    def _init_embedding(self):
        int_range = 0.5 / self.emb_size
        self.u_embeddings.weight.data.uniform_(-int_range, int_range)
        self.w_embeddings.weight.data.uniform_(-0, 0)

    def compute_context_matrix(self, u):
        pos_u_emb = []  # 上下文embedding
        for per_Xw in u:
            # 上下文矩阵的第一维不同词值不同，如第一个词上下文为c，第二个词上下文为c+1，需要统一化
            per_u_emb = self.u_embeddings(torch.LongTensor(per_Xw).cpu())  # 对上下文每个词转embedding
            per_u_numpy = per_u_emb.data.cpu().numpy()  # 转回numpy，好对其求和
            per_u_numpy = np.sum(per_u_numpy, axis=0)
            per_u_list = per_u_numpy.tolist()  # 为上下文词向量Xw的值
            pos_u_emb.append(per_u_list)  # 放回数组
        pos_u_emb = torch.FloatTensor(pos_u_emb).cpu()  # 转为tensor 大小 [ mini_batch_size * emb_size ]
        return pos_u_emb

    def forward(self, pos_u, pos_w, neg_u, neg_w):
        '''
        :param pos_u: 正样本的周围词，每个样本的周围词是一个list
        :param pos_w: 正样本的中心词
        :param neg_u: 负样本的周围词，每个样本的周围词是一个list
        :param neg_w: 负样本的中心词
        :return:
        '''
        pos_u_emb = self.compute_context_matrix(pos_u)
        pos_w_emb = self.w_embeddings(torch.LongTensor(pos_w).cpu())
        neg_u_emb = self.compute_context_matrix(neg_u)
        neg_w_emb = self.w_embeddings(torch.LongTensor(neg_w).cpu())

        # 计算梯度上升（ 结果 *（-1） 即可变为损失函数 ->可使用torch的梯度下降）
        score_1 = torch.mul(pos_u_emb, pos_w_emb)  # Xw.T * θu
        score_2 = torch.sum(score_1, dim=1)  # 点积和
        score_3 = F.logsigmoid(score_2)  # log (1-sigmoid (Xw.T * θw))
        neg_score_1 = torch.mul(neg_u_emb, neg_w_emb)  # Xw.T * θw
        neg_score_2 = torch.sum(neg_score_1, dim=1)  # 点积和
        neg_score_3 = F.logsigmoid(-neg_score_2)  # ∑neg(w) [log sigmoid (-Xw.T * θneg(w))]
        # L = log sigmoid (Xw.T * θw) + logsigmoid (-Xw.T * θw)
        loss = torch.sum(score_3) + torch.sum(neg_score_3)
        return -1 * loss

    # 存储embedding
    def save_embedding(self, id2word_dict, file_name):
        embedding = self.u_embeddings.weight.data.cpu().numpy()
        file_output = open(file_name, 'w')
        file_output.write('%d %d\n' % (self.vocab_size, self.emb_size))
        for id, word in id2word_dict.items():
            e = embedding[id]
            e = ' '.join(map(lambda x: str(x), e))
            file_output.write('%s %s\n' % (word, e))


