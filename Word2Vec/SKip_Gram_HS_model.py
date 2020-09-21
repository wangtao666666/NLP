# -*- coding: utf-8 -*-

# @Time : 2020/9/20 上午10:56
# @Author : TaoWang
# @Description :

import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGramModel(nn.Module):

    def __init__(self, emb_size, emb_dimension):
        """
        :param emb_size:
        :param emb_dimension:
        """
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.w_embedding = nn.Embedding(2*emb_size - 1, emb_dimension, sparse=True)
        self.v_embedding = nn.Embedding(2*emb_size - 1, emb_dimension, sparse=True)
        self._init_emb()

    def _init_emb(self):
        """
        :return:
        """
        initrange = 0.5/self.emb_dimension
        self.w_embedding.weight.data.uniform_(-initrange, initrange)
        self.v_embedding.weight.data.uniform_(-0, 0)

    def forward(self, pos_w, pos_v, neg_w, neg_v):
        """
        :param pos_w:
        :param pos_v:
        :param neg_w:
        :param neg_v:
        :return:
        """
        emb_w = self.w_embedding(torch.LongTensor(pos_w))
        neg_emb_w = self.v_embedding(torch.LongTensor(neg_w))

        emb_v = self.v_embedding(torch.LongTensor(pos_v))
        neg_emb_v = self.v_embedding(torch.LongTensor(neg_v))

        score = torch.mul(emb_w, emb_v).squeeze()
        score = torch.sum(score, dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = F.logsigmoid(score)

        neg_socre = torch.mul(neg_emb_w, neg_emb_v).squeeze()
        neg_socre = torch.sum(neg_socre, dim=1)
        neg_score = torch.clamp(neg_socre, max=10, min=-10)
        neg_socre = F.logsigmoid(-neg_score)

        loss = -1 * (torch.sum(score) + torch.sum(neg_socre))

        return loss

    def save_embedding(self, id2word, file_name):
        """
        :param id2word:
        :param file_name:
        :return:
        """
        embedding = self.w_embedding.weight.data.cpu().numpy()
        fout = open(file_name, "w")
        fout.write("%d %d\n" %(len(id2word), self.emb_dimension))
        for wid, w in id2word.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x)), e)
            fout.write("%s %s\n" %(w, e))

model = SkipGramModel(100,10)
id2word = dict()
for i in range(100):
    id2word[i] = str(i)

pos_w = [0, 0, 1, 1, 1]
pos_v = [1, 2, 0, 2, 3]
neg_w = [0, 0, 1, 1, 1]
neg_v = [54, 55, 61, 71, 82]

model.forward(pos_w, pos_v, neg_w, neg_v)

print(model.v_embedding.weight.data.cpu().numpy().shape)





