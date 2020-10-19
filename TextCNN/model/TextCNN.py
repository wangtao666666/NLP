# -*- coding: utf-8 -*-

# @Time : 2020/10/15 上午11:05
# @Author : TaoWang
# @Description :

import torch.nn.functional as F
import torch.nn as nn
import torch
from .BasicModule import BasicModule


class TextCNN(BasicModule):

    def __init__(self, config):

        super(TextCNN, self).__init__()

        """ 嵌入层"""
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed_size)

        """ 卷积层 """
        if config.cuda:
            self.convs = [nn.Conv1d(config.embed_size, config.filter_num, filter_size).cuda()
                          for filter_size in config.filter_size]
        else:
            self.convs = [nn.Conv1d(config.embed_size, config.filter_num, filter_size)
                          for filter_size in config.filter_size]

        """ Dropout层 """
        self.dropout = nn.Dropout(config.dropout)

        """ 分类层 """
        self.fc = nn.Linear(config.filter_num*len(config.filters), config.label_num)

    def conv_and_pool(self, x, conv):
        """
        :param x:
        :param conv:
        :return:
        """
        x = F.relu(conv(x))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)

        return x

    def forward(self, x):
        """
        :param x:
        :return:
        """
        out = self.embedding(x)
        out = out.transpose(1, 2).contiguous()
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)

        return out

