# -*- coding: utf-8 -*-

# @Time : 2020/10/15 上午9:17
# @Author : TaoWang
# @Description : 数据预处理


from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from torch.utils import data
import numpy as np
import random
import os


class MR_Dataset(data.Dataset):

    def __init__(self, state="train", k=0, embedding_type="word2vec"):

        self.path = os.path.abspath(".")
        if "data" not in self.path:
            self.path = self.path + "/data"

        pos_data = open(self.path + "/MR/rt-polarity.pos", errors="ignore").readlines()
        neg_data = open(self.path + "/MR/rt-polarity.neg", errors="ignore").readlines()
        datas = [item.split() for item in pos_data + neg_data]

        """ 获取最长句子参数 """
        max_sentence_length = max([len(sentence) for sentence in datas])
        labels = [1] * len(pos_data) + [0] * len(neg_data)

        """ 构建word2id """
        word2id = {"<pad>": 0}
        for i, sentence in enumerate(datas):
            for j, word in enumerate(sentence):
                if word not in word2id:
                    word2id[word] = len(word2id)
                datas[i][j] = word2id[word]

            """padding 成相同句子长度 矩阵 """
            datas[i] = datas[i] + [0] * (max_sentence_length - len(datas[i]))

        self.n_vocab, self.word2id = len(word2id), word2id

        if embedding_type == "word2vec":
            self.get_word2vec()
        elif embedding_type == "glove":
            self.get_glove_embedding()
        else:
            pass

        """ 打乱数据集 """
        c = list(zip(datas, labels))
        random.seed(1)
        random.shuffle(c)
        datas[:], labels[:] = zip(*c)

        """ 十折交叉 构建训练、测试、验证集合 """
        if state == "train":
            self.datas = datas[:int(k * len(datas)/10)] + datas[int((k + 1) * len(datas) / 10):]
            self.labels = labels[:int(k * len(datas) / 10)] + labels[int((k + 1) * len(labels) / 10):]
            self.datas = np.array(self.datas[0:int(0.9 * len(self.datas))])
            self.labels = np.array(self.labels[0:int(0.9 * len(self.labels))])
        elif state == "valid":
            self.datas = datas[:int(k * len(datas) / 10)] + datas[int((k+1) * len(datas) / 10):]
            self.labels = labels[:int(k * len(datas) / 10)] + labels[int((k+1) * len(labels) / 10):]
            self.datas = np.array(self.datas[int(0.9 * len(self.datas)):])
            self.labels = np.array(self.labels[int(0.9 * len(self.labels)):])
        elif state == "test":
            self.datas = np.array(datas[int(k * len(datas) / 10): int((k+1) * len(datas) / 10)])
            self.labels = np.array(labels[int(k * len(datas) / 10): int((k+1) * len(datas) / 10)])

    def __getitem__(self, index):

        return self.datas[index], self.labels[index]

    def __len__(self):

        return len(self.datas)

    def get_word2vec(self):
        """
        :Desc: 生成word2vec词向量
        :return: 根据词表生成词向量
        """
        if not os.path.exists(self.path + "/word2vec_embedding_mr.npy"):
            print("Reading Word2vec Embedding.....")
            word_model = KeyedVectors.load_word2vec_format(self.path + "/GoogleNews-vectors-negative300.bin.gz",
                                                           binary=True)

            """ 如果词在预训练的模型中可以找到词向量，用预训练好的词向量，否则构造和预训练词向量矩阵相同的均值和方差的随机初始化矩阵"""
            word_embed_list = []
            for word, index in self.word2id.items():
                try:
                    word_embed_list.append(word_model.get_vector(word))
                except:
                    pass

            mean, std = np.mean(np.array(word_embed_list)), np.std(np.array(word_embed_list))
            vocab_size, embed_size = self.n_vocab, 300
            """ 正态分布初始化矩阵 """
            embedding_weights = np.random.normal(mean, std, [vocab_size, embed_size])
            for word, index in self.word2id.items():
                try:
                    embedding_weights[index, :] = word_model.get_vector(word)
                except:
                    pass

            """ 保存生成的词向量"""
            np.save(self.path + "/word2vec_embedding_mr.npy", embedding_weights)
        else:
            """ 加载词向量 """
            embedding_weights = np.load(self.path + "/word2vec_embedding_mr.npy")

        self.weights = embedding_weights

    def get_glove_embedding(self):
        """
        :desc 生成glove词向量
        :return: 根据词表生成词向量
        """
        if not os.path.exists(self.path + "/glove_embedding_mr.npy"):
            if not os.path.exists(self.path + "/glove_word2vec.txt"):
                glove_file = datapath(self.path + "/glove.840B.300d.txt")
                tmp_file = get_tmpfile(self.path + "/glove_word2vec.txt")
                glove2word2vec(glove_file, tmp_file)
            else:
                tmp_file = get_tmpfile(self.path + "/glove_word2vec.txt")

            print("Reading Glove Embedding....")
            glove_model = KeyedVectors.load_word2vec_format(tmp_file)
            glove_word_list = []
            for word, index in self.word2id.items():
                try:
                    glove_word_list.append(glove_model.get_vector(word))
                except:
                    pass

            glove_mean, glove_std = np.mean(np.array(glove_word_list)), np.std(np.array(glove_word_list))

            vocab_size, embed_size = self.n_vocab, 300
            embedding_weights = np.random.normal(glove_mean, glove_std, [vocab_size, embed_size])

            for word, index in self.word2id.items():
                try:
                    embedding_weights[index, :] = glove_model.get_vector(word)
                except:
                    pass

            np.save(self.path + "/glove_embedding_mr.npy", embedding_weights)

        else:
            embedding_weights = np.load(self.path + "/glove_embedding_mr.npy")

        self.weights = embedding_weights


if __name__ == "__main__":
    mr_train_dataset = MR_Dataset()
    print(mr_train_dataset.path)
    print(mr_train_dataset.__len__())
    print(mr_train_dataset[0])

    mr_valid_dataset = MR_Dataset("valid")
    print(mr_valid_dataset.__len__())
    print(mr_valid_dataset[0])

    mr_test_valid = MR_Dataset("test")
    print(mr_test_valid.__len__())
    print(mr_test_valid[0])
