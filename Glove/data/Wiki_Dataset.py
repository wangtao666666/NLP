# -*- coding: utf-8 -*-

# @Time : 2020/10/11 上午10:59
# @Author : TaoWang
# @Description : 数据预处理

from torch.utils import data
import os
import numpy as np
import pickle


class Wiki_Dataset(data.DataLoader):

    def __init__(self, min_count, window_size):
        self.min_count = min_count
        self.window_size = window_size
        self.datas, self.labels = self.get_co_occur(data)

    def __getitem__(self, index):
        return self.datas[index], self.labels[index]

    def __len__(self):
        return len(self.datas)

    def read_data(self):
        data = open(self.path + "/text8.txt").read()
        data = data.split()
        self.word2freq = {}
        for word in data:
            if self.word2freq.get(word) != None:
                self.word2freq[word] += 1
            else:
                self.word2freq[word] = 1
        word2id = {}
        for word in self.word2freq:
            if self.word2freq[word] < self.min_count:
                continue
            else:
                if word2id.get(word) == None:
                    word2id[word] = len(word2id)
        self.word2id = word2id
        print(len(self.word2id))
        return data

    def get_co_occur(self, data):
        self.path = os.path.abspath('..')
        if "data" not in self.path:
            self.path += "/data"
        if not os.path.exists(self.path + "/label.npy"):
            print("Processing data...")
            data = self.read_data()
            print("Generating co-occurrences...")
            vocab_size = len(self.word2id)
            comat = np.zeros((vocab_size, vocab_size))
            for i in range(len(data)):
                if i % 1000000 == 0:
                    print(i, len(data))
                if self.word2id.get(data[i]) == None:
                    continue
                w_index = self.word2id[data[i]]
                for j in range(max(0, i - self.window_size), min(len(data), i + self.window_size + 1)):
                    if self.word2id.get(data[j]) == None or i == j:
                        continue
                    u_index = self.word2id[data[j]]
                    comat[w_index][u_index] += 1
            coocs = np.transpose(np.nonzero(comat))
            labels = []
            for i in range(len(coocs)):
                if i % 100000 == 0:
                    print(i, len(coocs))
                labels.append(comat[coocs[i][0]][coocs[i][1]])
            labels = np.array(labels)
            np.save(self.path + "/data.npy", coocs)
            np.save(self.path + "/label.npy", labels)
            pickle.dump(self.word2id, open(self.path + "/word2id", "wb"))
        else:
            coocs = np.load(self.path + "/data.npy")
            labels = np.load(self.path + "/label.npy")
            self.word2id = pickle.load(open(self.path + "/word2id", "rb"))
        return coocs, labels


if __name__ == "__main__":
    wiki_dataset = Wiki_Dataset(min_count=50, window_size=2)
    print(wiki_dataset.datas.shape)
    print(wiki_dataset.labels.shape)
    print(wiki_dataset.labels[0:100])
