# -*- coding: utf-8 -*-

# @Time : 2020/9/19 下午4:31
# @Author : TaoWang
# @Description : Skip Gram模型 + 负采样

import numpy as np
from collections import deque

# ************ 数据处理部分(无监督-选取中心词和周围词作为正样本，随机采取几个词作为负样本)************

def create_dict(min_count):
    """
    :param min_count: 词频出现的最少个数
    :return:
    """
    input_file = open("text8.txt", "r", encoding='utf-8')
    word_count_sum, sentence_count, word2id_dict, id2word_dict, wordid_frequency_dict, word_freq = 0, 0, {}, {}, {}, {}

    for line in input_file:

        line = line.strip().split()
        word_count_sum += len(line)
        sentence_count += 1

        for i, word in enumerate(line):
            if word_freq.get(word) is None:
                word_freq[word] = 1
            else:
                word_freq[word] += 1

        for i, word in enumerate(word_freq):
            if word_freq[word] < min_count:
                word_count_sum -= word_freq[word]
                continue

            word2id_dict[word] = len(word2id_dict)
            id2word_dict[len(id2word_dict)] = word
            wordid_frequency_dict[len(word2id_dict)-1] = word_freq[word]  # 因为上述添加一个词进去，所以需要索引减1

    return word2id_dict, id2word_dict, wordid_frequency_dict

def create_wordId_list(word2id_dict):
    """
    :param word2id_dict:  word -> id 的dict
    :return:
    """

    input_file = open("text8.txt", encoding='utf-8')
    sentence = input_file.readline()

    wordId_list = []  #一句中所有word对应的id
    sentences = sentence.strip().split(' ')

    for i, word in enumerate(sentences):
        try:
            word_id = word2id_dict[word]
            wordId_list.append(word_id)
        except:
            continue

    return wordId_list

def create_batch_pairs(batch_size, window_size, index, word_pairs_queue, wordId_list):
    """
    :desc 生成正样本 中心词 和 周围词
    :param batch_size:
    :param window_size: 窗口大小，周围词
    :param index: 读取到哪里
    :param word_pairs_queue:  一个队列
    :return:
    """
    while len(word_pairs_queue) < batch_size:
        if index == len(wordId_list):
            index = 0
        for _ in range(1000):
            for i in range(max(index - window_size, 0), min(index + window_size + 1, len(wordId_list))):
                # 左侧和右侧wordId_list特殊情况
                wordid_w = wordId_list[index]  # 中心词
                wordid_v = wordId_list[i]  # 周围词
                if index == i:  # 周围词等于中心词，跳过
                    continue
                word_pairs_queue.append((wordid_w, wordid_v))
            index += 1

    """ 返回batch大小正样本对 """
    result_pairs = []
    for _ in range(batch_size):
        result_pairs.append(word_pairs_queue.popleft())

    return result_pairs

def sample_table(wordid_frequency_dict):
    """
    : desc 按照采样频率，抽样数据
    :param wordid_frequency_dict:
    :return:
    """
    sample_table, sample_table_size = [], 1e8  # 设置词表大小1e8
    pow_frequency = np.array(list(wordid_frequency_dict.values())) ** 0.75 # 采样频率
    word_pow_sum = sum(pow_frequency)
    ratio_array = pow_frequency / word_pow_sum
    word_count_list = np.round(ratio_array * sample_table_size)  # 该词在词表中出现的个数 == 词的频率 ** 词表的大小

    for word_index, word_freq in enumerate(word_count_list):
        sample_table += [word_index] * int(word_freq)

    sample_table = np.array(sample_table)

    return sample_table

def negative_sampling(positive_pairs, neg_count):
    """
    : desc 随机采样构建负样本，一个中心词，随机采取neg_count个词语作为一个pairs作为负样本
    :param positive_pairs: 正样本对
    :param neg_count: 负样本对的列数
    :return:
    """
    neg_v = np.random.choice(sample_table, size=(len(positive_pairs), neg_count)).tolist()

    return neg_v

# min_count = 3
# word2id_dict, id2word_dict, wordid_frequency_dict = create_dict(1000)
# wordId_list = create_wordId_list(word2id_dict)
# index, word_pairs_queue = 0, deque()
# result_pairs = create_batch_pairs(32, 3, index, word_pairs_queue, wordId_list)
# sample_table = sample_table(wordid_frequency_dict)
# print(result_pairs)
# neg_v = negative_sampling(result_pairs, 3)
# print(neg_v)

class InputData:
    def __init__(self,input_file_name,min_count):
        self.input_file_name = input_file_name
        self.index = 0
        self.input_file = open(self.input_file_name,"r",encoding="utf-8")
        self.min_count = min_count
        self.wordid_frequency_dict = dict()
        self.word_count = 0
        self.word_count_sum = 0
        self.sentence_count = 0
        self.id2word_dict = dict()
        self.word2id_dict = dict()
        self._init_dict()  # 初始化字典
        self.sample_table = []
        self._init_sample_table()  # 初始化负采样映射表
        self.get_wordId_list()
        self.word_pairs_queue = deque()
        # 结果展示
        print('Word Count is:', self.word_count)
        print('Word Count Sum is', self.word_count_sum)
        print('Sentence Count is:', self.sentence_count)
    def _init_dict(self):
        word_freq = dict()
        for line in self.input_file:
            line = line.strip().split()
            self.word_count_sum +=len(line)
            self.sentence_count +=1
            for i,word in enumerate(line):
                if i%1000000==0:
                    print (i,len(line))
                if word_freq.get(word)==None:
                    word_freq[word] = 1
                else:
                    word_freq[word] += 1
        for i,word in enumerate(word_freq):
            if i % 100000 == 0:
                print(i, len(word_freq))
            if word_freq[word]<self.min_count:
                self.word_count_sum -= word_freq[word]
                continue
            self.word2id_dict[word] = len(self.word2id_dict)
            self.id2word_dict[len(self.id2word_dict)] = word
            self.wordid_frequency_dict[len(self.word2id_dict)-1] = word_freq[word]
        self.word_count =len(self.word2id_dict)
    def _init_sample_table(self):
        sample_table_size = 1e8
        pow_frequency = np.array(list(self.wordid_frequency_dict.values())) ** 0.75
        word_pow_sum = sum(pow_frequency)
        ratio_array = pow_frequency / word_pow_sum
        word_count_list = np.round(ratio_array * sample_table_size)
        for word_index, word_freq in enumerate(word_count_list):
            self.sample_table += [word_index] * int(word_freq)
        self.sample_table = np.array(self.sample_table)
        np.random.shuffle(self.sample_table)
    def get_wordId_list(self):
        self.input_file = open(self.input_file_name, encoding="utf-8")
        sentence = self.input_file.readline()
        wordId_list = []  # 一句中的所有word 对应的 id
        sentence = sentence.strip().split(' ')
        for i,word in enumerate(sentence):
            if i%1000000==0:
                print (i,len(sentence))
            try:
                word_id = self.word2id_dict[word]
                wordId_list.append(word_id)
            except:
                continue
        self.wordId_list = wordId_list
    def get_batch_pairs(self,batch_size,window_size):
        while len(self.word_pairs_queue) < batch_size:
            for _ in range(1000):
                if self.index == len(self.wordId_list):
                    self.index = 0
                wordId_w = self.wordId_list[self.index]
                for i in range(max(self.index - window_size, 0),
                                         min(self.index + window_size + 1,len(self.wordId_list))):

                    wordId_v = self.wordId_list[i]
                    if self.index == i:  # 上下文=中心词 跳过
                        continue
                    self.word_pairs_queue.append((wordId_w, wordId_v))
                self.index+=1
        result_pairs = []  # 返回mini-batch大小的正采样对
        for _ in range(batch_size):
            result_pairs.append(self.word_pairs_queue.popleft())
        return result_pairs


    # 获取负采样 输入正采样对数组 positive_pairs，以及每个正采样对需要的负采样数 neg_count 从采样表抽取负采样词的id
    # （假设数据够大，不考虑负采样=正采样的小概率情况）
    def get_negative_sampling(self, positive_pairs, neg_count):
        neg_v = np.random.choice(self.sample_table, size=(len(positive_pairs), neg_count)).tolist()
        return neg_v

    # 估计数据中正采样对数，用于设定batch
    def evaluate_pairs_count(self, window_size):
        return self.word_count_sum * (2 * window_size) - self.sentence_count * (
                    1 + window_size) * window_size













