# -*- coding: utf-8 -*-

# @Time : 2020/9/20 上午9:44
# @Author : TaoWang
# @Description :

import sys
sys.path.append("..")
import numpy as np
from collections import deque
from Huffman_tree import HuffmanTree

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


word2id_dict, id2word_dict, wordid_frequency_dict = create_dict(20)
wordId_list = create_wordId_list(word2id_dict)
index = 0
word_pairs_queue = deque()
result_pairs = create_batch_pairs(32, 3, index, word_pairs_queue, wordId_list)
huffman_tree = HuffmanTree(wordid_frequency_dict)
huffman_pos_path, huffman_neg_path = huffman_tree.get_all_pos_and_neg_path()

def get_pairs(pos_pairs):
    """
    :param pos_pairs:
    :return:
    """
    neg_word_pair, pos_word_pair = [], []
    for pair in pos_pairs:
        pos_word_pair += zip([pair[0]] * len(huffman_pos_path[pair[1]]), huffman_pos_path[pair[1]])
        neg_word_pair += zip([pair[0]] * len(huffman_neg_path[pair[1]]), huffman_neg_path[pair[1]])

    return pos_word_pair, neg_word_pair

pos_word_pair, neg_word_pair = get_pairs(result_pairs)

print(pos_word_pair)
print(neg_word_pair)

class InputData:
    def __init__(self, input_file_name, min_count):
        self.input_file_name = input_file_name
        self.index = 0
        self.input_file = open(self.input_file_name)  # 数据文件
        self.min_count = min_count  # 要淘汰的低频数据的频度
        self.wordId_frequency_dict = dict()  # 词id-出现次数 dict
        self.word_count = 0  # 单词数（重复的词只算1个）
        self.word_count_sum = 0  # 单词总数 （重复的词 次数也累加）
        self.sentence_count = 0  # 句子数
        self.id2word_dict = dict()  # 词id-词 dict
        self.word2id_dict = dict()  # 词-词id dict
        self._init_dict()  # 初始化字典
        self.huffman_tree = HuffmanTree(self.wordId_frequency_dict)  # 霍夫曼树
        self.huffman_pos_path, self.huffman_neg_path = self.huffman_tree.get_all_pos_and_neg_path()
        self.word_pairs_queue = deque()
        # 结果展示
        self.get_wordId_list()
        print('Word Count is:', self.word_count)
        print('Word Count Sum is', self.word_count_sum)
        print('Sentence Count is:', self.sentence_count)
        print('Tree Node is:', len(self.huffman_tree.huffman))

    def _init_dict(self):
        word_freq = dict()
        # 统计 word_frequency
        for line in self.input_file:
            line = line.strip().split(' ')  # 去首尾空格
            self.word_count_sum += len(line)
            self.sentence_count += 1
            for i,word in enumerate(line):
                if i%1000000==0:
                    print (i,len(line))
                try:
                    word_freq[word] += 1
                except:
                    word_freq[word] = 1
        word_id = 0
        # 初始化 word2id_dict,id2word_dict, wordId_frequency_dict字典
        for per_word, per_count in word_freq.items():
            if per_count < self.min_count:  # 去除低频
                self.word_count_sum -= per_count
                continue
            self.id2word_dict[word_id] = per_word
            self.word2id_dict[per_word] = word_id
            self.wordId_frequency_dict[word_id] = per_count
            word_id += 1
        self.word_count = len(self.word2id_dict)
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
    # 获取mini-batch大小的 正采样对 (w,v) w为目标词id,v为上下文中的一个词的id。上下文步长为window_size，即2c = 2*window_size
    def get_batch_pairs(self,batch_size,window_size):
        while len(self.word_pairs_queue) < batch_size:
            for _ in range(1000):
                if self.index == len(self.wordId_list):
                    self.index = 0
                for i in range(max(self.index - window_size, 0),
                                         min(self.index + window_size + 1,len(self.wordId_list))):
                    wordId_w = self.wordId_list[self.index]
                    wordId_v = self.wordId_list[i]
                    if self.index == i:  # 上下文=中心词 跳过
                        continue
                    self.word_pairs_queue.append((wordId_w, wordId_v))
                self.index+=1
        result_pairs = []  # 返回mini-batch大小的正采样对
        for _ in range(batch_size):
            result_pairs.append(self.word_pairs_queue.popleft())
        return result_pairs

    def get_pairs(self, pos_pairs):
        neg_word_pair = []
        pos_word_pair = []
        for pair in pos_pairs:
            pos_word_pair += zip([pair[0]] * len(self.huffman_pos_path[pair[1]]), self.huffman_pos_path[pair[1]])
            neg_word_pair += zip([pair[0]] * len(self.huffman_neg_path[pair[1]]), self.huffman_neg_path[pair[1]])
        return pos_word_pair, neg_word_pair


    # 估计数据中正采样对数，用于设定batch
    def evaluate_pairs_count(self, window_size):
        return self.word_count_sum * (2 * window_size - 1) - (self.sentence_count - 1) * (1 + window_size) * window_size

