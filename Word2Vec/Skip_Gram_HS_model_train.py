# -*- coding: utf-8 -*-

# @Time : 2020/9/20 下午1:51
# @Author : TaoWang
# @Description :

import sys
sys.path.append("..")
from SKip_Gram_HS_model import SkipGramModel
from Skip_Gram_HS import InputData
import torch.optim as optim
from tqdm import tqdm
import argumentparser as argumentparser

args = argumentparser.SKip_Gram_HS_ArgumentParser()
WINDOW_SIZE = args.window_size  # 上下文窗口c
BATCH_SIZE = args.batch_size  # mini-batch
MIN_COUNT = args.min_count  # 需要剔除的 低频词 的频
EMB_DIMENSION = args.embed_dimension  # embedding维度
LR = args.learning_rate  # 学习率

class Word2Vec:
    def __init__(self, input_file_name, output_file_name):
        self.output_file_name = output_file_name
        self.data = InputData(input_file_name, MIN_COUNT)
        self.model = SkipGramModel(self.data.word_count, EMB_DIMENSION)
        self.lr = LR
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

    def train(self):
        print("SkipGram Training......")
        pairs_count = self.data.evaluate_pairs_count(WINDOW_SIZE)
        print("pairs_count", pairs_count)
        batch_count = pairs_count / BATCH_SIZE
        print("batch_count", batch_count)
        process_bar = tqdm(range(int(batch_count)))
        for i in process_bar:
            pos_pairs = self.data.get_batch_pairs(BATCH_SIZE, WINDOW_SIZE)
            pos_pairs,neg_pairs = self.data.get_pairs(pos_pairs)
            pos_u = [pair[0] for pair in pos_pairs]
            pos_v = [int(pair[1]) for pair in pos_pairs]
            neg_u = [pair[0] for pair in neg_pairs]
            neg_v = [int(pair[1]) for pair in neg_pairs]
            self.optimizer.zero_grad()
            loss = self.model.forward(pos_u, pos_v, neg_u,neg_v)
            loss.backward()
            self.optimizer.step()
        self.model.save_embedding(self.data.id2word_dict, self.output_file_name)

if __name__ == '__main__':
    w2v = Word2Vec(input_file_name='text8.txt', output_file_name="skip_gram_hs.txt")
    w2v.train()


