# -*- coding: utf-8 -*-

# @Time : 2020/9/20 下午1:43
# @Author : TaoWang
# @Description :

import argparse

def CBOW_HS_ArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="skip-gram", help="skip-gram or cbow")
    parser.add_argument("--window_size",type=int,default=4,help="window size in word2vec")
    parser.add_argument("--batch_size",type=int,default=64,help="batch size during training phase")
    parser.add_argument("--min_count",type=int,default=3,help="min count of training word")
    parser.add_argument("--embed_dimension",type=int,default=100,help="embedding dimension of word embedding")
    parser.add_argument("--learning_rate",type=float,default=0.02,help="learning rate during training phase")
    parser.add_argument("--neg_count",type=int,default=4,help="neg count of skip-gram")
    return parser.parse_args()

def CBOW_NEG_ArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="skip-gram", help="skip-gram or cbow")
    parser.add_argument("--window_size",type=int,default=3,help="window size in word2vec")
    parser.add_argument("--batch_size",type=int,default=256,help="batch size during training phase")
    parser.add_argument("--min_count",type=int,default=3,help="min count of training word")
    parser.add_argument("--embed_dimension",type=int,default=100,help="embedding dimension of word embedding")
    parser.add_argument("--learning_rate",type=float,default=0.02,help="learning rate during training phase")
    parser.add_argument("--neg_count",type=int,default=4,help="neg count of skip-gram")
    return parser.parse_args()

def SKip_Gram_HS_ArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="skip-gram", help="skip-gram or cbow")
    parser.add_argument("--window_size",type=int,default=3,help="window size in word2vec")
    parser.add_argument("--batch_size",type=int,default=256,help="batch size during training phase")
    parser.add_argument("--min_count",type=int,default=3,help="min count of training word")
    parser.add_argument("--embed_dimension",type=int,default=100,help="embedding dimension of word embedding")
    parser.add_argument("--learning_rate",type=float,default=0.02,help="learning rate during training phase")
    return parser.parse_args()

def SKip_Gram_NGE_ArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="skip-gram", help="skip-gram or cbow")
    parser.add_argument("--window_size",type=int,default=3,help="window size in word2vec")
    parser.add_argument("--batch_size",type=int,default=256,help="batch size during training phase")
    parser.add_argument("--min_count",type=int,default=3,help="min count of training word")
    parser.add_argument("--embed_dimension",type=int,default=100,help="embedding dimension of word embedding")
    parser.add_argument("--learning_rate",type=float,default=0.02,help="learning rate during training phase")
    parser.add_argument("--neg_count",type=int,default=5,help="neg count of skip-gram")
    return parser.parse_args()
