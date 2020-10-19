# -*- coding: utf-8 -*-

# @Time : 2020/10/15 下午12:09
# @Author : TaoWang
# @Description :


from pytorchtools import EarlyStopping
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from model import TextCNN
from data import MR_Dataset
import numpy as np
import config as argumentparser
config = argumentparser.ArgumentParser()
config.filters = list(map(int, config.filters.split(",")))


torch.manual_seed(config.seed)

if torch.cuda.is_available():
    torch.cuda.set_device(config.gpu)


def get_test_result(data_iter, data_set):
    # 生成测试结果
    model.eval()
    data_loss = 0
    true_sample_num = 0
    for data, label in data_iter:
        if config.cuda and torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()
        out = model(data)
        loss = criterion(out, autograd.Variable(label.long()))
        data_loss += loss.data.item()
        true_sample_num += np.sum((torch.argmax(out, 1) == label).cpu().numpy())
    acc = true_sample_num / data_set.__len__()

    return data_loss, acc


acc = 0
for i in range(0, 10):
    # 10-cv
    early_stopping = EarlyStopping(patience=10,
                                   verbose=True,
                                   cv_index=i)

    training_set = MR_Dataset(state="train",
                              k=i,
                              embedding_type=config.embedding_type)

    if config.use_pretrained_embed:
        config.embedding_pretrained = torch.from_numpy(training_set.weight).float()
    else:
        config.embedding_pretrained = False

    config.n_vocab = training_set.n_vocab
    training_iter = torch.utils.data.DataLoader(dataset=training_set,
                                                batch_size=config.batch_size,
                                                shuffle=True,
                                                num_workers=2)
    valid_set = MR_Dataset(state="valid",
                           k=i,
                           embedding_type="no")

    valid_iter = torch.utils.data.DataLoader(dataset=valid_set,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             num_workers=2)

    test_set = MR_Dataset(state="test",
                          k=i,
                          embedding_type="no")

    test_iter = torch.utils.data.DataLoader(dataset=test_set,
                                            batch_size=config.batch_size,
                                            shuffle=False,
                                            num_workers=2)
    model = TextCNN(config)
    if config.cuda and torch.cuda.is_available():
        model.cuda()
        config.embedding_pretrained.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    count = 0
    loss_sum = 0
    for epoch in range(config.epoch):
        # 开始训练
        model.train()
        for data, label in training_iter:
            if config.cuda and torch.cuda.is_available():
                data = data.cuda()
                label = label.cuda()
            else:
                data = torch.autograd.Variable(data).long()
            label = torch.autograd.Variable(label).squeeze()
            out = model(data)
            l2_loss = config.l2*torch.sum(torch.pow(list(model.parameters())[1],2))
            loss = criterion(out, autograd.Variable(label.long()))+l2_loss
            loss_sum += loss.data.item()
            count += 1
            if count % 100 == 0:
                print("epoch", epoch, end='  ')
                print("The loss is: %.5f" % (loss_sum / 100))
                loss_sum = 0
                count = 0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # save the model in every epoch
        # 一轮训练结束，在验证集测试
        valid_loss,valid_acc = get_test_result(valid_iter,valid_set)
        early_stopping(valid_loss, model)
        print ("The valid acc is: %.5f" % valid_acc)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    # 1 fold训练结果
    model.load_state_dict(torch.load('./checkpoints/checkpoint%d.pt'%i))
    test_loss, test_acc = get_test_result(test_iter, test_set)
    print("The test acc is: %.5f" % test_acc)
    acc += test_acc/10

# 输出10-fold的平均acc
print("The test acc is: %.5f" % acc)


