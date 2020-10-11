# -*- coding: utf-8 -*-

# @Time : 2020/10/11 上午11:04
# @Author : TaoWang
# @Description : 训练模型

from data import Wiki_Dataset
from model import glove_model
from tqdm import tqdm
import torch
import numpy as np
import torch.optim as optim
import config as argumentparser


config = argumentparser.ArgumentParser()

# 设置GPU
if config.cuda and torch.cuda.is_available():
    torch.cuda.set_device(config.gpu)

torch.cuda.is_available()

# 导入训练集
wiki_dataset = Wiki_Dataset(min_count=config.min_count,window_size=config.window_size)
training_iter = torch.utils.data.DataLoader(dataset=wiki_dataset,
                                            batch_size=config.batch_size,
                                            shuffle=True,
                                            num_workers=2)

model = glove_model(len(wiki_dataset.word2id), config.embed_size, config.x_max, config.alpha)

# 将模型送进gpu
if config.cuda and torch.cuda.is_available():
    torch.cuda.set_device(config.gpu)
    model.cuda()

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

loss = -1
for epoch in range(config.epoch):
    # tqdm训练
    process_bar = tqdm(training_iter)
    for data, label in process_bar:
        w_data = torch.Tensor(np.array([sample[0] for sample in data])).long()
        v_data = torch.Tensor(np.array([sample[1] for sample in data])).long()
        if config.cuda and torch.cuda.is_available():
            w_data = w_data.cuda()
            v_data = v_data.cuda()
            label = label.cuda()
        loss_now = model(w_data, v_data, label)
        if loss == -1:
            loss = loss_now.data.item()
        else:
            # 平滑loss
            loss = 0.95*loss+0.05*loss_now.data.item()

        # 输出loss
        process_bar.set_postfix(loss=loss)
        process_bar.update()
        # 梯度更新
        optimizer.zero_grad()
        loss_now.backward()
        optimizer.step()

model.save_embedding()


