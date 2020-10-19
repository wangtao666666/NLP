# -*- coding: utf-8 -*-

# @Time : 2020/10/15 上午11:01
# @Author : TaoWang
# @Description :

import torch
import torch.nn as nn


class BasicModule(nn.Module):

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        """
        : 加载模型
        :param path: 模型位置
        :return:
        """
        self.load_state_dict(torch.load(path))

    def save(self, path):
        """
        :param path:
        :return:
        """
        torch.save(self.state_dict(), path)

    def foward(self):
        pass


if __name__ == "__main__":
    print("Running the BasicModule ....")
    model = BasicModule()