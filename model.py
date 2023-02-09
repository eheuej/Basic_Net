import torch.nn as nn
import torch
import torch.nn.functional as F


# 定义一个类，继承于nn.module这个父类
# 类中实现两个方法 一个是初始化函数：在这实现搭建网络所需要网路层结构
# forward 定义正向传播的一个过程
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # input(3,32,32) output(16,28,28)
        x = self.pool1(x)  # output(16,14,14)
        x = F.relu(self.conv2(x))  # output(32,10,10)
        x = self.pool2(x)  # output(32,5,5)
        x = x.view(-1, 32 * 5 * 5)  # output(32 * 5 * 5)
        x = F.relu(self.fc1(x))  # output(120)
        x = F.relu(self.fc2(x))  # output(80)
        x = self.fc3(x)  # output(10)
        return x
#
# a = torch.rand([32, 3, 32, 32])
# model = LeNet()
# print(model)
# output = model(a)
