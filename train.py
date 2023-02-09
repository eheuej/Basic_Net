import time
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch import optim as optim

from model import LeNet

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# batch_size = 4

# 导入50000张训练图片
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                         shuffle=False, num_workers=0)
test_data_iter = iter(testloader)
test_image, test_label = (test_data_iter.next())

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = LeNet()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

epoch_train_loss = []
epoch_test_accuracy = []
for epoch in range(5):
    running_loss = 0.0  # 累加训练过程的损失
    time_start = time.perf_counter()
    for step, data in enumerate(trainloader, start=0):
        inputs, labels = data
        # print("-------------")
        # print(inputs)
        # print("-------------")
        # print(labels)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        # 以本demo为例，训练集一共有50000个样本，batch_size = 50，那么完整的训练一次样本：iteration或step = 1000，epoch = 1
        running_loss += loss.item()
        if step % 500 == 499:
            with torch.no_grad():
                outputs = net(test_image)
                predict_y = torch.max(outputs, dim=1)[1]
                accuracy = (predict_y == test_label).sum().item() / test_label.size(0)
                print('[%d, %5d] train_loss: %.3f test_accuracy: %.3f' % (
                    epoch + 1, step + 1, running_loss / 500, accuracy))
                print('%f s' % (time.perf_counter() - time_start))  # 打印耗时
                running_loss = 0.0

print("Finish Training!!")
save_path = './LeNet.pth'
torch.save(net.state_dict(), save_path)
