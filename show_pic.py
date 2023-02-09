import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=50,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=5,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

test_data_iter = iter(testloader)
test_image, test_label = test_data_iter.next()


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    # 把原来图片格式为（channel,height,weight）,分别对应（0，1，2）---->转换为(height,weight,channel)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# print labels
print(' '.join(f'{classes[test_label[j]]:5s}' for j in range(5)))

# show images
imshow(torchvision.utils.make_grid(test_image))
