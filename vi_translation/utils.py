import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

def getDataLoader():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(root=os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data'), train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data'), train=False, download=True, transform=transform)

    # print("trainset: ", trainset[0][1])
    dogset = []
    catset = []
    for data, label in trainset:
        if label == 3:
            catset.append(data)
        elif label == 5:
            dogset.append(data)

    trainset = []
    for i in range(len(dogset)):
        tmp_data_set = (
            dogset[i],
            catset[i]
        )
        trainset.append(tmp_data_set)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

    return trainloader, testloader

def saveImg(e, type, img):
    grid_img = torchvision.utils.make_grid(img, normalize=True).permute(1,2,0)
    npimg = grid_img.detach().clone().cpu().numpy()
    plt.imsave('vi_translation/results/train5/' + type + '_' + str(e)  + '.png', npimg)