import sys
import os
# print(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

from vi_translation.vision_translation import VitTranslation
import torchvision.models as models
# from PIL import Image
# import numpy as np

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

classes = ['plane', 'car', 'bird', 'cat','deer','dog','frog','horse','ship','truck']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VitTranslation(32, 4, 3, 512).to(device)

# for epoch in range(100):

for i, data in enumerate(trainloader):
    running_loss = 0.0

    dogdata = data[0]
    catdata = data[1]
    # get the inputs; data is a list of [inputs, labels]
    print("data: ", dogdata.shape)
    # print("data 1: ", data[1].shape)
    break