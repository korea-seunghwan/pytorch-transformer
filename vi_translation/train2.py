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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root=os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data'), train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

classes = ['plane', 'car', 'bird', 'cat','deer','dog','frog','horse','ship','truck']


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VitTranslation(32, 4, 3, 512).to(device)

for epoch in range(100):

    for i, data in enumerate(trainloader, 0):
        running_loss = 0.0
        for j, data2 in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs1, labels1 = data
            inputs2, labels2 = data2

            inputs1 = inputs1.float().to(device)
            labels1 = labels1.long().to(device)

            inputs2 = inputs2.float().to(device)
            labels2 = labels2.long().to(device)

            output = model(inputs1, inputs2)

            print("output : ", output[:,0,0])