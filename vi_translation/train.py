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

from vi_translation.vision_translation import DiscriminatorCNN, VitTranslation, Generator, Discriminator
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
modelG = Generator().to(device)
modelD = Discriminator().to(device)
modelD_CNN = DiscriminatorCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

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

            # print('inputs shape: ', inputs1.shape)
            # print('labels shape: ', labels1.shape)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            outputs = model(inputs1, inputs2)

            N, P, C = outputs.shape

            # image discriminator
            outputD = modelD(outputs[:,0,:])
            

            inputG = outputs[:,1:,:]
            inputG = torch.transpose(inputG, 1, 2)
            inputG = inputG.reshape(N, C, 8, 8)
            # print(inputG.shape)
            plt.imsave('vi_translation/test.png', np.transpose(inputG[0,-3:,:].detach().cpu().numpy(), (1,2,0)))

            # image generator
            outputG = modelG(inputG)
            
            loss_D = modelD_CNN(outputG)

            # loss calculate
            lossD = criterion(outputD, labels2)
            loss_G = criterion(loss_D, labels2)

            loss = lossD + loss_G

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if j % 50 == 0:
                print('[%d %5d] loss: %.3f' % (epoch +1, i+1, running_loss / 50))
                running_loss = 0.
            # print(outputs.shape)
            # print(outputs.shape)
            
            


