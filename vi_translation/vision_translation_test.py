import sys
import os
# print(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt

from vi_translation.vision_translation import Generator, VitTranslationEncoder
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

modelE = VitTranslationEncoder(32, 4, 3, 512).to(device)
modelG = Generator().to(device)

criterion = nn.BCELoss()

real_label = 1.
fake_label = 0.

optimizerE = torch.optim.Adam(modelE.parameters(), lr=0.001, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(modelG.parameters(), lr=0.001, betas=(0.5, 0.999))

iters = 0
img_list_input = []
img_list_generate = []
num_epochs = 100

for epoch in range(num_epochs):
    for i, data in enumerate(trainloader):
        inputs, labels = data

        inputs = inputs.float().to(device)
        labels = labels.long().to(device)

        modelE.zero_grad()

        # Format batch
        N = inputs.size(0)
        label = torch.full((N,), real_label, dtype=torch.float, device=device)

        # Forward pass real batch through D
        outputs, outputs2 = modelE(inputs)
        outputs2 = outputs2.view(-1)

        # Calculate gradients for D in backward pass
        # print('output size: ', output.shape)
        # print('label size: ', label.shape)
        errD_real = criterion(outputs2, label)
        errD_real.backward()
        D_x = outputs2.mean().item()

        # Generate batch of latent vectors
        inputG = outputs[:,1:,:]
        inputG = torch.transpose(inputG, 1, 2)
        inputG = inputG.reshape(N, 512, 8, 8)
        fake = modelG(inputG)
        label.fill_(fake_label)

        # classify all fake batch with D
        outputs, outputs2 = modelE(fake.detach_())
        outputs2 = outputs2.view(-1)

        errD_fake = criterion(outputs2, label)
        errD_fake.backward()

        D_G_z1 = outputs2.mean().item()

        errD = errD_real + errD_fake

        # update D
        optimizerE.step()

        ####################################################################
        # update G
        modelG.zero_grad()
        label.fill_(real_label)
        outputs, outputs2 = modelE(fake)
        outputs2 = outputs2.view(-1)

        # print('output2 shape: ', output2.shape) 
        # print('label shape: ', label.shape)

        errG = criterion(outputs2, label)
        errG.backward()

        D_G_z2 = outputs2.mean().item()

        # update G
        optimizerG.step()

        if i % 50 == 0 :
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(trainloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            plt.figure(figsize=(8, 8))
            plt.axis('off')
            plt.title('inputs')
            plt.imsave("vi_translation/test.png", np.transpose(vutils.make_grid(inputs, padding=2, normalize=True).detach().clone().cpu().numpy(), (1,2,0)))

            plt.figure(figsize=(8, 8))
            plt.axis('off')
            plt.title('inputs')
            plt.imsave("vi_translation/test_G.png", np.transpose(vutils.make_grid(fake, padding=2, normalize=True).detach().clone().cpu().numpy(), (1,2,0)))