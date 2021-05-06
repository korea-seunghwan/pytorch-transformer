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

from vi_translation.vision_translation import VitTranslation, Generator, VitTranslationEncoder, Discriminator
from vi_translation.utils import getDataLoader, saveImg
import torchvision.models as models
# from PIL import Image
# import numpy as np

trainloader, testloader = getDataLoader()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

real_label = 1.
fake_label = 0.


model = VitTranslation(32, 4, 3, 512).to(device)
modelG = Generator().to(device)
modelD = Discriminator().to(device)

# Setup Adam optimizers for both G and D
optimizerM = torch.optim.Adam(model.parameters(), lr=0.01)
optimizerD = torch.optim.Adam(modelD.parameters(), lr=0.01)
optimizerG = torch.optim.Adam(modelG.parameters(), lr=0.01)

criterionM = nn.CrossEntropyLoss()
criterion = nn.BCELoss()

img_list = []
G_losses = []
D_losses = []
iters = 0

for epoch in range(100):

    for i, data in enumerate(trainloader):
        running_loss = 0.0

        dogdata = data[0]
        catdata = data[1]
        dogdata = dogdata.float().to(device)
        catdata = catdata.float().to(device)

        output = model(dogdata, catdata)

        inputG = output[:, 0]
        inputG = inputG.unsqueeze(-1).unsqueeze(-1)
        outputG = modelG(inputG)

        ##########################################################
        # 1. update D network: maximize log(D(x)) + log(1-D(G(z)))
        ##########################################################
        modelD.zero_grad()
        
        label = torch.full((dogdata.size(0),), real_label, dtype=torch.float, device=device)
        # print('label shape: ', label.shape)
        # Forward pass real batch through D
        outputD = modelD(catdata).view(-1)
        # print('outputD shape: ', outputD.shape)
        # calculate loss on all-real batch
        errD_real = criterion(outputD, label)

        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = outputD.mean().item()

        # Train with all-fake batch
        label.fill_(fake_label)
        # classify all fake batch with D
        outputD = modelD(outputG.detach_()).view(-1)
        # calculate D's loss on the all-fake batch
        errD_fake = criterion(outputD, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = outputD.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # update D
        optimizerD.step()

        ##########################################################
        # 2. update G network: maximize log(D(G(z)))
        ##########################################################
        modelG.zero_grad()
        label.fill_(real_label)
        outputD = modelD(outputG).view(-1)
        # calculate G's loss based on this output
        errG = criterion(outputD, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = outputD.mean().item()
        # update G
        optimizerG.step()


        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, 100, i, len(trainloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == 99) and (i == len(trainloader)-1)):
            with torch.no_grad():
                saveImg(epoch, outputG)

        iters += 1

        # print(outputG.shape)
        # saveImg(1, outputG)
        # print(outputG)





