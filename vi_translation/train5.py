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

from vi_translation.translation import Generator, Discriminator
from vi_translation.utils import getDataLoader, saveImg
import torchvision.models as models
# from PIL import Image
# import numpy as np

trainloader, testloader = getDataLoader()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

real_label = 1.
fake_label = 0.

EMBEDED_SIZE = 512

epochs = 100

netG = Generator().to(device)
netD = Discriminator(img_size=32, patch_size=4, in_chans=3, embed_dim=EMBEDED_SIZE).to(device)

optimizerG = torch.optim.Adam(netG.parameters(), lr=0.01)
optimizerD = torch.optim.Adam(netD.parameters(), lr=0.01)

criterion = nn.BCEWithLogitsLoss()

for epoch in range(epochs):
    for i, data in enumerate(trainloader):
        dogdata = data[0].float().to(device)
        catdata = data[1].float().to(device)

        ######################################################################
        #   1. Update D network: maximize log(D(z)) + log(1-D(G(z)))
        ######################################################################
        netD.zero_grad()

        N = dogdata.size(0)

        label = torch.full((N, EMBEDED_SIZE), fake_label, dtype=torch.float, device=device)

        # Forward pass real batch through D
        output = netD(dogdata, catdata)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # print("errD_real: ", errD_real)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        # print(output_.mean().item())
        D_x = output.mean().item()
        
        ## Train with all-fake batch
        # Generate batch of latent vectors
        output = output.unsqueeze(-1).unsqueeze(-1)
        fake = netG(output)

        label.fill_(fake_label)
        
        # Classify all fake batch with D
        output = netD(fake.detach_(), catdata)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # print("errD_fake: ", errD_fake)
        # Calculate the gradients for this batch , accumulated with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()
        
        ######################################################################
        #   2. Update G network: maximize log(D(G(z)))
        ######################################################################
        netG.zero_grad()
        label.fill_(real_label)

        output = netD(fake, catdata)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()

        optimizerG.step()

        if i % 50 == 0:
            saveImg(epoch, 'real', catdata)
            saveImg(epoch, 'output', fake)
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (epoch, epochs, i, len(trainloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))


