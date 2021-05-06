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
import cv2

from vi_translation.utils import getDataLoader, saveImg

trainloader, testloader = getDataLoader()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Attention(nn.Module):
    def __init__(self):
        super().__init__()

        self.attn_drop = nn.Dropout(0.2)

    def forward(self, x):
        N, C, H, W = x.shape

        out = x.flatten(2)
        out = out.transpose(0, 1)

        q, k, v = out[0], out[1], out[2]
        k_t = k.transpose(-2, -1)
        dp = (q @ k_t) * 0.5
        attn = dp.softmax(dim=-1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v

        return weighted_avg

model = Attention().to(device)

# for epoch in range(100):
for i, data in enumerate(trainloader):
    dogdata = data[0]
    catdata = data[1]
    dogdata = dogdata.float().to(device)
    catdata = catdata.float().to(device)

    output = model(catdata)
    N = output.size(0)
    output = output.reshape(N, 1, 32, 32)

    saveImg(1, catdata)
    saveImg(2, output)
    print(output.shape)
    break

