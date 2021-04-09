import sys
import os
# print(sys.path)
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms

from transformers import BertForTokenClassification

from PIL import Image
import numpy as np

dataset = dset.ImageFolder(root="/data/bsh/datas/cifar-10/train/",
                transform=transforms.Compose([transforms.ToTensor()]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=8)

for i, data in enumerate(dataloader):
    print(data[0])
    print(data[1])