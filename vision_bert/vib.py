import sys
import os
# print(sys.path)
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms

from transformers import BertTokenizer, BertForTokenClassification

from PIL import Image
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForTokenClassification.from_pretrained('bert-base-uncased')

image = np.asfarray(Image.open('/data/bsh/datas/cifar-10/train/airplane/9992.png'))
image = image.reshape(-1)
image = torch.from_numpy(image).long()
image = image.flatten()
print(image)

# inputs = tokenizer.build_inputs_with_special_tokens(image).tolist()
outputs1, outputs2 = model(image.unsqueeze(0))
print(outputs1)
print(outputs2)

# dataset = dset.ImageFolder(root="/data/bsh/datas/cifar-10/train/")

# dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=8)

# for i, data in enumerate(dataloader):
#     print(data[0])
#     print(data[1])