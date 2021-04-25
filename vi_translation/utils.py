import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from PIL import Image

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def image_to_data(path):
    cat_data = torch.tensor([])
    dog_data = torch.tensor([])
    for folder_list in sorted(os.listdir(path)):
        for img_list in sorted(os.listdir(os.path.join(path, folder_list))):
            img = transform(Image.open(
                os.path.join(path, folder_list, img_list)))
            img = img.unsqueeze(0)
            # print(type(img))
            if folder_list == 'cat':
                cat_data = torch.cat((cat_data, img), dim=0)
            elif folder_list == 'dog':
                dog_data = torch.cat((dog_data, img), dim=0)

    # cat_data = np.array(cat_data)
    # dog_data = np.array(dog_data)

    # print('cat_data: ', cat_data.shape)
    # print('dog_data: ', dog_data.shape)

    return cat_data, dog_data

def getDataset(path):
    data_array = []
    cat_data, dog_data = image_to_data(path)
    for i in range(cat_data.shape[0]):
        tmp_data_tuple = (
            cat_data[i],
            dog_data[i]
        )
        data_array.append(tmp_data_tuple)

    # data_array = torch.Tensor(data_array)

    train_size = int(0.8 * len(data_array))
    test_size = len(data_array) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        data_array, [train_size, test_size])

    print('train_dataset type: ', type(train_dataset))
    print('test_dataset type: ', type(test_dataset))

    return train_dataset, test_dataset