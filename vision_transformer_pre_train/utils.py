import os
import shutil

import pandas as pd

dat = pd.read_csv('/data/bsh/datas/cifar-10/trainLabels.csv')
label = []
for row_index, row in dat.iterrows():
    label.append(row.iloc[1:]['label'])
    # if row_index > 5:
    #     break
    # print(row.iloc[1:])

# print(label)
file_index = 0
for file in os.listdir('/data/bsh/datas/cifar-10/train'):
    if '.png' in file:
        shutil.move('/data/bsh/datas/cifar-10/train/' + file, '/data/bsh/datas/cifar-10/train/'+label[file_index] + '/' + file)
        # print(file)
        file_index += 1