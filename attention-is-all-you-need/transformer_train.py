import torch
import torch.nn as nn
import os
import json
from torchnlp.download import download_file_maybe_extract

def squad_dataset(directory='./data',
                  train=True,
                  dev=True,
                  train_filename='train-v2.0.json',
                  dev_filename='dev-v2.0.json',
                  check_files_train=['train-v2.0.json'],
                  check_files_dev=['dev-v2.0.json'],
                  url_train='https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json',
                  url_dev='https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json'):

    download_file_maybe_extract(url=url_train, directory=directory, check_files=check_files_train)
    download_file_maybe_extract(url=url_dev, directory=directory, check_files=check_files_dev)

    ret = []
    splits = [(train, train_filename), (dev, dev_filename)]
    splits = [f for (requested, f) in splits if requested]
    for filename in splits:
        full_path = os.path.join(directory, filename)
        with open(full_path, 'r') as temp:
            ret.append(json.load(temp)['data'])

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)


# print(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data'))
out = squad_dataset(directory=os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data'))
print(out[0][0]['paragraphs'][0].keys())
