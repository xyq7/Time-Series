import torch
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dir='./data/processed/single-stock/train.pkl', T=10):
        self.data = pd.read_pickle(dir)
        self.dir = dir
        self.T = T

    def __len__(self):
        return len(self.data) - self.T

    def __getitem__(self, idx):
        data = self.data[idx: idx + self.T]
        label = self.data[idx + self.T:idx + self.T + 1]['close']
        return np.array(data), label.item()


# # example
# dataset = Dataset()
# dataloader = torch.utils.data.DataLoader(dataset)
# for data in dataloader:
#     print(data)
