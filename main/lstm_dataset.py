from math import ceil

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import Normalizer
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class TimeSeriesDataset(Dataset):
    def __init__(self, transform=None):
        xy = pd.read_pickle('./datasets/sets/dataset.pkl')
        self.x = xy.drop(columns=['label'])
        self.y = xy['label']
        self.n_samples = xy.shape[0]
        self.transform = transform

    def __getitem__(self, item):
        sample = self.x.iloc[item], self.y.iloc[item]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples


class DownSample:
    def __init__(self, factor):
        self.factor = factor
        self.signal = ceil(3001 / self.factor)
        self.time_arr = np.linspace(0.0, 30.0, self.signal)

    def __call__(self, sample):
        x, y = sample
        res = []
        for val in x:
            val = val[::self.factor]
            val = val[:self.signal]
            res.append(val)
        res = np.array(res)
        return torch.tensor(res), torch.tensor(y)


# TODO Normalize, scale
# TODO split (train, test, valid), shuffle (time-series), k-fold,
dataset = TimeSeriesDataset(transform=DownSample(2))  # TODO Try different HZ
print(dataset.__getitem__(0))
