import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


class NewDataset(Dataset):
    def __init__(self, transform=None):
        xy = np.loadtxt('PATH', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = xy[:, :]  # n_samples, n_features
        self.y = xy[:, [0]]  # n_samples, 1
        self.n_samples = xy.shape[0]
        self.transform = transform

    def __getitem__(self, item):
        sample = self.x[item], self.y[item]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples


class ToTensor:
    def __call__(self, sample):
        x, y = sample
        return torch.from_numpy(x), torch.from_numpy(y)


class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        x, y = sample
        x *= self.factor
        return x, y


composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = NewDataset(transform=composed)
