import os
import pickle
import numpy as np
import ipdb
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils import data
from sklearn.model_selection import train_test_split

class PytorchDataset(data.Dataset):
    """ Custom PyTorch dataset """
    def __init__(self, features, labels):
        self.features = features.astype('float32')
        self.labels = labels.astype('float32')

    def __len__(self):
        "Returns the size of the dataset"
        return self.features.shape[0]

    def __getitem__(self, index):
        "Returns an element from the dataset"
        features = self.features[index]
        label = self.labels[index]

        return features, label


def dataset_to_dataloader(dataset, batch_size=100, drop_last=False, shuffle=True):
    """
    Transforms an instance of the Dataset class into a Pytorch Dataset and then
    into a Pytorch Dataloader
    """

    features, labels = dataset.features, dataset.labels
    py_dataset = PytorchDataset(features, labels)
    dataloader = data.DataLoader(py_dataset,
                                 batch_size=batch_size,
                                 drop_last=drop_last,
                                 shuffle=shuffle)

    return dataloader

