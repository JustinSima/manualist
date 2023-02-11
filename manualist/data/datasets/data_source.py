""" Class for storing data loaders from train/val/split directories."""
from dataclasses import dataclass

import torch


@dataclass
class DataSource:
    """ A convenience wrapper for my preference."""
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader

    train_dataset: torch.utils.data.Dataset
    val_dataset: torch.utils.data.Dataset
    test_dataset: torch.utils.data.Dataset

    def __init__(self):
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
