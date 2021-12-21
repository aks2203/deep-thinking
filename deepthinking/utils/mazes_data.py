""" mazes_data.py
    Maze related dataloaders

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""

import torch
from torch.utils import data
from easy_to_hard_data import MazeDataset

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115),
#     Unused import (W0611).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, W0611


def prepare_maze_loader(train_batch_size, test_batch_size, train_data, test_data, shuffle=True):

    train_data = MazeDataset("../../../data", train=True, size=train_data, download=True)
    testset = MazeDataset("../../../data", train=False, size=test_data, download=True)

    train_split = int(0.8 * len(train_data))

    trainset, valset = torch.utils.data.random_split(train_data,
                                                     [train_split,
                                                      int(len(train_data) - train_split)],
                                                     generator=torch.Generator().manual_seed(42))

    trainloader = data.DataLoader(trainset,
                                  num_workers=0,
                                  batch_size=train_batch_size,
                                  shuffle=shuffle,
                                  drop_last=True)
    valloader = data.DataLoader(valset,
                                num_workers=0,
                                batch_size=test_batch_size,
                                shuffle=False,
                                drop_last=False)
    testloader = data.DataLoader(testset,
                                 num_workers=0,
                                 batch_size=test_batch_size,
                                 shuffle=False,
                                 drop_last=False)

    loaders = {"train": trainloader, "test": testloader, "val": valloader}

    return loaders
