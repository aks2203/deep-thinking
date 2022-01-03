""" chess_data.py
    Chess related dataloaders

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""

import torch

from torch.utils import data
from easy_to_hard_data import ChessPuzzleDataset


# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115),
#     Unused import (W0611).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, W0611


class FlippedChessPuzzleDataset(ChessPuzzleDataset):
    """Class to get flipped chess data. In this setting the player to move next is always
    at the bottom of the board, and the king is always on the right"""
    def __init__(self, root: str,
                 train: bool = True,
                 idx_start: int = None,
                 idx_end: int = None,
                 who_moves: bool = True,
                 download: bool = True):
        ChessPuzzleDataset.__init__(self, root, train, idx_start, idx_end, who_moves, download)
        rotate_idx = (self.who_moves == 1).squeeze()
        rotated_puzzles = torch.flip(self.puzzles[rotate_idx], [2])
        self.puzzles[rotate_idx] = rotated_puzzles
        rotated_targets = torch.flip(self.targets[rotate_idx], [1])
        self.targets[rotate_idx] = rotated_targets


def prepare_chess_loader(train_batch_size, test_batch_size, train_data, test_data, shuffle=True):

    trainset = FlippedChessPuzzleDataset("../../../data", idx_start=0, idx_end=train_data, who_moves=False,
                                         download=True)
    testset = FlippedChessPuzzleDataset("../../../data", idx_start=test_data-100000, idx_end=test_data,
                                        who_moves=False, download=True)

    train_split = int(0.8 * len(trainset))

    trainset, valset = torch.utils.data.random_split(trainset,
                                                     [train_split, int(len(trainset)-train_split)],
                                                     generator=torch.Generator().manual_seed(42))

    trainloader = data.DataLoader(trainset, num_workers=0,
                                  batch_size=train_batch_size,
                                  shuffle=shuffle,
                                  drop_last=True)
    valloader = data.DataLoader(valset, num_workers=0,
                                batch_size=test_batch_size,
                                shuffle=False,
                                drop_last=False)
    testloader = data.DataLoader(testset, num_workers=0,
                                 batch_size=test_batch_size,
                                 shuffle=False,
                                 drop_last=False)

    loaders = {"train": trainloader, "test": testloader, "val": valloader}

    return loaders
