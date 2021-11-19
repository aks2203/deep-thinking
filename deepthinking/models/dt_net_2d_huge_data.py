""" dt_net_2d.py
    DeepThinking network 2D.

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""

import torch
from torch import nn

from .blocks import BasicBlock2D as BasicBlock

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702)
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914


class DTNet(nn.Module):
    """DeepThinking Network 2D model class"""

    def __init__(self, block, num_blocks, width, in_channels=3, recall=True, **kwargs):
        super().__init__()

        self.recall = recall
        self.width = int(width)
        proj_conv = nn.Conv2d(in_channels, width, kernel_size=3,
                              stride=1, padding=1, bias=False)

        conv_recall = nn.Conv2d(width + in_channels, width, kernel_size=3,
                                stride=1, padding=1, bias=False)

        recur_layers = []
        if recall:
            recur_layers.append(conv_recall)

        for i in range(len(num_blocks)):
            recur_layers.append(self._make_layer(block, width, num_blocks[i], stride=1))

        head_conv1 = nn.Conv2d(width, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        head_conv2 = nn.Conv2d(32, 8, kernel_size=3,
                               stride=1, padding=1, bias=False)
        head_conv3 = nn.Conv2d(8, 2, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.projection = nn.Sequential(proj_conv, nn.ReLU())
        self.recur_block = nn.Sequential(*recur_layers)
        self.head = nn.Sequential(head_conv1, nn.ReLU(),
                                  head_conv2, nn.ReLU(),
                                  head_conv3)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for strd in strides:
            layers.append(block(self.width, planes, strd))
            self.width = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, iters_to_do, interim_thought=None, **kwargs):
        initial_thought = self.projection(x)

        if interim_thought is None:
            interim_thought = initial_thought

        for i in range(iters_to_do):
            if self.recall:
                interim_thought = torch.cat([interim_thought, x], 1)
            interim_thought = self.recur_block(interim_thought)
            out = self.head(interim_thought)

        return out


def dt_net_2d_huge_data(width, **kwargs):
    return DTNet(BasicBlock, [2], width=width, in_channels=kwargs["in_channels"], recall=False)


def dt_net_recallx_2d_huge_data(width, **kwargs):
    return DTNet(BasicBlock, [2], width=width, in_channels=kwargs["in_channels"], recall=True)
