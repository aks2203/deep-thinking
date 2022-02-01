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
from deepthinking.utils.testing_utils import get_predicted

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

        all_outputs = torch.zeros((x.size(0), iters_to_do, 2, x.size(2), x.size(3))).to(x.device)

        for i in range(iters_to_do):
            if self.recall:
                interim_thought = torch.cat([interim_thought, x], 1)
            interim_thought = self.recur_block(interim_thought)
            out = self.head(interim_thought)
            all_outputs[:, i] = out

        if self.training:
            return out, interim_thought

        return all_outputs


class DTNetTestDefMaxHugeMaze(nn.Module):
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

        targets = kwargs["targets"]
        problem = kwargs["problem"]

        if interim_thought is None:
            interim_thought = initial_thought

        confidence_array = torch.zeros(iters_to_do, x.size(0)).to(x.device)
        corrects_array = torch.zeros(iters_to_do, x.size(0)).to(x.device)
        softmax = torch.nn.functional.softmax

        corrects_max_conf = torch.zeros(iters_to_do).to(x.device)
        corrects_default = torch.zeros(iters_to_do).to(x.device)
        wrong_pixels = torch.zeros(iters_to_do).to(x.device)

        for i in range(iters_to_do):
            if self.recall:
                interim_thought = torch.cat([interim_thought, x], 1)
            interim_thought = self.recur_block(interim_thought)
            out = self.head(interim_thought)

            conf = softmax(out.detach(), dim=1).max(1)[0]
            conf = conf.view(conf.size(0), -1)
            if problem == "mazes":
                conf = conf * x.max(1)[0].view(conf.size(0), -1)
            confidence_array[i] = conf.sum([1])
            predicted = get_predicted(x, out, problem)
            corrects_array[i] = torch.amin(predicted == targets, dim=[1])
            corrects_default[i] += torch.amin(predicted == targets, dim=[1]).sum().item()
            wrong_pixels[i] += (predicted != targets).sum().item()

            if i % 100 == 0:
                print(i)

        correct_this_iter = corrects_array[torch.cummax(confidence_array, dim=0)[1],
                                           torch.arange(corrects_array.size(1))]
        corrects_max_conf += correct_this_iter.sum(dim=1)

        return corrects_max_conf, corrects_default, wrong_pixels


class DTNetSudoku(nn.Module):
    """DeepThinking Network 2D model class"""

    def __init__(self, block, num_blocks, width, in_channels=3, recall=True, **kwargs):
        super().__init__()
        self.in_channels = 1
        self.out_channels = 10

        self.recall = recall
        self.width = int(width)
        proj_conv = nn.Conv2d(self.in_channels, width, kernel_size=3,
                              stride=1, padding=1, bias=False)

        conv_recall = nn.Conv2d(width + self.in_channels, width, kernel_size=3,
                                stride=1, padding=1, bias=False)

        recur_layers = []
        if recall:
            recur_layers.append(conv_recall)

        for i in range(len(num_blocks)):
            recur_layers.append(self._make_layer(block, width, num_blocks[i], stride=1))

        head_conv1 = nn.Conv2d(width, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        head_conv2 = nn.Conv2d(64, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        head_conv3 = nn.Conv2d(32, self.out_channels, kernel_size=3,
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

        all_outputs = torch.zeros((x.size(0), iters_to_do, self.out_channels, x.size(2), x.size(3))).to(x.device)

        for i in range(iters_to_do):
            if self.recall:
                interim_thought = torch.cat([interim_thought, x], 1)
            interim_thought = self.recur_block(interim_thought)
            out = self.head(interim_thought)
            all_outputs[:, i] = out

        if self.training:
            return out, interim_thought

        return all_outputs


def dt_net_2d(width, **kwargs):
    return DTNet(BasicBlock, [2], width=width, in_channels=kwargs["in_channels"], recall=False)


def dt_net_recallx_2d(width, **kwargs):
    return DTNet(BasicBlock, [2], width=width, in_channels=kwargs["in_channels"], recall=True)


def dt_net_recallx_2d_sudoku(width, **kwargs):
    return DTNetSudoku(BasicBlock, [2], width=width, recall=True)

def dt_net_recallx_2d_sudoku_3(width, **kwargs):
    return DTNetSudoku(BasicBlock, [3], width=width, recall=True)

def dt_net_recallx_2d_test_huge(width, **kwargs):
    return DTNetTestDefMaxHugeMaze(BasicBlock, [2], width=width, in_channels=kwargs["in_channels"], recall=True)