""" feedforward_net_2d.py
    Feed-forward 2D convolutional neural network.

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


class FeedForwardNet(nn.Module):
    """Modified Residual Network model class"""

    def __init__(self, block, num_blocks, width, in_channels=3, recall=True, max_iters=8, group_norm=False):
        super().__init__()

        self.width = int(width)
        self.recall = recall
        self.group_norm = group_norm

        proj_conv = nn.Conv2d(in_channels, width, kernel_size=3, stride=1, padding=1, bias=False)

        if self.recall:
            self.recall_layer = nn.Conv2d(width+in_channels, width, kernel_size=3,
                                          stride=1, padding=1, bias=False)
        else:
            self.recall_layer = nn.Sequential()

        self.feedforward_layers = nn.ModuleList()
        for _ in range(max_iters):
            internal_block = []
            for j in range(len(num_blocks)):
                internal_block.append(self._make_layer(block, width, num_blocks[j], stride=1))
            self.feedforward_layers.append(nn.Sequential(*internal_block))

        head_conv1 = nn.Conv2d(width, 32, kernel_size=3, stride=1, padding=1, bias=False)
        head_conv2 = nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1, bias=False)
        head_conv3 = nn.Conv2d(8, 2, kernel_size=3, stride=1, padding=1, bias=False)

        self.iters = max_iters
        self.projection = nn.Sequential(proj_conv, nn.ReLU())
        self.head = nn.Sequential(head_conv1, nn.ReLU(), head_conv2, nn.ReLU(), head_conv3)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for strd in strides:
            layers.append(block(self.width, planes, strd, self.group_norm))
            self.width = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, iters_to_do, interim_thought=None, iters_elapsed=0, **kwargs):
        assert (iters_elapsed + iters_to_do) <= self.iters
        initial_thought = self.projection(x)

        if interim_thought is None:
            interim_thought = initial_thought

        all_outputs = torch.zeros((x.size(0), iters_to_do, 2, x.size(2), x.size(3))).to(x.device)

        for i, layer in enumerate(self.feedforward_layers[iters_elapsed:iters_elapsed+iters_to_do]):
            if self.recall:
                interim_thought = torch.cat([interim_thought, x], 1)
                interim_thought = self.recall_layer(interim_thought)
            interim_thought = layer(interim_thought)
            out = self.head(interim_thought)
            all_outputs[:, i] = out

        if self.training:
            return out, interim_thought
        else:
            return all_outputs


def feedforward_net_2d(width, **kwargs):
    return FeedForwardNet(BasicBlock, [2], width, in_channels=kwargs["in_channels"],
                          recall=False, max_iters=kwargs["max_iters"])


def feedforward_net_recall_2d(width, **kwargs):
    return FeedForwardNet(BasicBlock, [2], width, in_channels=kwargs["in_channels"],
                          recall=True, max_iters=kwargs["max_iters"])


def feedforward_net_gn_2d(width, **kwargs):
    return FeedForwardNet(BasicBlock, [2], width, in_channels=kwargs["in_channels"],
                          recall=False, max_iters=kwargs["max_iters"], group_norm=True)


def feedforward_net_recall_gn_2d(width, **kwargs):
    return FeedForwardNet(BasicBlock, [2], width, in_channels=kwargs["in_channels"],
                          recall=True, max_iters=kwargs["max_iters"], group_norm=True)


# Testing
if __name__ == "__main__":
    net = feedforward_net_recall_2d(width=5, in_channels=3, max_iters=5)
    print(net)
    x_test = torch.rand(4 * 3 * 5 * 5).reshape([4, 3, 5, 5])
    out_test, _ = net(x_test)
    print(out_test.shape)
    out_test, _ = net(x_test, n=2, k=2)
    print(out_test.shape)

    net.eval()
    outputs = net(x_test)
    print(outputs.shape)
    outputs = net(x_test, n=2, k=2)
    print(outputs.shape)
