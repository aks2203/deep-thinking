""" dt_net_1d.py
    DeepThinking 1D convolutional neural network.

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""

import torch
from torch import nn

from .blocks import BasicBlock1D as BasicBlock

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702)
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914


class EnergyBlock(torch.nn.Module):
    """Exposes a forward pass through gradient descent of a parametrized energy."""

    def __init__(self, width, kernel_size=3, step_size=0.1, recall=True):
        super().__init__()
        self.width = width
        self.step_size = 0.1
        self.recall = True  # has direct access to input data

        # Conjugate Net:
        self.operator = nn.Conv1d(width + 1, width, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, inputs):
        """This is a conjugate-net type energy as in p.74 of the tech report."""
        with torch.enable_grad():
            x = inputs[:, :self.width]
            x.requires_grad_()
            x = torch.nn.functional.layer_norm(x, x.shape[1:])
            Ex = (self.operator(inputs) * x).sum() + torch.where(x >= 0, x**2, torch.zeros_like(x)).sum()  # this is homogenous with bias
            grads = torch.autograd.grad(Ex, x, create_graph=True)[0]
        return x - self.step_size * grads

    def energy(self, inputs):
        return (self.operator(inputs) * x).sum()

class DTEnergy1D(nn.Module):
    """DeepThinking 1D Network model class"""

    def __init__(self, block, num_blocks, width, recall, **kwargs):
        super().__init__()

        self.width = int(width)
        self.recall = recall

        proj_conv = nn.Conv1d(1, width, kernel_size=3,
                              stride=1, padding=1, bias=False)


        head_conv1 = nn.Conv1d(width, width, kernel_size=3,
                               stride=1, padding=1, bias=False)
        head_conv2 = nn.Conv1d(width, int(width/2), kernel_size=3,
                               stride=1, padding=1, bias=False)
        head_conv3 = nn.Conv1d(int(width/2), 2, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.projection = nn.Sequential(proj_conv, nn.ReLU())
        self.recur_block = EnergyBlock(width, kernel_size=3, recall=recall)
        self.head = nn.Sequential(head_conv1, nn.ReLU(),
                                  head_conv2, nn.ReLU(),
                                  head_conv3)



    def forward(self, x, iters_to_do, interim_thought=None, **kwargs):
        initial_thought = self.projection(x)

        if interim_thought is None:
            interim_thought = initial_thought

        all_outputs = torch.zeros((x.size(0), iters_to_do, 2, x.size(2))).to(x.device)

        for i in range(iters_to_do):
            if self.recall:
                interim_thought = torch.cat([interim_thought, x], 1)

            interim_thought = self.recur_block(interim_thought)
            out = self.head(interim_thought)
            all_outputs[:, i] = out

        if self.training:
            return out, interim_thought

        return all_outputs


def dt_energy_1d(width, **kwargs):
    return DTEnergy1D(BasicBlock, [2], width, recall=False)


def dt_energy_recallx_1d(width, **kwargs):
    return DTEnergy1D(BasicBlock, [2], width, recall=True)
