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
from .blocks import LocRNNBlock2D as LocRNNBlock
from .blocks import LocRNNEIBlock2D as LocRNNEIBlock
# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702)
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914


class DTNet(nn.Module):
    """DeepThinking Network 2D model class"""

    def __init__(self, block, num_blocks, width, in_channels=3, recall=True, group_norm=False, **kwargs):
        super().__init__()

        self.recall = recall
        self.x_to_h = kwargs['x_to_h']
        if 'split_gate' in kwargs.keys():
            self.split_gate = kwargs['split_gate']
        else:
            self.split_gate = False
        self.width = int(width)
        self.group_norm = group_norm
        proj_conv = nn.Conv2d(in_channels, width, kernel_size=3,
                              stride=1, padding=1, bias=False)

        conv_recall = nn.Conv2d(width + in_channels, width, kernel_size=3,
                                stride=1, padding=1, bias=False)

        recur_layers = []
        if block == LocRNNBlock:
            recur_layers.append(block(width, timesteps=kwargs['max_iters'], recall=self.recall, x_to_h=self.x_to_h, split_gate=self.split_gate))
            self.block_type = "locrnn"
        elif block == LocRNNEIBlock:
            recur_layers.append(block(width, timesteps=kwargs['max_iters']))
            self.block_type = "locrnn_ei"
        else:
            if recall:
                recur_layers.append(conv_recall)
            
            for i in range(len(num_blocks)):
                recur_layers.append(self._make_layer(block, width, num_blocks[i], stride=1))
            self.block_type = "basicblock"

        head_conv1 = nn.Conv2d(width, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        head_conv2 = nn.Conv2d(32, 8, kernel_size=3,
                               stride=1, padding=1, bias=False)
        head_conv3 = nn.Conv2d(8, 2, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.projection = nn.Sequential(proj_conv, nn.ReLU())
        if not self.block_type.startswith("locrnn"):
            self.recur_block = nn.Sequential(*recur_layers)
        else:
            self.recur_block = recur_layers[0]

        self.head = nn.Sequential(head_conv1, nn.ReLU(),
                                  head_conv2, nn.ReLU(),
                                  head_conv3)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for strd in strides:
            layers.append(block(self.width, planes, strd, group_norm=self.group_norm))
            self.width = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x, iters_to_do, interim_thought=None, return_hidden=False, **kwargs):
        if self.block_type == "locrnn":
            return self.forward_locrnn(x, iters_to_do, interim_thought, return_hidden=return_hidden)
        elif self.block_type == "locrnn_ei":
            return self.forward_locrnn_ei(x, iters_to_do, interim_thought, return_hidden=return_hidden)
        else:
            return self.forward_dtnet(x, iters_to_do, interim_thought, return_hidden=return_hidden)
    
    def forward_locrnn_ei(self, x, iters_to_do, interim_thought=None, return_hidden=False, **kwargs):
        all_outputs = torch.zeros((x.size(0), iters_to_do, 2, x.size(2), x.size(3))).to(x.device)
        out = self.projection(x)
        inter_outputs = self.recur_block(out, iters_to_do, 
                                            interim_thought=interim_thought, 
                                            stepwise_predictions=True,
                                            image=x)
        for i in range(iters_to_do):
            out = self.head(inter_outputs[i])
            all_outputs[:, i] = out
        if self.training:
            return out, all_outputs
        if return_hidden:
            return all_outputs, inter_outputs
        return all_outputs
    
    def forward_locrnn(self, x, iters_to_do, interim_thought=None, return_hidden=False, **kwargs):
        all_outputs = torch.zeros((x.size(0), iters_to_do, 2, x.size(2), x.size(3))).to(x.device)
        
        out = self.projection(x)
        inter_outputs = self.recur_block(out, iters_to_do, 
                                            interim_thought=interim_thought, 
                                            stepwise_predictions=True,
                                            image=x,
                                            )
        for i in range(iters_to_do):
            out = self.head(inter_outputs[0][i])
            all_outputs[:, i] = out
        if self.training:
            return out, all_outputs
        if return_hidden:
            return all_outputs, inter_outputs
        return all_outputs

    def forward_dtnet(self, x, iters_to_do, interim_thought=None, return_hidden=False, **kwargs):
        initial_thought = self.projection(x)

        if interim_thought is None:
            interim_thought = initial_thought

        all_outputs = torch.zeros((x.size(0), iters_to_do, 2, x.size(2), x.size(3))).to(x.device)
        inter_outputs = []
        for i in range(iters_to_do):
            if self.recall:
                interim_thought = torch.cat([interim_thought, x], 1)
            interim_thought = self.recur_block(interim_thought)
            inter_outputs.append(interim_thought)
            out = self.head(interim_thought)
            all_outputs[:, i] = out

        if self.training:
            return out, interim_thought
        if return_hidden:
            return all_outputs, inter_outputs
        return all_outputs


def dt_net_2d(width, **kwargs):
    return DTNet(BasicBlock, [2], width=width, in_channels=kwargs["in_channels"], recall=False)


def dt_net_recall_2d(width, **kwargs):
    return DTNet(BasicBlock, [2], width=width, in_channels=kwargs["in_channels"], recall=True, x_to_h=None)


def dt_net_gn_2d(width, **kwargs):
    return DTNet(BasicBlock, [2], width=width, in_channels=kwargs["in_channels"], recall=False, group_norm=True)


def dt_net_recall_gn_2d(width, **kwargs):
    return DTNet(BasicBlock, [2], width=width, in_channels=kwargs["in_channels"], recall=True, group_norm=True)

def locrnn_2d(width, **kwargs):
    return DTNet(LocRNNBlock, [1], width=width, in_channels=kwargs["in_channels"], max_iters=kwargs['max_iters'], recall=False, x_to_h=False)

def locrnn_2d_x_to_h(width, **kwargs):
    return DTNet(LocRNNBlock, [1], width=width, in_channels=kwargs["in_channels"], max_iters=kwargs['max_iters'], recall=False, x_to_h=True)

def locrnn_2d_recall(width, **kwargs):
    return DTNet(LocRNNBlock, [1], width=width, in_channels=kwargs["in_channels"], max_iters=kwargs['max_iters'], recall=True, x_to_h=True)

def locrnn_2d_recall_splitgate(width, **kwargs):
    return DTNet(LocRNNBlock, [1], width=width, in_channels=kwargs["in_channels"], max_iters=kwargs['max_iters'], recall=True, x_to_h=False, split_gate=True)

def locrnn_2d_recall_x_to_h(width, **kwargs):
    return DTNet(LocRNNBlock, [1], width=width, in_channels=kwargs["in_channels"], max_iters=kwargs['max_iters'], recall=True, x_to_h=True)

def locrnn_ei_2d(width, **kwargs):
    return DTNet(LocRNNEIBlock, [1], width=width, in_channels=kwargs["in_channels"], max_iters=kwargs['max_iters'], recall=True, x_to_h=False)