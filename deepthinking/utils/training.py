""" training.py
    Utilities for training models

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""

from dataclasses import dataclass
from random import randrange

import torch
from icecream import ic
from tqdm import tqdm

from deepthinking.utils.testing import get_predicted


# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115, C0114),
#     Unused import (W0611).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, C0114, W0611


@dataclass
class TrainingSetup:
    """Attributes to describe the training precedure"""
    optimizer: "typing.Any"
    scheduler: "typing.Any"
    warmup: "typing.Any"
    clip: "typing.Any"
    alpha: "typing.Any"
    max_iters: "typing.Any"
    problem: "typing.Any"
    late_predictions: "typing.Any"


def get_output_for_prog_loss(inputs, max_iters, net):
    # get features from n iterations to use as input
    n = randrange(0, max_iters)

    # do k iterations using intermediate features as input
    k = randrange(1, max_iters - n + 1)

    if n > 0:
        _, interim_thought = net(inputs, iters_to_do=n)
        if isinstance(interim_thought, tuple):
            interim_thought = (interim_thought[0].detach(), interim_thought[1].detach())
        else:
            interim_thought = interim_thought.detach()
    else:
        interim_thought = None
    outputs, _ = net(inputs, iters_elapsed=n, iters_to_do=k, interim_thought=interim_thought)
    return outputs, k


def train(net, loaders, mode, train_setup, device):
    if mode == "progressive":
        train_loss, acc = train_progressive(net, loaders, train_setup, device)
    else:
        raise ValueError(f"{ic.format()}: train_{mode}() not implemented.")
    return train_loss, acc


def train_progressive(net, loaders, train_setup, device):
    trainloader = loaders["train"]
    net.train()
    optimizer = train_setup.optimizer
    lr_scheduler = train_setup.scheduler
    warmup_scheduler = train_setup.warmup
    alpha = train_setup.alpha
    max_iters = train_setup.max_iters
    k = 0
    problem = train_setup.problem
    clip = train_setup.clip
    ce_weight = torch.Tensor([1, 7])
    if torch.cuda.is_available():
        ce_weight = ce_weight.cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=ce_weight, reduction="none")
    # criterion = torch.nn.CrossEntropyLoss(reduction="none")
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, leave=False)):
        inputs, targets = inputs.to(device), targets.to(device).long()
        targets = targets.view(targets.size(0), -1)
        if problem == "mazes":
            mask = inputs.view(inputs.size(0), inputs.size(1), -1).max(dim=1)[0] > 0

        optimizer.zero_grad()

        # get fully unrolled loss if alpha is not 1 (if it is 1, this loss term is not used
        # so we save time by settign it equal to 0).
        outputs_max_iters, _ = net(inputs, iters_to_do=max_iters)
        if alpha != 1:
            outputs_max_iters = outputs_max_iters.view(outputs_max_iters.size(0),
                                                       outputs_max_iters.size(1), -1)
            loss_max_iters = criterion(outputs_max_iters, targets)
        else:
            loss_max_iters = torch.zeros_like(targets).float()

        # get progressive loss if alpha is not 0 (if it is 0, this loss term is not used
        # so we save time by setting it equal to 0).
        if alpha != 0:
            if train_setup.late_predictions:
                n = randrange(max(1, max_iters-10), max_iters)
                # do k iterations using intermediate features as input
                k = randrange(1, 5)
                outputs_n_k, all_outputs_n_k = net(inputs, iters_to_do=n+k)
                all_outputs_n_k = all_outputs_n_k.view(all_outputs_n_k.size(0),
                                                        all_outputs_n_k.size(1),
                                                        all_outputs_n_k.size(2), -1)
                losses_n_k = [criterion(all_outputs_n_k[:, ii], targets) for ii in range(n, n+k)]
                loss_progressive = sum(losses_n_k)
                # outputs_n_k = outputs_n_k.view(outputs_n_k.size(0), outputs_n_k.size(1), -1)
                # loss_progressive = criterion(outputs_n_k, targets)
            else:
                if not batch_idx:
                    print("Training with progressive loss")
                outputs, k = get_output_for_prog_loss(inputs, max_iters, net)
                outputs = outputs.view(outputs.size(0), outputs.size(1), -1)
                loss_progressive = criterion(outputs, targets)
        else:
            loss_progressive = torch.zeros_like(targets).float()

        if problem == "mazes":
            loss_max_iters = (loss_max_iters * mask)
            loss_max_iters = loss_max_iters[mask > 0]
            loss_progressive = (loss_progressive * mask)
            loss_progressive = loss_progressive[mask > 0]
        loss_max_iters_mean = loss_max_iters.mean()
        loss_progressive_mean = loss_progressive.mean()

        loss = (1 - alpha) * loss_max_iters_mean + alpha * loss_progressive_mean
        loss.backward()

        if clip is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        train_loss += loss.item()
        predicted = get_predicted(inputs, outputs_max_iters, problem)
        correct += torch.amin(predicted == targets, dim=[-1]).sum().item()
        total += targets.size(0)

    train_loss = train_loss / (batch_idx + 1)
    acc = 100.0 * correct / total

    lr_scheduler.step()
    warmup_scheduler.dampen()

    return train_loss, acc
