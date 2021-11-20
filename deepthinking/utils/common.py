""" common.py
    Utility functions that are common to all tasks

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""

from datetime import datetime

import torch
from icecream import ic
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR

import deepthinking.models as models
from .mazes_data import prepare_maze_loader
from .prefix_sums_data import prepare_prefix_loader
from .chess_data import prepare_chess_loader
from .sudoku import prepare_sudoku_loader

from .warmup import ExponentialWarmup
# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115


def get_dataloaders(args):
    if args.problem == "prefix_sums":
        return prepare_prefix_loader(train_batch_size=args.train_batch_size,
                                     test_batch_size=args.test_batch_size,
                                     train_data=args.train_data,
                                     test_data=args.test_data)
    elif args.problem == "mazes":
        return prepare_maze_loader(train_batch_size=args.train_batch_size,
                                   test_batch_size=args.test_batch_size,
                                   train_data=args.train_data,
                                   test_data=args.test_data)
    elif args.problem == "chess":
        return prepare_chess_loader(train_batch_size=args.train_batch_size,
                                    test_batch_size=args.test_batch_size,
                                    train_data=args.train_data,
                                    test_data=args.test_data)
    elif args.problem == "sudoku":
        return prepare_sudoku_loader(train_batch_size=args.train_batch_size,
                                    test_batch_size=args.test_batch_size,
                                    train_data=args.train_data,
                                    test_data=args.test_data)

    else:
        raise ValueError(f"Invalid problem spec. {args.problem}")

def get_model(model, width, max_iters, in_channels=3):
    model = model.lower()
    net = getattr(models, model)(width=width, in_channels=in_channels, max_iters=max_iters)
    return net


def get_optimizer(optimizer_name, net, max_iters, epochs, lr, lr_decay, lr_schedule, lr_factor,
                  lr_throttle, warmup_period, state_dict):

    optimizer_name = optimizer_name.lower()

    if lr_throttle:
        # Reducing the lr here for the recurrent layers helps with stability,
        # To date (July 21, 2021), we may only need this for maze models.
        base_params = [p for n, p in net.named_parameters() if "recur" not in n]
        recur_params = [p for n, p in net.named_parameters() if "recur" in n]
        iters = max_iters
        all_params = [{"params": base_params}, {"params": recur_params, "lr": lr / iters}]
    else:
        base_params = [p for n, p in net.named_parameters()]
        recur_params = []
        iters = 1
        all_params = [{"params": base_params}]

    # all_params = [{"params": base_params}, {"params": recur_params, "lr": lr / iters}]

    if optimizer_name == "sgd":
        optimizer = SGD(all_params, lr=lr, weight_decay=2e-4, momentum=0.9)
    elif optimizer_name == "adam":
        optimizer = Adam(all_params, lr=lr, weight_decay=2e-4)
    else:
        raise ValueError(f"{ic.format()}: Optimizer choise of {optimizer_name} not yet implmented.")

    if state_dict is not None:
        optimizer.load_state_dict(state_dict)
        warmup_scheduler = ExponentialWarmup(optimizer, warmup_period=0)
    else:
        warmup_scheduler = ExponentialWarmup(optimizer, warmup_period=warmup_period)

    if lr_decay.lower() == "step":

        lr_scheduler = MultiStepLR(optimizer, milestones=lr_schedule,
                                   gamma=lr_factor, last_epoch=-1)

    elif lr_decay.lower() == "cosine":
        lr_scheduler = CosineAnnealingLR(optimizer, epochs, eta_min=0, last_epoch=-1, verbose=False)
    else:
        raise ValueError(f"{ic.format()}: Learning rate decay style {lr_decay} not yet implemented.")

    return optimizer, warmup_scheduler, lr_scheduler


def load_model_from_checkpoint(model, model_path, width, problem, max_iters, device):
    epoch = 0
    optimizer = None

    in_channels = 3
    if problem == "chess":
        in_channels = 12
    elif problem == "sudoku":
        in_channels = 10

    net = get_model(model, width, in_channels=in_channels, max_iters=max_iters)
    net = net.to(device)
    if device == "cuda":
        net = torch.nn.DataParallel(net)
    if model_path is not None:
        print(f"Loading model from checkpoint {model_path}...")
        state_dict = torch.load(model_path, map_location=device)
        net.load_state_dict(state_dict["net"])
        epoch = state_dict["epoch"] + 1
        optimizer = state_dict["optimizer"]

    return net, epoch, optimizer


def now():
    return datetime.now().strftime("%Y%m%d %H:%M:%S")
