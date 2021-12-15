""" test_model.py
    Test models

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""

import argparse
import logging
import uuid
from collections import OrderedDict

import json

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import deepthinking as dt
import deepthinking.utils.logging_utils as lg


# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115


@hydra.main(config_path="config", config_name="test_model_cfg")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    if cfg.hyp.save_period is None:
        cfg.hyp.save_period = cfg.hyp.epochs
    log = logging.getLogger()
    log.info("\n_________________________________________________\n")
    log.info("test_model.py main() running.")
    log.info(OmegaConf.to_yaml(cfg))

    training_args = OmegaConf.load(cfg.args_path)
    args.alpha = training_args["alpha"]
    args.epochs = training_args["epochs"]
    args.lr = training_args["lr"]
    args.lr_factor = training_args["lr_factor"]
    args.max_iters = training_args["max_iters"]
    args.model = training_args["model"]
    args.optimizer = training_args["optimizer"]
    args.train_data = training_args["train_data"]
    args.train_mode = training_args["train_mode"]
    args.width = training_args["width"]

    args.train_mode, args.test_mode = args.train_mode.lower(), args.test_mode.lower()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.backends.cudnn.benchmark = True

    for arg in vars(args):
        log.info(f"{arg}: {getattr(args, arg)}")

    assert 0 <= args.alpha <= 1, "Weighting for loss (alpha) not in [0, 1], exiting."

    _, args.output = lg.get_dirs_for_saving(args)
    lg.to_json(vars(args), args.output, "args.json")

    ####################################################
    #               Dataset and Network and Optimizer
    loaders = dt.utils.get_dataloaders(args)

    net, start_epoch, optimizer_state_dict = dt.utils.load_model_from_checkpoint(args.model,
                                                                                 args.model_path,
                                                                                 args.width,
                                                                                 args.problem,
                                                                                 args.max_iters,
                                                                                 device)

    args.test_iterations.append(args.max_iters)
    args.test_iterations = list(set(args.test_iterations))
    args.test_iterations.sort()

    pytorch_total_params = sum(p.numel() for p in net.parameters())

    log.info(f"This {args.model} has {pytorch_total_params/1e6:0.3f} million parameters.")
    ####################################################

    ####################################################
    #        Test
    log.info("==> Starting testing...")

    if args.quick_test:
        test_acc = dt.test(net, [loaders["test"]], args.test_mode, args.test_iterations,
                           args.problem, device)
        test_acc = test_acc[0]
        val_acc, train_acc = None, None
    else:
        test_acc, val_acc, train_acc = dt.test(net,
                                               [loaders["test"], loaders["val"], loaders["train"]],
                                               args.test_mode,
                                               args.test_iterations,
                                               args.problem, device)

    log.info(f"{dt.utils.now()} Training accuracy: {train_acc}")
    log.info(f"{dt.utils.now()} Val accuracy: {val_acc}")
    log.info(f"{dt.utils.now()} Testing accuracy (hard data): {test_acc}")

    model_name_str = f"{args.model}_width={args.width}"
    stats = OrderedDict([("epochs", args.epochs),
                         ("learning rate", args.lr),
                         ("lr", args.lr),
                         ("lr_factor", args.lr_factor),
                         ("max_iters", args.max_iters),
                         ("model", model_name_str),
                         ("model_path", args.model_path),
                         ("num_params", pytorch_total_params),
                         ("optimizer", args.optimizer),
                         ("val_acc", val_acc),
                         ("run_id", args.run_id),
                         ("test_acc", test_acc),
                         ("test_data", args.test_data),
                         ("test_iters", args.test_iterations),
                         ("test_mode", args.test_mode),
                         ("train_data", args.train_data),
                         ("train_acc", train_acc),
                         ("train_batch_size", args.train_batch_size),
                         ("train_mode", args.train_mode),
                         ("alpha", args.alpha)])
    lg.to_json(stats, args.output, "stats.json")
    ####################################################


if __name__ == "__main__":
    main()
