""" train_model.py
    Train, test, and save models

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""

import json
import logging
import os
import sys

from collections import OrderedDict
import logging

import hydra
import numpy as np
import torch
from icecream import ic
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

import deepthinking as dt
import deepthinking.utils.logging_utils as lg

# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115


@hydra.main(config_path="config", config_name="train_model_config")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    if cfg.hyp.save_period is None:
        cfg.hyp.save_period = cfg.hyp.epochs
    log = logging.getLogger()
    log.info("\n_________________________________________________\n")
    log.info("train_model.py main() running.")
    log.info(OmegaConf.to_yaml(cfg))

    dt.utils.setup_test_iterations(cfg)
    assert 0 <= cfg.hyp.alpha <= 1, "Weighting for loss (alpha) not in [0, 1], exiting."
    writer = SummaryWriter(log_dir=f"tensorboard")

    ####################################################
    #               Dataset and Network and Optimizer
    loaders = dt.utils.get_dataloaders(cfg)

    net, start_epoch, optimizer_state_dict = dt.utils.load_model_from_checkpoint(cfg.model.model,
                                                                                 cfg.model.model_path,
                                                                                 cfg.model.width,
                                                                                 cfg.problem,
                                                                                 cfg.hyp.max_iters,
                                                                                 device)

    pytorch_total_params = sum(p.numel() for p in net.parameters())
    log.info(f"This {cfg.model.model} has {pytorch_total_params/1e6:0.3f} million parameters.")
    log.info(f"Training will start at epoch {start_epoch}.")

    optimizer, warmup_scheduler, lr_scheduler = dt.utils.get_optimizer(cfg.hyp.optimizer,
                                                                       net,
                                                                       cfg.hyp.epochs,
                                                                       cfg.hyp.lr,
                                                                       cfg.hyp.lr_decay,
                                                                       cfg.hyp.lr_schedule,
                                                                       cfg.hyp.lr_factor,
                                                                       cfg.hyp.warmup_period,
                                                                       optimizer_state_dict)
    train_setup = dt.TrainingSetup(optimizer=optimizer,
                                   scheduler=lr_scheduler,
                                   warmup=warmup_scheduler,
                                   clip=cfg.hyp.clip,
                                   alpha=cfg.hyp.alpha,
                                   max_iters=cfg.hyp.max_iters,
                                   problem=cfg.problem,
                                   throttle=cfg.hyp.lr_throttle)
    ####################################################

    ####################################################
    #        Train
    log.info(f"==> Starting training for {max(cfg.hyp.epochs - start_epoch, 0)} epochs...")
    highest_val_acc_so_far = -1
    best_so_far = False

    for epoch in range(start_epoch, cfg.hyp.epochs):

        loss, acc = dt.train(net, loaders, cfg.hyp.train_mode, train_setup, device,
                             disable_tqdm=cfg.hyp.disable_tqdm)
        val_acc = dt.test(net, [loaders["val"]], cfg.hyp.test_mode, [cfg.hyp.max_iters],
                          cfg.problem, device, disable_tqdm=cfg.hyp.disable_tqdm)[0][cfg.hyp.max_iters]
        if val_acc > highest_val_acc_so_far:
            best_so_far = True
            highest_val_acc_so_far = val_acc

        log.info(f"Training loss at epoch {epoch}: {loss}")
        log.info(f"Training accuracy at epoch {epoch}: {acc}")
        log.info(f"Val accuracy at epoch {epoch}: {val_acc}")

        # if the loss is nan, then stop the training
        if np.isnan(float(loss)):
            raise ValueError(f"{ic.format()} Loss is nan, exiting...")

        # TensorBoard loss writing
        writer.add_scalar("Loss/loss", loss, epoch)
        writer.add_scalar("Accuracy/acc", acc, epoch)
        writer.add_scalar("Accuracy/val_acc", val_acc, epoch)

        for i in range(len(optimizer.param_groups)):
            writer.add_scalar(f"Learning_rate/group{i}",
                              optimizer.param_groups[i]["lr"],
                              epoch)

        if (epoch + 1) % cfg.hyp.val_period == 0:
            test_acc, val_acc, train_acc = dt.test(net,
                                                   [loaders["test"],
                                                    loaders["val"],
                                                    loaders["train"]],
                                                   cfg.hyp.test_mode,
                                                   cfg.hyp.test_iterations,
                                                   cfg.problem,
                                                   device, disable_tqdm=cfg.hyp.disable_tqdm)
            log.info(f"Training accuracy: {train_acc}")
            log.info(f"Val accuracy: {val_acc}")
            log.info(f"Test accuracy (hard data): {test_acc}")

            tb_last = cfg.hyp.test_iterations[-1]
            lg.write_to_tb([train_acc[tb_last], val_acc[tb_last], test_acc[tb_last]],
                           ["train_acc", "val_acc", "test_acc"],
                           epoch,
                           writer)
        # check to see if we should save
        save_now = (epoch + 1) % cfg.hyp.save_period == 0 or \
                   (epoch + 1) == cfg.hyp.epochs or best_so_far

        if save_now:
            state = {"net": net.state_dict(), "epoch": epoch, "optimizer": optimizer.state_dict()}
            out_str = f"model_{'best' if best_so_far else ''}.pth"
            best_so_far = False
            log.info(f"Saving model to: {out_str}")
            torch.save(state, out_str)
    writer.flush()
    writer.close()
    ####################################################

    ####################################################
    #        Test
    log.info("==> Starting testing...")
    log.info("Testing the best checkpoint from training.")

    # load the best checkpoint
    model_path = os.path.join("model_best.pth")
    net, _, _ = dt.utils.load_model_from_checkpoint(cfg.model.model, model_path, cfg.model.width, cfg.problem,
                                                    cfg.hyp.max_iters, device)

    test_acc, val_acc, train_acc = dt.test(net,
                                           [loaders["test"], loaders["val"], loaders["train"]],
                                           cfg.hyp.test_mode,
                                           cfg.hyp.test_iterations,
                                           cfg.problem, device, disable_tqdm=cfg.hyp.disable_tqdm)

    log.info(f"Training accuracy: {train_acc}")
    log.info(f"Val accuracy: {val_acc}")
    log.info(f"Testing accuracy (hard data): {test_acc}")


    model_name_str = f"{cfg.model.model}_width={cfg.model.width}"
    stats = OrderedDict([("epochs", cfg.hyp.epochs),
                         ("lr", cfg.hyp.lr),
                         ("lr_factor", cfg.hyp.lr_factor),
                         ("max_iters", cfg.hyp.max_iters),
                         ("model", model_name_str),
                         ("model_path", model_path),
                         ("num_params", pytorch_total_params),
                         ("optimizer", cfg.hyp.optimizer),
                         ("val_acc", val_acc),
                         ("run_id", cfg.run_id),
                         ("test_acc", test_acc),
                         ("test_data", cfg.hyp.test_data),
                         ("test_iters", list(cfg.hyp.test_iterations)),
                         ("test_mode", cfg.hyp.test_mode),
                         ("train_data", cfg.hyp.train_data),
                         ("train_acc", train_acc),
                         ("train_batch_size", cfg.hyp.train_batch_size),
                         ("train_mode", cfg.hyp.train_mode),
                         ("alpha", cfg.hyp.alpha)])
    with open(os.path.join("stats.json"), "w") as fp:
        json.dump(stats, fp)
    log.info(stats)
    ####################################################


if __name__ == "__main__":
    run_id = dt.utils.generate_run_id()
    sys.argv.append(f"+run_id={run_id}")
    main()
