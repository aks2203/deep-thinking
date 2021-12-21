""" test_model.py
    Test models

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""

import logging
import os
import sys
from collections import OrderedDict

import json

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import deepthinking as dt

# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115


@hydra.main(config_path="config", config_name="test_model_config")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    if cfg.hyp.save_period is None:
        cfg.hyp.save_period = cfg.hyp.epochs
    log = logging.getLogger()
    log.info("\n_________________________________________________\n")
    log.info("test_model.py main() running.")
    log.info(OmegaConf.to_yaml(cfg))

    training_args = OmegaConf.load(os.path.join(cfg.model.model_path, ".hydra/config.yaml"))
    cfg_keys_to_load = [("hyp", "alpha"),
                        ("hyp", "epochs"),
                        ("hyp", "lr"),
                        ("hyp", "lr_factor"),
                        ("model", "max_iters"),
                        ("model", "model"),
                        ("hyp", "optimizer"),
                        ("problem", "train_data"),
                        ("hyp", "train_mode"),
                        ("model", "width")]
    for k1, k2 in cfg_keys_to_load:
        cfg[k1][k2] = training_args[k1][k2]
    # dt.utils.setup_test_iterations(cfg)
    log.info(OmegaConf.to_yaml(cfg))

    ####################################################
    #               Dataset and Network and Optimizer
    loaders = dt.utils.get_dataloaders(cfg)

    model_path = os.path.join(cfg.model.model_path, "model_best.pth")
    net, start_epoch, optimizer_state_dict = dt.utils.load_model_from_checkpoint(cfg.model.model,
                                                                                 model_path,
                                                                                 cfg.model.width,
                                                                                 cfg.problem.name,
                                                                                 cfg.model.max_iters,
                                                                                 device)
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    log.info(f"This {cfg.model.model} has {pytorch_total_params/1e6:0.3f} million parameters.")
    ####################################################

    ####################################################
    #        Test
    log.info("==> Starting testing...")
    test_iterations = list(range(cfg.model.test_iterations["low"], cfg.model.test_iterations["high"] + 1))
    if cfg.quick_test:
        test_acc = dt.test(net, [loaders["test"]], cfg.hyp.test_mode, test_iterations, cfg.problem.name, device)
        test_acc = test_acc[0]
        val_acc, train_acc = None, None
    else:
        test_acc, val_acc, train_acc = dt.test(net,
                                               [loaders["test"], loaders["val"], loaders["train"]],
                                               cfg.hyp.test_mode,
                                               test_iterations,
                                               cfg.problem.name, device)

    log.info(f"{dt.utils.now()} Training accuracy: {train_acc}")
    log.info(f"{dt.utils.now()} Val accuracy: {val_acc}")
    log.info(f"{dt.utils.now()} Testing accuracy (hard data): {test_acc}")

    model_name_str = f"{cfg.model.model}_width={cfg.model.width}"
    stats = OrderedDict([("epochs", cfg.hyp.epochs),
                         ("lr", cfg.hyp.lr),
                         ("lr_factor", cfg.hyp.lr_factor),
                         ("max_iters", cfg.model.max_iters),
                         ("model", model_name_str),
                         ("model_path", model_path),
                         ("num_params", pytorch_total_params),
                         ("optimizer", cfg.hyp.optimizer),
                         ("val_acc", val_acc),
                         ("run_id", cfg.run_id),
                         ("test_acc", test_acc),
                         ("test_data", cfg.problem.test_data),
                         ("test_iters", test_iterations),
                         ("test_mode", cfg.hyp.test_mode),
                         ("train_data", cfg.problem.train_data),
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
