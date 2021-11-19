""" test_model.py
    Test models

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""

import argparse
import uuid
from collections import OrderedDict

import json
import torch

import deepthinking as dt
import deepthinking.utils.logging_utils as lg


# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115


def main():

    print("\n_________________________________________________\n")
    print(dt.utils.now(), "train_model.py main() running.")

    parser = argparse.ArgumentParser(description="Deep Thinking")

    parser.add_argument("--args_path", default=None, type=str, help="where are the args saved?")
    parser.add_argument("--model_path", default=None, type=str, help="from where to load model")
    parser.add_argument("--model", default=None, type=str, help="If None load from args else take this architecture")
    parser.add_argument("--output", default="output_default", type=str, help="output subdirectory")
    parser.add_argument("--problem", default="prefix_sums", type=str,
                        help="one of 'prefix_sums', 'mazes', or 'chess'")
    parser.add_argument("--quick_test", action="store_true", help="test with test data only")
    parser.add_argument("--test_batch_size", default=500, type=int, help="batch size for testing")
    parser.add_argument("--test_data", default=48, type=int, help="which data to test on")
    parser.add_argument("--test_iterations", nargs="+", default=[30, 40], type=int,
                        help="how many iterations for testing")
    parser.add_argument("--test_mode", default="max_conf", type=str, help="testing mode")
    parser.add_argument("--train_batch_size", default=100, type=int, help="training batch size")

    args = parser.parse_args()
    args.run_id = uuid.uuid1().hex
    with open(args.args_path, "r") as fp:
        args_dict = json.load(fp)
    training_args = args_dict["0"]

    args.alpha = training_args["alpha"]
    args.epochs = training_args["epochs"]
    args.lr = training_args["lr"]
    args.lr_factor = training_args["lr_factor"]
    args.max_iters = training_args["max_iters"]
    if args.model == None:
        args.model = training_args["model"]
    args.optimizer = training_args["optimizer"]
    args.train_data = training_args["train_data"]
    args.train_mode = training_args["train_mode"]
    args.width = training_args["width"]

    args.train_mode, args.test_mode = args.train_mode.lower(), args.test_mode.lower()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.backends.cudnn.benchmark = True

    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

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

    print(f"This {args.model} has {pytorch_total_params/1e6:0.3f} million parameters.")
    ####################################################

    ####################################################
    #        Test
    print("==> Starting testing...")

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

    print(f"{dt.utils.now()} Training accuracy: {train_acc}")
    print(f"{dt.utils.now()} Val accuracy: {val_acc}")
    print(f"{dt.utils.now()} Testing accuracy (hard data): {test_acc}")

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
