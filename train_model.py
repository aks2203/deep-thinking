""" train_model.py
    Train, test, and save models

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""

import argparse
import os

import uuid
from collections import OrderedDict

# Comet must be imported before torch
# import comet_ml
import numpy as np
import torch
from icecream import ic

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

    parser.add_argument("--alpha", default=1, type=float,
                        help="weight to be used with progressive loss")
    parser.add_argument("--clip", default=None, type=float,
                        help="max gradient magnitude for training")
    parser.add_argument("--epochs", default=150, type=int, help="number of epochs for training")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--lr_decay", default="step", type=str, help="which kind of lr decay")
    parser.add_argument("--lr_factor", default=0.1, type=float, help="learning rate decay factor")
    parser.add_argument("--lr_schedule", nargs="+", default=[60, 100], type=int,
                        help="when to decrease lr")
    parser.add_argument("--lr_throttle", action="store_true",
                        help="reduce the lr for recurrent layers, this is needed for mazes.",)
    parser.add_argument("--max_iters", default=30, type=int, help="maximum number of iterations")
    parser.add_argument("--model", default="dt_net_recallx_1d", type=str, help="architecture")
    parser.add_argument("--model_path", default=None, type=str, help="where is the model saved?")
    parser.add_argument("--no_shuffle", action="store_false", dest="shuffle",
                        help="shuffle training data?")
    parser.add_argument("--optimizer", default="adam", type=str, help="optimizer")
    parser.add_argument("--output", default="output_default", type=str, help="output subdirectory")
    parser.add_argument("--problem", default="prefix_sums", type=str,
                        help="one of 'prefix_sums', 'mazes', or 'chess'")
    parser.add_argument("--save_period", default=None, type=int, help="how often to save")
    parser.add_argument("--test_batch_size", default=500, type=int, help="batch size for testing")
    parser.add_argument("--test_data", default=48, type=int, help="which data to test on")
    parser.add_argument("--test_iterations", nargs="+", default=[30, 40], type=int,
                        help="how many iterations for testing")
    parser.add_argument("--test_mode", default="max_conf", type=str, help="testing mode")
    parser.add_argument("--train_batch_size", default=100, type=int, help="training batch size")
    parser.add_argument("--train_data", default=32, type=int, help="which data to train on")
    parser.add_argument("--train_log", default="train_log.txt", type=str, help="log file name")
    parser.add_argument("--train_mode", default="progressive", type=str, help="training mode")
    parser.add_argument("--use_comet", action="store_true", help="whether to use comet logging")
    parser.add_argument("--val_period", default=20, type=int, help="how often to validate")
    parser.add_argument("--warmup_period", default=5, type=int, help="warmup period")
    parser.add_argument("--width", default=400, type=int, help="width of the network")

    args = parser.parse_args()
    args.test_iterations.append(args.max_iters)
    args.test_iterations = list(set(args.test_iterations))
    args.test_iterations.sort()
    args.run_id = uuid.uuid1().hex
    args.train_mode, args.test_mode = args.train_mode.lower(), args.test_mode.lower()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    if args.save_period is None:
        args.save_period = args.epochs

    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    assert 0 <= args.alpha <= 1, "Weighting for loss (alpha) not in [0, 1], exiting."

    args.checkpoint, args.output = lg.get_dirs_for_saving(args)
    writer = lg.setup_tb(args.train_log, args.output)
    comet_exp = lg.setup_comet(args)
    lg.to_json(vars(args), args.checkpoint, f"{args.run_id}_args.json")

    ####################################################
    #               Dataset and Network and Optimizer
    loaders = dt.utils.get_dataloaders(args)

    net, start_epoch, optimizer_state_dict = dt.utils.load_model_from_checkpoint(args.model,
                                                                                 args.model_path,
                                                                                 args.width,
                                                                                 args.problem,
                                                                                 args.max_iters,
                                                                                 device)

    pytorch_total_params = sum(p.numel() for p in net.parameters())
    #print(net)
    print(f"This {args.model} has {pytorch_total_params/1e6:0.3f} million parameters.")
    print(f"Training will start at epoch {start_epoch}.")

    optimizer, warmup_scheduler, lr_scheduler = dt.utils.get_optimizer(args.optimizer,
                                                                       net,
                                                                       args.max_iters,
                                                                       args.epochs,
                                                                       args.lr,
                                                                       args.lr_decay,
                                                                       args.lr_schedule,
                                                                       args.lr_factor,
                                                                       args.lr_throttle,
                                                                       args.warmup_period,
                                                                       optimizer_state_dict)
    train_setup = dt.TrainingSetup(optimizer=optimizer,
                                   scheduler=lr_scheduler,
                                   warmup=warmup_scheduler,
                                   clip=args.clip,
                                   alpha=args.alpha,
                                   max_iters=args.max_iters,
                                   problem=args.problem)
    ####################################################

    ####################################################
    #        Train
    print(f"==> Starting training for {max(args.epochs - start_epoch, 0)} epochs...")
    highest_val_acc_so_far = -1
    best_so_far = False

    for epoch in range(start_epoch, args.epochs):

        loss, acc = dt.train(net, loaders, args.train_mode, train_setup, device,
                             disable_tqdm=args.use_comet)
        val_acc = dt.test(net, [loaders["val"]], args.test_mode, [args.max_iters],
                          args.problem, device, disable_tqdm=args.use_comet)[0][args.max_iters]
        if val_acc > highest_val_acc_so_far:
            best_so_far = True
            highest_val_acc_so_far = val_acc

        print(f"{dt.utils.now()} Training loss at epoch {epoch}: {loss}")
        print(f"{dt.utils.now()} Training accuracy at epoch {epoch}: {acc}")

        # if the loss is nan, then stop the training
        if np.isnan(float(loss)):
            raise ValueError(f"{ic.format()} Loss is nan, exiting...")

        # TensorBoard loss writing
        writer.add_scalar("Loss/loss", loss, epoch)
        writer.add_scalar("Accuracy/acc", acc, epoch)

        for i in range(len(optimizer.param_groups)):
            writer.add_scalar(f"Learning_rate/group{i}",
                              optimizer.param_groups[i]["lr"],
                              epoch)

        if (epoch + 1) % args.val_period == 0:
            test_acc, val_acc, train_acc = dt.test(net,
                                                   [loaders["test"],
                                                    loaders["val"],
                                                    loaders["train"]],
                                                   args.test_mode,
                                                   args.test_iterations,
                                                   args.problem,
                                                   device, disable_tqdm=args.use_comet)
            print(f"{dt.utils.now()} Training accuracy: {train_acc}")
            print(f"{dt.utils.now()} Val accuracy: {val_acc}")
            print(f"{dt.utils.now()} Test accuracy (hard data): {test_acc}")

            tb_last = args.test_iterations[-1]
            lg.write_to_tb([train_acc[tb_last], val_acc[tb_last], test_acc[tb_last]],
                           ["train_acc", "val_acc", "test_acc"],
                           epoch,
                           writer)
            if comet_exp:
                lg.log_to_comet(comet_exp, train_acc, val_acc, test_acc, epoch)

        # check to see if we should save
        save_now = (epoch + 1) % args.save_period == 0 or \
                   (epoch + 1) == args.epochs or best_so_far

        if save_now:
            state = {"net": net.state_dict(), "epoch": epoch, "optimizer": optimizer.state_dict()}

            out_str = os.path.join(args.checkpoint,
                                   f"{args.run_id}_{'best' if best_so_far else ''}.pth")
            best_so_far = False

            print(f"{dt.utils.now()} Saving model to: {out_str}")
            torch.save(state, out_str)
            lg.to_json(out_str, args.output, "checkpoint_path.json")

    writer.flush()
    writer.close()
    ####################################################

    ####################################################
    #        Test
    print("==> Starting testing...")
    print("\t\t Testing the best checkpoint from training.")

    # load the best checkpoint
    model_path = os.path.join(args.checkpoint, f"{args.run_id}_best.pth")
    net, _, _ = dt.utils.load_model_from_checkpoint(args.model, model_path, args.width, args.problem,
                                           args.max_iters, device)

    test_acc, val_acc, train_acc = dt.test(net,
                                           [loaders["test"], loaders["val"], loaders["train"]],
                                           args.test_mode,
                                           args.test_iterations,
                                           args.problem, device, disable_tqdm=args.use_comet)

    print(f"{dt.utils.now()} Training accuracy: {train_acc}")
    print(f"{dt.utils.now()} Val accuracy: {val_acc}")
    print(f"{dt.utils.now()} Testing accuracy (hard data): {test_acc}")

    if comet_exp:
        lg.log_to_comet(comet_exp, train_acc, val_acc, test_acc, epoch, out_str)

    model_name_str = f"{args.model}_width={args.width}"
    stats = OrderedDict([("epochs", args.epochs),
                         ("learning rate", args.lr),
                         ("lr", args.lr),
                         ("lr_factor", args.lr_factor),
                         ("max_iters", args.max_iters),
                         ("model", model_name_str),
                         ("model_path", model_path),
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
