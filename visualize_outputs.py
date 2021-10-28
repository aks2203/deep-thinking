""" visualize_model.py
    Visualize the problem soving process

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""

import argparse
import os
import uuid
from glob import glob

import imageio
import json
import torch
import torchvision

import deepthinking as dt
import deepthinking.utils.logging_utils as lg
from deepthinking.utils.testing_utils import get_predicted


# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115


def save_frames_of_output(net, inputs, targets, iters, problem, save_path, device):
    net.train()
    with torch.no_grad():
        inputs, targets = inputs.to(device), targets.to(device)
        interim_thought = None
        for ite in range(iters):
            outputs, interim_thought = net(inputs, iters_to_do=1, interim_thought=interim_thought)
            predicted = get_predicted(inputs, outputs, problem)

            predicted = predicted.reshape(1, inputs.size(2), inputs.size(3))
            predicted = torch.stack([predicted] * 3, dim=1).float()
            torchvision.utils.save_image(torchvision.utils.make_grid(predicted),
                                         os.path.join(save_path, f"outputs_{ite}.png"))
    return


def make_gif(output_path):
    images = []
    filenames = glob(os.path.join(output_path, f"outputs_*.png"))
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(os.path.join(output_path, "outputs.gif"), images)


def main():

    print("\n_________________________________________________\n")
    print(dt.utils.now(), "visualize_outputs.py main() running.")

    parser = argparse.ArgumentParser(description="Deep Thinking - visualize")

    parser.add_argument("--args_path", default=None, type=str, help="where are the args saved?")
    parser.add_argument("--model_path", default=None, type=str, help="from where to load model")
    parser.add_argument("--output", default="output_default", type=str, help="output subdirectory")
    parser.add_argument("--test_data", default=48, type=int, help="which data to test on")
    parser.add_argument("--test_iterations", nargs="+", default=[30, 40], type=int,
                        help="how many iterations for testing")
    parser.add_argument("--test_input_index", default=0, type=int, help="which input to visualize")
    parser.add_argument("--test_mode", default="max_conf", type=str, help="testing mode")

    args = parser.parse_args()
    args.run_id = uuid.uuid1().hex
    with open(args.args_path, "r") as fp:
        args_dict = json.load(fp)
    training_args = args_dict["0"]

    # args.alpha = training_args["alpha"]
    args.alpha = training_args["weight_for_loss"]
    args.epochs = training_args["epochs"]
    args.lr = training_args["lr"]
    args.lr_factor = training_args["lr_factor"]
    args.max_iters = training_args["max_iters"]
    args.model = training_args["model"]
    args.model = "dt_net_recallx_2d"
    args.optimizer = training_args["optimizer"]
    args.problem = training_args["problem"]
    args.test_batch_size = 1
    args.train_batch_size = 1
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
    loader = dt.utils.get_dataloaders(args)["test"]

    net, start_epoch, optimizer_state_dict = dt.utils.load_model_from_checkpoint(args.model,
                                                                                 args.model_path,
                                                                                 args.width,
                                                                                 args.problem,
                                                                                 args.max_iters,
                                                                                 device)

    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print(f"This {args.model} has {pytorch_total_params/1e6:0.3f} million parameters.")
    ####################################################

    ####################################################
    #        Get Frames
    print("==> Starting testing...")

    dataloader_iter = iter(loader)
    for _ in range(args.test_input_index+1):
        inputs, targets = next(dataloader_iter)

    iters = max(args.test_iterations)
    save_frames_of_output(net, inputs, targets, iters, args.problem, args.output, device)
    make_gif(args.output)
    ####################################################


if __name__ == "__main__":
    main()
