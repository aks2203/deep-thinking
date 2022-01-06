""" logging_utils.py
    Utility functions for logging experiments to CometML and TensorBoard

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""
import os

# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115


def get_dirs_for_saving(args):

    # generate string for saving, to be used below when saving checkpoints and stats
    base_dir = f"{args.model}_{args.optimizer}" \
               f"_train_mode={args.train_mode}" \
               f"_width={args.width}" \
               f"_max_iters={args.max_iters}" \
               f"_alpha={args.alpha}" \
               f"_lr={args.lr}" \
               f"_batchsize={args.train_batch_size}" \
               f"_epoch={args.epochs - 1}"

    checkpoint = os.path.join("checkpoints", args.output, base_dir)
    result = os.path.join("results", args.output, base_dir, args.run_id)

    for path in [checkpoint, result]:
        if not os.path.isdir(path):
            os.makedirs(path)

    return checkpoint, result


def write_to_tb(stats, stat_names, epoch, writer):
    for name, stat in zip(stat_names, stats):
        stat_name = os.path.join("val", name)
        writer.add_scalar(stat_name, stat, epoch)
