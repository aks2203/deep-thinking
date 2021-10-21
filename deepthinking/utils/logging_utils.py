""" logging_utils.py
    Utility functions for logging experiments to CometML and TensorBoard

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""
import json
import os

from torch.utils.tensorboard import SummaryWriter

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


def log_to_comet(comet_exp, train_acc, val_acc, test_acc, epoch, out_str=None):
    metrics = {}
    for key in train_acc.keys():
        metrics.update({f"val_acc_{key}": val_acc[key],
                        f"train_acc_{key}": train_acc[key],
                        f"test_acc_{key}": test_acc[key]})
    comet_exp.log_metrics(metrics, epoch=epoch)
    if out_str:
        comet_exp.log_parameter("ckpt_path", out_str)


def setup_comet(args):
    if args.use_comet:
        import comet_ml
        experiment = comet_ml.Experiment(project_name="deepthinking",
                                         auto_param_logging=False,
                                         auto_metric_logging=False,
                                         disabled=False,
                                         parse_args=False)

        experiment.add_tag(args.train_mode)
        experiment.log_parameters(vars(args))
        return experiment
    return None


def setup_tb(train_log, output):
    writer = SummaryWriter(log_dir=f"{output}/runs/{train_log[:-4]}")
    return writer


def to_json(data_to_save, out_dir, log_file_name):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, log_file_name)

    if os.path.isfile(fname):
        with open(fname, "r") as fp:
            data_from_json = json.load(fp)
            num_entries = data_from_json["num entries"]
        data_from_json[num_entries] = data_to_save
        data_from_json["num entries"] += 1
        with open(fname, "w") as fp:
            json.dump(data_from_json, fp)
    else:
        data_from_json = {0: data_to_save, "num entries": 1}
        with open(fname, "w") as fp:
            json.dump(data_from_json, fp)


def write_to_tb(stats, stat_names, epoch, writer):
    for name, stat in zip(stat_names, stats):
        stat_name = os.path.join("val", name)
        writer.add_scalar(stat_name, stat, epoch)
