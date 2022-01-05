import argparse
import glob
import json
import os.path

from omegaconf import OmegaConf
from tabulate import tabulate


def get_trained_checkpoints(filepath, model_list, alpha_list):
    checkpoints = []
    num_checkpoints = 0
    for f_name in glob.iglob(f"{filepath}/**/*training*/", recursive=True):
        cfg_name = os.path.join(f_name, ".hydra/config.yaml")
        cfg = OmegaConf.load(cfg_name)
        if model_list is None or cfg.problem.model.model in model_list:
            if alpha_list is None or cfg.problem.hyp.alpha in alpha_list:
                checkpoints.append((cfg.run_id,
                                    cfg.problem.model.model,
                                    cfg.problem.hyp.alpha))
                num_checkpoints += 1
    return checkpoints, num_checkpoints


def main():
    parser = argparse.ArgumentParser(description="Analysis parser")
    parser.add_argument("filepath", type=str)
    parser.add_argument("--alpha_list", type=float, nargs="+", default=None,
                        help="only plot models with model name in given list")
    parser.add_argument("--filter", type=float, default=0,
                        help="cutoff for filtering by training acc?")
    parser.add_argument("--model_list", type=str, nargs="+", default=None,
                        help="only plot models with model name in given list")
    args = parser.parse_args()
    checkpoints, num_checkopints = get_trained_checkpoints(args.filepath,
                                                           args.model_list,
                                                           args.alpha_list)
    head = ["Model Path", "Model", "Alpha"]

    print(f"I see {num_checkopints} runs")
    print(tabulate(checkpoints, headers=head))


if __name__ == "__main__":
    main()
