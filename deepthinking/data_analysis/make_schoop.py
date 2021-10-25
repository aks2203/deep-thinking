""" make_schoop.py
    For generating schoopy plots

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""

import argparse
from datetime import datetime

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from make_table import get_table


def get_schoopy_plot(table, error_bars=True):
    fig, ax = plt.subplots(figsize=(20, 9))

    models = set(table.model)
    test_datas = set(table.test_data)
    alphas = set(table.alpha)

    sns.lineplot(data=table,
                 x="test_iter",
                 y="test_acc_mean",
                 hue="model",
                 size="alpha",
                 sizes=(2, 8),
                 style="test_data" if len(test_datas) > 1 else None,
                 palette="dark",
                 dashes=True,
                 units=None,
                 legend="auto",
                 ax=ax)

    if error_bars and "test_acc_sem" in table.keys():
        for model in models:
            for test_data in test_datas:
                for alpha in alphas:
                    data = table[(table.model == model) &
                                 (table.test_data == test_data) &
                                 (table.alpha == alpha)]
                    plt.fill_between(data.test_iter,
                                     data.test_acc_mean - data.test_acc_sem,
                                     data.test_acc_mean + data.test_acc_sem,
                                     alpha=0.1, color="k")

    tr = table.max_iters.max()  # training regime number
    ax.fill_between([0, tr], [105, 105], alpha=0.3, label="Training Regime")
    return ax


def main():
    parser = argparse.ArgumentParser(description="Analysis parser")
    parser.add_argument("--alpha_list", type=float, nargs="+", default=None,
                        help="only plot models with alphas in given list")
    parser.add_argument("filepath", type=str)
    parser.add_argument("--filter", type=float, default=None,
                        help="cutoff for filtering by training acc?")
    parser.add_argument("--plot_name", type=str, default=None, help="where to save image?")
    parser.add_argument("--max_iters_list", type=int, nargs="+", default=None,
                        help="only plot models with max iters in given list")
    parser.add_argument("--model_list", type=str, nargs="+", default=None,
                        help="only plot models with model name in given list")
    parser.add_argument("--width_list", type=str, nargs="+", default=None,
                        help="only plot models with widths in given list")
    parser.add_argument("--max", action="store_true", help="add max values to table?")
    parser.add_argument("--min", action="store_true", help="add min values too table?")
    parser.add_argument("--xlim", type=float, nargs="+", default=None, help="x limits for plotting")
    parser.add_argument("--ylim", type=float, nargs="+", default=None, help="y limits for plotting")
    args = parser.parse_args()

    if args.plot_name is None:
        now = datetime.now().strftime("%m%d-%H.%M")
        args.plot_name = f"schoop{now}.png"
        plot_title = "Schoopy Plot"
    else:
        plot_title = args.plot_name[:-4]

    # get table of results
    table = get_table(args.filepath,
                      args.max,
                      args.min,
                      filter_at=args.filter,
                      max_iters_list=args.max_iters_list,
                      alpha_list=args.alpha_list,
                      width_list=args.width_list,
                      model_list=args.model_list)

    # reformat and reindex table for plotting purposes
    table.columns = table.columns.map("_".join)
    table.columns.name = None
    table = table.reset_index()
    print(table.round(2).to_markdown())
    ax = get_schoopy_plot(table)

    ax.legend(fontsize=26, loc="upper left", bbox_to_anchor=(1.0, 0.8))
    x_max = table.test_iter.max()
    x = np.arange(20, x_max + 1, 10 if (x_max <= 100) else 100)
    ax.tick_params(axis="y", labelsize=34)
    ax.set_xticks(x)
    ax.set_xticklabels(x, fontsize=34, rotation=37)
    if args.xlim is None:
        ax.set_xlim([x.min() - 0.5, x.max() + 0.5])
    else:
        ax.set_xlim(args.xlim)
    if args.ylim is None:
        ax.set_ylim([0, 103])
    else:
        ax.set_ylim(args.ylim)
    ax.set_xlabel("Test-Time Iterations", fontsize=34)
    ax.set_ylabel("Accuracy (%)", fontsize=34)
    ax.set_title(plot_title, fontsize=34)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tight_layout()

    plt.savefig(args.plot_name)
    # plt.show()


if __name__ == "__main__":
    main()
