""" make_table.py
    For generating pivot tables

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""

import argparse
import copy
import glob
import json

import pandas as pd


def get_little_df_from_one_run(data_dict):
    num_new_dicts = len(data_dict["test_iters"])
    test_iters = data_dict["test_iters"]
    out_dict = {}
    for i in range(num_new_dicts):
        new_dict_i = copy.deepcopy(data_dict)
        new_dict_i["test_acc"] = data_dict["test_acc"][str(test_iters[i])] \
            if data_dict["test_acc"] else 0
        new_dict_i["val_acc"] = data_dict["val_acc"][str(test_iters[i])] \
            if data_dict["val_acc"] else 0
        new_dict_i["train_acc"] = data_dict["train_acc"][str(test_iters[i])] \
            if data_dict["train_acc"] else 0
        new_dict_i["test_iter"] = test_iters[i]
        out_dict[i] = new_dict_i
    little_df = pd.DataFrame.from_dict(out_dict, orient="index")
    return little_df


def get_df(filepath, acc_filter):
    pd.set_option("display.max_rows", None)
    df = pd.DataFrame()
    num_checkpoints = 0
    for f_name in glob.iglob(f"{filepath}/**/*testing*/stats.json", recursive=True):
        num_checkpoints += 1
        with open(f_name, "r") as fp:
            data = json.load(fp)
        if acc_filter is not None:
            m = data["max_iters"]
            if data["train_acc"][str(m)] > acc_filter:
                df = df.append(get_little_df_from_one_run(data))
        else:
            df = df.append(get_little_df_from_one_run(data))
    num_trained = len(df)
    return df, num_checkpoints, num_trained


def get_table(filepath, disp_max, disp_min, filter_at=None, max_iters_list=None,
              alpha_list=None, width_list=None, model_list=None):
    pd.set_option("display.max_rows", None)
    df, num_checkpoints, num_trained = get_df(filepath, filter_at)
    df["count"] = 1

    if max_iters_list:
        frames = []
        for max_iters in max_iters_list:
            frames.append(df[df["max_iters"] == int(max_iters)])
        df = pd.concat(frames)
    if alpha_list:
        frames = []
        for alpha in alpha_list:
            frames.append(df[df["alpha"] == float(alpha)])
        df = pd.concat(frames)
    if width_list:
        frames = []
        for width in width_list:
            frames.append(df[df["model"].str.contains(str(width))])
        df = pd.concat(frames)
    if model_list:
        frames = []
        for model in model_list:
            frames.append(df[df["model"].str.contains(model)])
        df = pd.concat(frames)

    # df = df[df.model_path.str.contains("best")]
    index = ["model", "test_data", "max_iters", "alpha", "test_mode", "test_iter"]

    values = ["mean", "sem"]
    if disp_max:
        values.append("max")
    if disp_min:
        values.append("min")
    df.drop_duplicates(subset=["model_path", "test_iter", "test_data", "test_mode"], inplace=True)
    try:
        table = pd.pivot_table(df, index=index, aggfunc={"train_acc": values,
                                                         "val_acc": values,
                                                         "test_acc": values,
                                                         "count": "count"})
    except:
        raise KeyError("No data in the table. Check the path to test results. "
                       "Possibly, there are no results from models that trained "
                       "with accuracy above the filter value.")
    return table


def main():
    parser = argparse.ArgumentParser(description="Analysis parser")
    parser.add_argument("filepath", type=str)
    parser.add_argument("--filter", type=float, default=None,
                        help="cutoff for filtering by training acc?")
    parser.add_argument("--max_iters_list", type=int, nargs="+", default=None,
                        help="only plot models with max iters in given list")
    parser.add_argument("--model_list", type=str, nargs="+", default=None,
                        help="only plot models with model name in given list")
    parser.add_argument("--width_list", type=str, nargs="+", default=None,
                        help="only plot models with widths in given list")
    parser.add_argument("--alpha_list", type=float, nargs="+", default=None,
                        help="only plot models with alphas in given list")
    parser.add_argument("--max", action="store_true", help="add max values to table?")
    parser.add_argument("--min", action="store_true", help="add min values too table?")
    parser.add_argument("--save_name", type=str, default=None, help="add min values too table?")

    args = parser.parse_args()
    table = get_table(args.filepath,
                      args.max,
                      args.min,
                      filter_at=args.filter,
                      max_iters_list=args.max_iters_list,
                      alpha_list=args.alpha_list,
                      width_list=args.width_list,
                      model_list=args.model_list)

    table = table.round(3)
    print(table.to_markdown())

    if args.save_name is not None:
        table.to_csv(args.save_name)


if __name__ == "__main__":
    main()
