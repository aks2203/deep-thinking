import argparse
import glob
import json
from tabulate import tabulate


def get_trained_checkpoints(filepath, acc_filter):
    checkpoints = []
    num_checkpoints = 0
    num_trained = 0
    for f_name in glob.iglob(f"{filepath}/**/*training*/stats.json", recursive=True):
        with open(f_name, "r") as fp:
            print(f_name)
            data = json.load(fp)

        m = data["max_iters"]
        model_path = "/".join(f_name.split("/")[1:-1])
        if data["train_acc"][str(m)] > acc_filter:
            checkpoints.append((model_path, m, data["train_acc"][str(m)]))
            num_trained += 1
        num_checkpoints += 1
    return checkpoints, num_trained, num_checkpoints


def main():
    parser = argparse.ArgumentParser(description="Analysis parser")
    parser.add_argument("filepath", type=str)
    parser.add_argument("--filter", type=float, default=0,
                        help="cutoff for filtering by training acc?")
    args = parser.parse_args()
    checkpoints, num_trained, num_checkopints = get_trained_checkpoints(args.filepath, args.filter)
    head = ["Model Path", "Max Iters", "Training Acc"]

    print(f"In total, {num_checkopints} models are saved")
    print(f"Of those, {num_trained} trained, with training accuracy higher than {args.filter}")
    print(tabulate(checkpoints, headers=head))
    print("")
    print("Checkpoint directories in copy-pastable form:")
    for c in checkpoints:
        print(c[0])


if __name__ == "__main__":
    main()
