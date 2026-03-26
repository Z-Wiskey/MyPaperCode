from pathlib import Path

import numpy as np

from parse_args import args
from task_registry import run_task


def predict_crime(emb):
    return run_task(args.task_package, "crime", emb, args.data_path, display=False)


def predict_check(emb):
    return run_task(args.task_package, "check", emb, args.data_path, display=False)


def clustering(emb):
    return run_task(args.task_package, "clustering", emb, args.data_path, display=False)


if __name__ == "__main__":
    emb_path = Path(args.best_emb_path)
    if not emb_path.exists():
        emb_path = Path("emb.npy")
    emb = np.load(emb_path, allow_pickle=True)
    print("crime", predict_crime(emb))
    print("check", predict_check(emb))
    print("clustering", clustering(emb))
