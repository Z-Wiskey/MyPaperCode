from pathlib import Path

import numpy as np

from parse_args import args
from task_registry import run_task


def predict_crime(emb):
    return run_task(args.task_package, "crime", emb, args.data_path, display=False)


def predict_checkin(emb):
    return run_task(args.task_package, "checkin", emb, args.data_path, display=False)


def predict_check(emb):
    return predict_checkin(emb)


def predict_servicecall(emb):
    return run_task(args.task_package, "servicecall", emb, args.data_path, display=False)


if __name__ == "__main__":
    emb_path = Path(args.best_emb_path)
    if not emb_path.exists():
        emb_path = Path("emb.npy")
    emb = np.load(emb_path, allow_pickle=True)
    print("crime", predict_crime(emb))
    print("checkin", predict_checkin(emb))
    print("servicecall", predict_servicecall(emb))
