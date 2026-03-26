from pathlib import Path

import numpy as np

from tasks_NewYork.common import predict_regression


def do_tasks(embs, data_path="./data_NewYork", display=True):
    if display:
        print("Service Call Prediction:")
    labels = np.load(Path(data_path) / "serviceCall_counts.npy", allow_pickle=True)
    return predict_regression(embs, labels, display=display)
