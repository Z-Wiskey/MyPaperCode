import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize


def _load_labels(data_path):
    with open(Path(data_path) / "mh_cd.json", "r", encoding="utf-8") as file_obj:
        mh_cd = json.load(file_obj)

    region_num = len(mh_cd)
    labels = np.zeros(region_num)
    for i in range(region_num):
        labels[i] = mh_cd[str(i)]
    return labels


def classify(embs, labels, random_state=3):
    embs_norm = normalize(embs, norm="l2", axis=1)
    n_clusters = len(np.unique(labels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    pred_labels = kmeans.fit_predict(embs_norm)
    nmi = normalized_mutual_info_score(labels, pred_labels)
    ari = adjusted_rand_score(labels, pred_labels)
    return nmi, ari


def do_tasks(embs, data_path="./data_NewYork", display=True):
    if display:
        print("Functional Zone Clustering:")
    labels = _load_labels(data_path)
    nmi, ari = classify(embs, labels)
    if display:
        print(f"NMI: {nmi:.3f}")
        print(f"ARI: {ari:.3f}")
    return nmi, ari
