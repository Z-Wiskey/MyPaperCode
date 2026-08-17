from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from parse_args import args


def _load_npy(data_path, file_name):
    return np.load(Path(data_path) / file_name, allow_pickle=True)


def load_data():
    data_path = args.data_path

    vis_path = Path(data_path) / args.vis_embedding
    if vis_path.exists():
        vis_emb = np.load(vis_path, allow_pickle=True)
    else:
        rng = np.random.default_rng(args.seed)
        vis_emb = rng.standard_normal((args.regions_num, args.embedding_size)).astype(np.float32)
        print(
            f"warning: {vis_path} not found, using seeded random initialization "
            f"with shape {vis_emb.shape}."
        )

    mobility_adj = _load_npy(data_path, args.mobility_adj).squeeze()
    mobility = mobility_adj.copy()
    mobility = mobility / np.mean(mobility)

    poi_similarity = _load_npy(data_path, args.poi_similarity)
    poi_similarity[np.isnan(poi_similarity)] = 0

    landuse_similarity = _load_npy(data_path, args.landuse_similarity)
    landuse_similarity[np.isnan(landuse_similarity)] = 0

    s_adj = mobility_adj.copy()
    d_adj = mobility_adj.T.copy()

    return vis_emb, poi_similarity, landuse_similarity, s_adj, d_adj, mobility


def graph_to_COO(similarity, importance_k):
    region_num = similarity.shape[0]
    top_k = min(importance_k, region_num)
    graph = np.eye(region_num, dtype=np.float32)

    for i in range(region_num):
        top_indices = np.argsort(similarity[:, i])[-top_k:]
        graph[top_indices, i] = 1
        graph[i, top_indices] = 1

    edge_index = sp.coo_matrix(graph)
    return np.vstack((edge_index.row, edge_index.col))


def create_graph(similarity, importance_k):
    return graph_to_COO(similarity, importance_k)


def pair_sample(neighbor):
    region_num = len(neighbor)
    positive = torch.zeros(region_num, dtype=torch.long)
    negative = torch.zeros(region_num, dtype=torch.long)

    for i in range(region_num):
        region_idx = np.random.randint(len(neighbor[i]))
        positive[i] = int(neighbor[i][region_idx])

    for i in range(region_num):
        neg_region = np.random.randint(region_num)
        while neg_region in neighbor[i] or neg_region == i:
            neg_region = np.random.randint(region_num)
        negative[i] = neg_region

    return positive, negative


def create_neighbor_graph(neighbor):
    region_num = len(neighbor)
    graph = np.eye(region_num, dtype=np.float32)

    for i in range(region_num):
        for region in neighbor[i]:
            graph[i, region] = 1
            graph[region, i] = 1

    sparse_graph = sp.coo_matrix(graph)
    return np.stack((sparse_graph.row, sparse_graph.col))
