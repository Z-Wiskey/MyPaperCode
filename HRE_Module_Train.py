import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

import utils
from HRE_Module import HRE
from parse_args import args
from task_registry import expand_tasks, run_task


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def resolve_device():
    if args.device == "cuda" and not torch.cuda.is_available():
        print("cuda is unavailable, falling back to cpu")
        return torch.device("cpu")
    return torch.device(args.device)


DEVICE = resolve_device()


def prepare_training_inputs():
    vis_emb, poi_similarity, landuse_similarity, s_adj, d_adj, mobility = utils.load_data()

    poi_edge_index = torch.tensor(
        utils.create_graph(poi_similarity, args.importance_k), dtype=torch.long, device=DEVICE
    )
    s_edge_index = torch.tensor(
        utils.create_graph(s_adj, args.importance_k), dtype=torch.long, device=DEVICE
    )
    d_edge_index = torch.tensor(
        utils.create_graph(d_adj, args.importance_k), dtype=torch.long, device=DEVICE
    )
    n_edge_index = torch.tensor(
        utils.create_graph(landuse_similarity, args.importance_k), dtype=torch.long, device=DEVICE
    )

    mobility_tensor = torch.tensor(mobility, dtype=torch.float32, device=DEVICE)
    poi_similarity_tensor = torch.tensor(poi_similarity, dtype=torch.float32, device=DEVICE)
    landuse_similarity_tensor = torch.tensor(landuse_similarity, dtype=torch.float32, device=DEVICE)

    features = torch.tensor(vis_emb, dtype=torch.float32, device=DEVICE)
    features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-6)
    features = F.normalize(features, p=2, dim=1)
    features.requires_grad_(True)

    rel_emb = [torch.randn(args.embedding_size, device=DEVICE) for _ in range(4)]
    edge_index = [poi_edge_index, s_edge_index, d_edge_index, n_edge_index]
    return features, rel_emb, edge_index, mobility_tensor, poi_similarity_tensor, landuse_similarity_tensor


def mob_loss(s_emb, d_emb, mob):
    inner_prod = torch.mm(s_emb, d_emb.T)
    ps_hat = F.softmax(inner_prod, dim=-1)
    inner_prod = torch.mm(d_emb, s_emb.T)
    pd_hat = F.softmax(inner_prod, dim=-1)
    return torch.sum(-torch.mul(mob, torch.log(ps_hat)) - torch.mul(mob, torch.log(pd_hat)))


def general_loss(embeddings, similarity):
    inner_prod = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    return F.mse_loss(inner_prod, similarity)


def evaluate_task(embs, task_name, display=False):
    result = run_task(args.task_package, task_name, embs, args.data_path, display=display)
    mae, rmse, r2 = result
    return {"task": task_name, "mae": float(mae), "rmse": float(rmse), "r2": float(r2), "score": float(r2)}


def better_than(current_metrics, best_metrics):
    if best_metrics is None:
        return True

    if current_metrics["score"] != best_metrics["score"]:
        return current_metrics["score"] > best_metrics["score"]

    return current_metrics["rmse"] < best_metrics.get("rmse", float("inf"))


def save_best_outputs(embs, metrics):
    save_dir = Path(args.save_folder)
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.best_emb_path, embs)
    with open(args.best_metrics_path, "w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, indent=2)


def print_best_checkpoint(metrics):
    print("best checkpoint----------")
    print(f"selection_task: {metrics['task']}")
    print(f"epoch:          {metrics['epoch']}")
    print(f"MAE:            {metrics['mae']:.3f}")
    print(f"RMSE:           {metrics['rmse']:.3f}")
    print(f"R2:             {metrics['r2']:.3f}")
    print(f"train_loss:     {metrics['train_loss']:.6f}")


def log_eval(epoch, loss_value, metrics):
    display_epoch = epoch + 1
    print(
        f"epoch={display_epoch:04d} loss={loss_value:.6f} selection_task={metrics['task']} "
        f"rmse={metrics['rmse']:.4f} mae={metrics['mae']:.4f} r2={metrics['r2']:.4f}"
    )


def train(net, features, rel_emb, edge_index, mobility, poi_similarity, landuse_similarity):
    region_volume = torch.sum(mobility, dim=1)
    region_volume_log = torch.log(region_volume + 1e-6).detach()
    volume_predictor = torch.nn.Linear(args.embedding_size, 1).to(DEVICE)

    optimizer = optim.Adam(
        [
            {"params": net.parameters()},
            {"params": features, "lr": args.feature_learning_rate},
            {"params": volume_predictor.parameters(), "lr": args.aux_learning_rate},
        ],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    loss_fn_mse = torch.nn.MSELoss()
    loss_fn_reconstruction = torch.nn.MSELoss()

    best_metrics = None

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        region_emb, landuse_emb, poi_emb, s_emb, d_emb, loss_cross = net(features, rel_emb, edge_index)

        m_loss = mob_loss(s_emb, d_emb, mobility)
        poi_loss = loss_fn_reconstruction(torch.mm(poi_emb, poi_emb.T), poi_similarity)
        landuse_loss = general_loss(landuse_emb, landuse_similarity)

        pred_volume = volume_predictor(region_emb).squeeze()
        volume_loss = loss_fn_mse(pred_volume, region_volume_log)

        structural_loss = poi_loss + m_loss + landuse_loss + args.volume_loss_weight * volume_loss
        loss = structural_loss + loss_cross * args.cross_loss_weight

        loss.backward()
        optimizer.step()

        should_eval = (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1
        if not should_eval:
            continue

        with torch.no_grad():
            current_emb = region_emb.detach().cpu().numpy()
            current_metrics = evaluate_task(current_emb, args.selection_task, display=False)
            current_metrics["epoch"] = epoch + 1
            current_metrics["train_loss"] = float(loss.item())
            log_eval(epoch, loss.item(), current_metrics)
            if better_than(current_metrics, best_metrics):
                best_metrics = current_metrics.copy()
                save_best_outputs(current_emb, best_metrics)

    return best_metrics


def test():
    best_emb = np.load(args.best_emb_path, allow_pickle=True)

    for task_name in expand_tasks(args.task):
        print()
        print(f">>>>>>>>>>>>>>>>>   {task_name}")
        run_task(args.task_package, task_name, best_emb, args.data_path, display=True)


def main():
    set_seed(args.seed)
    print("device:", DEVICE)
    print(f"city: {args.city}")
    print(f"task: {args.task}")
    print(f"data_path: {args.data_path}")
    print(f"save_folder: {args.save_folder}")

    features, rel_emb, edge_index, mobility, poi_similarity, landuse_similarity = prepare_training_inputs()
    net = HRE(args.embedding_size, args.dropout, args.gcn_layers).to(DEVICE)

    print("training-----------------")
    net.train()
    best_metrics = train(net, features, rel_emb, edge_index, mobility, poi_similarity, landuse_similarity)

    if best_metrics is None:
        raise RuntimeError("Training finished without producing evaluation metrics.")

    print_best_checkpoint(best_metrics)

    print("downstream task test-----")
    test()


if __name__ == "__main__":
    main()
