import utils
from parse_args import args
from task import predict_crime, clustering, predict_check
from HRE_Module import HRE

import random
from tqdm import tqdm
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F

seed = 2022
torch.manual_seed(seed=seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)
vis_emb, poi_similarity, s_adj, d_adj, mobility, neighbor = utils.load_data()
poi_edge_index = utils.create_graph(poi_similarity, args.importance_k)
s_edge_index = utils.create_graph(s_adj, args.importance_k)
d_edge_index = utils.create_graph(d_adj, args.importance_k)
n_edge_index = utils.create_neighbor_graph(neighbor)

poi_edge_index = torch.tensor(poi_edge_index, dtype=torch.long).to(args.device)
s_edge_index = torch.tensor(s_edge_index, dtype=torch.long).to(args.device)
d_edge_index = torch.tensor(d_edge_index, dtype=torch.long).to(args.device)
n_edge_index = torch.tensor(n_edge_index, dtype=torch.long).to(args.device)

mobility = torch.tensor(mobility, dtype=torch.float32).to(args.device)
poi_similarity = torch.tensor(
    poi_similarity, dtype=torch.float32).to(args.device)

# 视觉语义注入图结构节点信息
features = torch.from_numpy(vis_emb).to(args.device)
# 方案A: Z-score 标准化 (推荐)
features = (features - features.mean(0)) / (features.std(0) + 1e-6)
# 方案B: L2 正则化 (将模长缩放为1)
features = F.normalize(features, p=2, dim=1)
features.requires_grad = True  # 关键：允许求导

# features = torch.randn(args.regions_num, args.embedding_size).to(args.device)
# poi_r = torch.from_numpy(vis_emb).to(args.device)
# s_r = torch.from_numpy(vis_emb).to(args.device)
# d_r = torch.from_numpy(vis_emb).to(args.device)
# n_r = torch.from_numpy(vis_emb).to(args.device)
poi_r = torch.randn(args.embedding_size).to(args.device)
s_r = torch.randn(args.embedding_size).to(args.device)
d_r = torch.randn(args.embedding_size).to(args.device)
n_r = torch.randn(args.embedding_size).to(args.device)
rel_emb = [poi_r, s_r, d_r, n_r]
edge_index = [poi_edge_index, s_edge_index, d_edge_index, n_edge_index]


def mob_loss(s_emb, d_emb, mob):
    inner_prod = torch.mm(s_emb, d_emb.T)
    ps_hat = F.softmax(inner_prod, dim=-1)
    inner_prod = torch.mm(d_emb, s_emb.T)
    pd_hat = F.softmax(inner_prod, dim=-1)
    loss = torch.sum(-torch.mul(mob, torch.log(ps_hat)) -
                     torch.mul(mob, torch.log(pd_hat)))
    return loss

# def mob_loss(s_emb, d_emb, mob):
#     # 第一部分：Source -> Destination
#     inner_prod_s = torch.mm(s_emb, d_emb.T)
#     # 使用 log_softmax 直接获得 log(probability)，数值更稳定
#     log_ps_hat = F.log_softmax(inner_prod_s, dim=-1)
#
#     # 第二部分：Destination -> Source
#     inner_prod_d = torch.mm(d_emb, s_emb.T)
#     log_pd_hat = F.log_softmax(inner_prod_d, dim=-1)
#
#     # 计算 Loss
#     loss = torch.sum(-torch.mul(mob, log_ps_hat) -
#                      torch.mul(mob, log_pd_hat))
#     return loss


def train(net):
    # 1. 在循环外预计算每个区域的总流量 (作为 Ground Truth)
    # mobility 是 (N, N) 的矩阵，sum(dim=1) 得到每个区域的总流出量
    # 取 log 是为了平滑长尾分布，防止数值过大
    region_volume = torch.sum(mobility, dim=1)
    region_volume_log = torch.log(region_volume + 1e-6).detach()  # 加上 epsilon 防止 log(0)

    # 简单定义一个线性层用于预测流量 (可以放在 HRE 模型里，也可以在这里临时定义)
    # 为了方便，建议直接在 HRE_Module.py 里加，或者在这里用 embedding 的模长来约束
    # 方案 A: 约束 Embedding 的模长 (Norm) 与流量正相关
    # 方案 B: 加一个小的预测头 (推荐，更灵活)
    volume_predictor = torch.nn.Linear(args.embedding_size, 1).to(args.device)

    optimizer = optim.Adam(
        [
            {'params': net.parameters()},
            {'params': features, 'lr': 1e-3},
            {'params': volume_predictor.parameters(), 'lr': 1e-3}  # 新增
        ], lr=args.learning_rate, weight_decay=5e-3)

    loss_fn_mse = torch.nn.MSELoss()  # 用于回归流量
    loss_fn1 = torch.nn.TripletMarginLoss()
    loss_fn2 = torch.nn.MSELoss()

    best_rmse = 10000
    best_mae = 10000
    best_r2 = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        region_emb, n_emb, poi_emb, s_emb, d_emb, loss_cross = net(
            features, rel_emb, edge_index)

        pos_idx, neg_idx = utils.pair_sample(neighbor)

        geo_loss = loss_fn1(n_emb, n_emb[pos_idx], n_emb[neg_idx])

        m_loss = mob_loss(s_emb, d_emb, mobility)

        poi_loss = loss_fn2(torch.mm(poi_emb, poi_emb.T), poi_similarity)

        # === 新增代码开始 ===
        # 预测该区域的流量等级
        pred_volume = volume_predictor(region_emb).squeeze()

        # 计算流量损失 (Volume Loss)
        # 这迫使 Embedding 包含能够线性映射回“总流量”的信息
        l_volume = loss_fn_mse(pred_volume, region_volume_log)
        # === 新增代码结束 ===

        l_str = poi_loss + m_loss + geo_loss + 0.5 * l_volume

        # 动态调整 lambda_cross

        lambda_cross = 1  # 之后再加入，且权重保持较小

        loss = l_str + loss_cross * lambda_cross

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            mae, rmse, r2 = predict_crime(region_emb.detach().cpu().numpy())
            # mae, rmse, r2 = predict_check(region_emb.detach().cpu().numpy())
            # nmi, ari = clustering(region_emb.detach().cpu().numpy())
            # print(nmi, ari)
            if rmse < best_rmse and mae < best_mae and best_r2 < r2:
                best_rmse = rmse
                best_mae = mae
                best_r2 = r2
                best_epoch = epoch
            print(epoch, rmse, mae, r2, loss.item())
            # np.save('emb', region_emb.detach().cpu().numpy())

    print('best_rmse:', best_rmse)
    print('best_mae:', best_mae)
    print('best_r2:', best_r2)
    print('best_epoch:', best_epoch)


def test(net):
    region_emb, _, _, _, _, _ = net(features, rel_emb, edge_index, False)
    print('>>>>>>>>>>>>>>>>>   crime')
    mae, rmse, r2 = predict_crime(region_emb.detach().cpu().numpy())
    print("MAE:  %.3f" % mae)
    print("RMSE: %.3f" % rmse)
    print("R2:   %.3f" % r2)
    print('>>>>>>>>>>>>>>>>>   check')
    mae, rmse, r2 = predict_check(region_emb.detach().cpu().numpy())
    print("MAE:  %.3f" % mae)
    print("RMSE: %.3f" % rmse)
    print("R2:   %.3f" % r2)
    print('>>>>>>>>>>>>>>>>>   clustering')
    nmi, ari = clustering(region_emb.detach().cpu().numpy())
    print("NMI: %.3f" % nmi)
    print("ARI: %.3f" % ari)

    np.save('emb', region_emb.detach().cpu().numpy())


if __name__ == '__main__':
    net = HRE(args.embedding_size, args.dropout,
              args.gcn_layers).to(args.device)
    print('training-----------------')
    net.train()
    train(net)
    net.eval()
    print('downstream task test-----')
    test(net)
