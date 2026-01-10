import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv


# class RelationGCN(nn.Module):
#     def __init__(self, embedding_size, dropout, gcn_layers):
#         super(RelationGCN, self).__init__()
#         self.gcn_layers = gcn_layers
#         self.dropout = dropout
#
#         self.gcns = nn.ModuleList([GCNConv(in_channels=embedding_size, out_channels=embedding_size)
#                                    for _ in range(self.gcn_layers)])
#         self.bns = nn.ModuleList([nn.BatchNorm1d(embedding_size)
#                                  for _ in range(self.gcn_layers - 1)])
#         self.relation_transformation = nn.ModuleList([nn.Linear(embedding_size, embedding_size)
#                                                       for _ in range(self.gcn_layers)])
#
#     def forward(self, features, rel_emb, edge_index, is_training=True):
#         n_emb = features
#         poi_emb = features
#         s_emb = features
#         d_emb = features
#         poi_r, s_r, d_r, n_r = rel_emb
#         poi_edge_index, s_edge_index, d_edge_index, n_edge_index = edge_index
#         for i in range(self.gcn_layers - 1):
#             tmp = n_emb
#             n_emb = tmp + F.leaky_relu(self.bns[i](
#                 self.gcns[i](torch.multiply(n_emb, n_r), n_edge_index)))
#             n_r = self.relation_transformation[i](n_r)
#             if is_training:
#                 n_emb = F.dropout(n_emb, p=self.dropout)
#
#             tmp = poi_emb
#             poi_emb = tmp + F.leaky_relu(self.bns[i](
#                 self.gcns[i](torch.multiply(poi_emb, poi_r), poi_edge_index)))
#             poi_r = self.relation_transformation[i](poi_r)
#             if is_training:
#                 poi_emb = F.dropout(poi_emb, p=self.dropout)
#
#             tmp = s_emb
#             s_emb = tmp + F.leaky_relu(self.bns[i](
#                 self.gcns[i](torch.multiply(s_emb, s_r), s_edge_index)))
#             s_r = self.relation_transformation[i](s_r)
#             if is_training:
#                 s_emb = F.dropout(s_emb, p=self.dropout)
#
#             tmp = d_emb
#             d_emb = tmp + F.leaky_relu(self.bns[i](
#                 self.gcns[i](torch.multiply(d_emb, d_r), d_edge_index)))
#             d_r = self.relation_transformation[i](d_r)
#             if is_training:
#                 d_emb = F.dropout(d_emb, p=self.dropout)
#
#         n_emb = self.gcns[-1](torch.multiply(n_emb, n_r), n_edge_index)
#         poi_emb = self.gcns[-1](torch.multiply(poi_emb, poi_r), poi_edge_index)
#         s_emb = self.gcns[-1](torch.multiply(s_emb, s_r), s_edge_index)
#         d_emb = self.gcns[-1](torch.multiply(d_emb, d_r), d_edge_index)
#
#         # rel update
#         n_r = self.relation_transformation[-1](n_r)
#         poi_r = self.relation_transformation[-1](poi_r)
#         s_r = self.relation_transformation[-1](s_r)
#         d_r = self.relation_transformation[-1](d_r)
#
#         return n_emb, poi_emb, s_emb, d_emb, n_r, poi_r, s_r, d_r

class RelationGAT(nn.Module):
    def __init__(self, embedding_size, dropout, gcn_layers):
        super(RelationGAT, self).__init__()
        self.gcn_layers = gcn_layers
        self.dropout = dropout

        # [修改点2] 将 GCNConv 替换为 GATConv
        # 参考 FlexiReg 代码，GATConv 能够处理注意力权重
        # heads=1 保证输出维度仍为 embedding_size，与后续模块兼容
        self.gats = nn.ModuleList([
            GATConv(in_channels=embedding_size,
                    out_channels=embedding_size,
                    heads=1,
                    dropout=dropout)  # GAT内部也有dropout机制，对应论文公式
            for _ in range(self.gcn_layers)
        ])

        self.bns = nn.ModuleList([nn.BatchNorm1d(embedding_size)
                                  for _ in range(self.gcn_layers - 1)])

        self.relation_transformation = nn.ModuleList([nn.Linear(embedding_size, embedding_size)
                                                      for _ in range(self.gcn_layers)])

    def forward(self, features, rel_emb, edge_index, is_training=True):
        n_emb = features
        poi_emb = features
        s_emb = features
        d_emb = features
        poi_r, s_r, d_r, n_r = rel_emb
        poi_edge_index, s_edge_index, d_edge_index, n_edge_index = edge_index

        for i in range(self.gcn_layers - 1):
            # ----------------- Neighbor Relation -----------------
            tmp = n_emb
            # [修改点3] 调用 self.gats[i]
            # 论文公式 (63-69): GAT 自动计算注意力系数 alpha_ij 并聚合
            n_emb_out = self.gats[i](torch.multiply(n_emb, n_r), n_edge_index)
            n_emb = tmp + F.leaky_relu(self.bns[i](n_emb_out))

            n_r = self.relation_transformation[i](n_r)
            if is_training:
                n_emb = F.dropout(n_emb, p=self.dropout)

            # ----------------- POI Relation -----------------
            tmp = poi_emb
            poi_emb_out = self.gats[i](torch.multiply(poi_emb, poi_r), poi_edge_index)
            poi_emb = tmp + F.leaky_relu(self.bns[i](poi_emb_out))

            poi_r = self.relation_transformation[i](poi_r)
            if is_training:
                poi_emb = F.dropout(poi_emb, p=self.dropout)

            # ----------------- Source Relation -----------------
            tmp = s_emb
            s_emb_out = self.gats[i](torch.multiply(s_emb, s_r), s_edge_index)
            s_emb = tmp + F.leaky_relu(self.bns[i](s_emb_out))

            s_r = self.relation_transformation[i](s_r)
            if is_training:
                s_emb = F.dropout(s_emb, p=self.dropout)

            # ----------------- Destination Relation -----------------
            tmp = d_emb
            d_emb_out = self.gats[i](torch.multiply(d_emb, d_r), d_edge_index)
            d_emb = tmp + F.leaky_relu(self.bns[i](d_emb_out))

            d_r = self.relation_transformation[i](d_r)
            if is_training:
                d_emb = F.dropout(d_emb, p=self.dropout)

        # ----------------- Final Layer -----------------
        # 最后一层通常不需要 BatchNorm 和残差 (视具体网络设计而定，保持与原代码逻辑一致)
        n_emb = self.gats[-1](torch.multiply(n_emb, n_r), n_edge_index)
        poi_emb = self.gats[-1](torch.multiply(poi_emb, poi_r), poi_edge_index)
        s_emb = self.gats[-1](torch.multiply(s_emb, s_r), s_edge_index)
        d_emb = self.gats[-1](torch.multiply(d_emb, d_r), d_edge_index)

        # rel update
        n_r = self.relation_transformation[-1](n_r)
        poi_r = self.relation_transformation[-1](poi_r)
        s_r = self.relation_transformation[-1](s_r)
        d_r = self.relation_transformation[-1](d_r)

        return n_emb, poi_emb, s_emb, d_emb, n_r, poi_r, s_r, d_r

class CrossLayer(nn.Module):
    def __init__(self, embedding_size):
        super(CrossLayer, self).__init__()
        self.alpha_n = nn.Parameter(torch.tensor(0.95))
        self.alpha_poi = nn.Parameter(torch.tensor(0.95))
        self.alpha_d = nn.Parameter(torch.tensor(0.95))
        self.alpha_s = nn.Parameter(torch.tensor(0.95))

        self.attn = nn.MultiheadAttention(
            embed_dim=embedding_size, num_heads=4)

    def forward(self, n_emb, poi_emb, s_emb, d_emb):
        stk_emb = torch.stack((n_emb, poi_emb, d_emb, s_emb))
        fusion, _ = self.attn(stk_emb, stk_emb, stk_emb)

        n_f = fusion[0] * self.alpha_n + (1 - self.alpha_n) * n_emb
        poi_f = fusion[1] * self.alpha_poi + (1 - self.alpha_poi) * poi_emb
        d_f = fusion[2] * self.alpha_d + (1 - self.alpha_d) * d_emb
        s_f = fusion[3] * self.alpha_s + (1 - self.alpha_s) * s_emb

        return n_f, poi_f, s_f, d_f


class AttentionFusionLayer(nn.Module):
    def __init__(self, embedding_size):
        super(AttentionFusionLayer, self).__init__()
        self.q = nn.Parameter(torch.randn(embedding_size))
        self.fusion_lin = nn.Linear(embedding_size, embedding_size)

    def forward(self, n_f, poi_f, s_f, d_f):
        n_w = torch.mean(torch.sum(F.leaky_relu(
            self.fusion_lin(n_f)) * self.q, dim=1))
        poi_w = torch.mean(torch.sum(F.leaky_relu(
            self.fusion_lin(poi_f)) * self.q, dim=1))
        s_w = torch.mean(torch.sum(F.leaky_relu(
            self.fusion_lin(s_f)) * self.q, dim=1))
        d_w = torch.mean(torch.sum(F.leaky_relu(
            self.fusion_lin(d_f)) * self.q, dim=1))

        w_stk = torch.stack((n_w, poi_w, s_w, d_w))
        w = torch.softmax(w_stk, dim=0)

        region_feature = w[0] * n_f + w[1] * poi_f + w[2] * s_f + w[3] * d_f
        return region_feature


class CrossModalFusion(nn.Module):
    def __init__(self, embedding_size, temperature=0.07):
        super(CrossModalFusion, self).__init__()
        self.embedding_size = embedding_size
        self.temperature = temperature

        # 1. 特征预变换 (Feature Transformation)
        # 将原始视觉特征映射到更高的语义空间，使其更适合与结构特征融合
        self.vis_transform = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.LeakyReLU(),
            nn.Linear(embedding_size, embedding_size)
        )

        # 2. 投影头 (用于对比损失)
        self.proj_vis = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.LeakyReLU()
        )
        self.proj_str = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.LeakyReLU()
        )

        # 3. 注意力融合 (Attention)
        self.W_Q = nn.Linear(embedding_size, embedding_size)
        self.W_K = nn.Linear(embedding_size, embedding_size)
        self.W_V = nn.Linear(embedding_size, embedding_size)

        # 4. 动态门控机制 (Dynamic Gating)
        # 输入: [结构特征, 视觉补充特征] -> 输出: 门控值 (0~1)
        # 这允许模型针对每个节点单独决定融合多少视觉信息
        self.gate_net = nn.Sequential(
            nn.Linear(2 * embedding_size, 1),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(embedding_size)

    def contrastive_loss(self, z_v, z_s):
        z_v = F.normalize(z_v, dim=1)
        z_s = F.normalize(z_s, dim=1)

        # 计算相似度
        sim_matrix = torch.mm(z_s, z_v.T) / self.temperature
        labels = torch.arange(z_v.size(0)).to(z_v.device)

        loss_s2v = F.cross_entropy(sim_matrix, labels)
        loss_v2s = F.cross_entropy(sim_matrix.T, labels)

        return loss_s2v + loss_v2s

    def forward(self, h_vis, h_str):
        # -------------------------------------------------------
        # Part 1: 跨模态一致性 Loss (改进：梯度截断)
        # -------------------------------------------------------
        # 我们只希望调整 h_str 来对齐 h_vis，不希望 h_vis 被带跑偏
        # 所以这里对 h_vis 进行 detach，将其视为固定的 Target
        z_v = self.proj_vis(h_vis.detach())
        z_s = self.proj_str(h_str)

        loss_cross = self.contrastive_loss(z_v, z_s)

        # -------------------------------------------------------
        # Part 2: 特征变换与注意力
        # -------------------------------------------------------
        # 对视觉特征进行升维变换
        h_vis_trans = self.vis_transform(h_vis)

        # Norm
        h_vis_norm = self.norm(h_vis_trans)
        h_str_norm = self.norm(h_str)

        Q = self.W_Q(h_str_norm)
        K = self.W_K(h_vis_norm)
        V = self.W_V(h_vis_norm)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embedding_size ** 0.5)
        A_sv = torch.softmax(scores, dim=-1)
        Z_sv = torch.matmul(A_sv, V)  # [N, D]

        # -------------------------------------------------------
        # Part 3: 动态门控残差融合 (Dynamic Gated Residual)
        # -------------------------------------------------------
        # 计算每个节点的门控值
        # 如果 Z_sv (视觉信息) 与 h_str (结构信息) 冲突或无效，
        # gate 会趋向于 0，退化为纯结构模型
        gate_input = torch.cat([h_str, Z_sv], dim=1)  # [N, 2D]
        gate = self.gate_net(gate_input)  # [N, 1]

        # 最终融合
        final_emb = h_str + gate * Z_sv
        final_emb = self.norm(final_emb)

        return final_emb, loss_cross


class HRE(nn.Module):
    def __init__(self, embedding_size, dropout, gcn_layers):
        super(HRE, self).__init__()

        # self.relation_gcns = RelationGCN(embedding_size, dropout, gcn_layers)
        self.relation_gcns = RelationGAT(embedding_size, dropout, gcn_layers)

        self.cross_layer = CrossLayer(embedding_size)

        self.fusion_layer = AttentionFusionLayer(embedding_size)

        # [新增] 跨模态融合模块
        self.cross_modal_fusion = CrossModalFusion(embedding_size)

    def forward(self, features, rel_emb, edge_index, is_training=True):
        # features 即为 h_vis (视觉嵌入)
        h_vis = features

        # 1. 结构学习 (GAT/GCN)
        poi_emb, s_emb, d_emb, n_emb, poi_r, s_r, d_r, n_r = self.relation_gcns(
            features, rel_emb, edge_index, is_training)

        n_f, poi_f, s_f, d_f = self.cross_layer(n_emb, poi_emb, s_emb, d_emb)

        # 3. 结构特征融合 (Fusion Layer) -> 得到 h_str
        # 这是纯粹基于结构模态学习到的区域表示
        h_str = self.fusion_layer(n_f, poi_f, s_f, d_f)

        # 4. [新增] 跨模态一致性增强与融合
        # 输入: 原始视觉特征 h_vis, 学习到的结构特征 h_str
        # 输出: 最终区域嵌入 final_emb, 跨模态损失 loss_cross
        final_emb, loss_cross = self.cross_modal_fusion(h_vis, h_str)

        # 更新特定关系的特征用于辅助损失计算 (保持原有逻辑)
        # 注意：这里使用 final_emb 还是 h_str 取决于后续任务是基于融合特征还是结构特征
        # 通常下游任务用 final_emb，但结构重建损失(Mobility等)可能需要用结构特征计算
        # 这里为了不破坏原有的结构重建逻辑，辅助变量仍然基于 h_str 或者你可以尝试改为 final_emb
        n_f = torch.multiply(h_str, n_r)
        poi_f = torch.multiply(h_str, poi_r)
        s_f = torch.multiply(h_str, s_r)
        d_f = torch.multiply(h_str, d_r)

        return final_emb, n_f, poi_f, s_f, d_f, loss_cross
