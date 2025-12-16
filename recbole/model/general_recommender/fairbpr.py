# -*- coding: utf-8 -*-
# @Time   : 2020/6/25
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/9/16
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

r"""
BPR
################################################
Reference:
    Steffen Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback." in UAI 2009.
"""

import torch
import torch.nn as nn

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.layers import MLPLayers
from recbole.model.loss import BPRLoss
from recbole.utils import InputType
from recbole.utils.fair_utils import args2class, forward_rq_item_epoch, gen_extra_embedding
from recbole.rq.models.rqvae import RQVAE

class FairBPR(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way."""

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(FairBPR, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.loss = BPRLoss()
        args2class(self, config)
        self.rq_model_item = RQVAE(
            in_dim=self.embedding_size,
            num_emb_list=[64, 32, 32],
            e_dim=16,
            layers=[64, 32, 16],
            dropout_prob=0.1,
            bn=False,
            loss_type='mse',
            quant_loss_weight=1,
            beta=0.25,
            kmeans_init=True,
            kmeans_iters=100,
            sk_epsilons=[0.0, 0.0, 0.0],
            sk_iters=50,
            pop_dim=self.embedding_size
        )
        self.item_strong_dim = self.embedding_size * ((len(self.eInfo['item'])) - 1)
        self.item_strong_info = nn.Parameter()
        self.fusion_side_info = MLPLayers(
            [self.item_strong_dim, self.embedding_size],
            dropout=0.2, activation="sigmoid", last_activation=False
        )
        self.side_info_ln = nn.LayerNorm(self.embedding_size)
        self.fusion_side_info = nn.Sequential(
            self.fusion_side_info,
            self.side_info_ln
        )
        self.fusion_gate_layer = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),  # 输入维度减半
            nn.Tanh(),
            nn.Linear(self.embedding_size, 1),
            nn.Sigmoid()
        )
        self.NEG_PREFIX = "neg_"


        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.item_extra_embedding, self.user_extra_embedding = (
            gen_extra_embedding(self.eInfo, self.embedding_size, self.device)
        )

    def process_item_side_info(self, interaction, excluded_info:list):
        res = []
        for extra_info in self.eInfo['item']:
            item = interaction[extra_info]
            if extra_info in excluded_info:
                continue
            emb = self.extra_embedding_forward(extra_info, item)
            res.append(emb)
        res.append(self.item_embedding(interaction[self.ITEM_ID]))
        return torch.concat(res, dim=-1)

    def get_fused_embeddings(self, id_embeddings, side_embeddings):
        """
        利用门控机制融合 ID 和 Side Info
        """
        # ============================================================
        # 【修改点】: 只使用 ID Embedding 来计算门控系数
        # 逻辑：根据 Item 自身的特性（如是否热门、是否训练充分）来决定融合比例
        # ============================================================
        gate_input = id_embeddings

        # 计算门控系数 g: [N, 1]
        gate = self.fusion_gate_layer(gate_input)

        # 加权融合 (保持不变)
        # E_final = (1 - g) * ID + g * Side
        fused_embeddings = (1 - gate) * id_embeddings + gate * side_embeddings

        return fused_embeddings

    def get_batch_mlp_input(self, interaction, prefix: str):
        """
        获取指定 Item 的 3个 Side Info Embedding 并拼接，作为 MLP 的输入
        """
        res = []
        # 遍历 3 个 Side Info (例如 category, color, brand)
        for extra_info in self.eInfo['item']:
            if extra_info == 'popularity':
                continue
            # 从 interaction 里拿到对应的特征 ID
            # 注意：这里需要确保 interaction 能通过索引拿到 item_indices 对应的特征
            # 如果 interaction 是整个 batch 的，你需要先 slice 出来

            # 这里为了通用性，假设 dataset.get_item_feature 存在
            # 或者如果 interaction 包含了当前 batch 的列，直接取
            feature_val = interaction[prefix + extra_info]

            # 如果 feature_val 已经是 batch 后的数据，直接用
            emb = self.extra_embedding_forward(extra_info, feature_val)
            res.append(emb)

        # 拼接: [batch_size, 3 * emb_dim]
        return torch.concat(res, dim=-1)

    def extra_embedding_forward(self, extra_info, item):
        embedding_layer = self.item_extra_embedding[extra_info]
        self.item_embedding_name = []
        emb = embedding_layer(item)
        if item.dim() == 2:
            mask = (item != 0).unsqueeze(-1)  # [batch_size, seq_len, 1]
            emb_masked = emb * mask  # [batch_size, seq_len, embedding_dim]
            valid_count = mask.sum(dim=1)  # [batch_size, 1]
            mean_emb = emb_masked.sum(dim=1) / valid_count.clamp(min=1)  # [batch_size, embedding_dim]
            return mean_emb
        else:
            # 一维时直接返回 embedding
            return emb

    def get_user_embedding(self, user):
        r"""Get a batch of user embedding tensor according to input user's id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        return self.user_embedding(user)

    def get_item_embedding(self, item):
        r"""Get a batch of item embedding tensor according to input item's id.

        Args:
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
        """
        return self.item_embedding(item)

    def forward(self, user, item, custom_item_matrix=None):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        if custom_item_matrix is not None:
            item_e = custom_item_matrix
        return user_e, item_e

    def get_strength_item_embedding(self, interaction, pos_item, prefix:str = ''):
        batch_mlp_input = self.get_batch_mlp_input(interaction, prefix)

        # --------------------------------------------------------
        # Step 2: Online MLP 计算 (热计算 -> 算梯度)
        # --------------------------------------------------------
        # 这部分带有梯度，反向传播会更新 MLP 和 Side Info 的 Embedding 表
        batch_online_emb = self.fusion_side_info(batch_mlp_input)
        input_matrix = self.get_fused_embeddings(self.item_embedding(pos_item), batch_online_emb)
        return input_matrix

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]


        # all_item = torch.cat([pos_item, neg_item], dim=0)
        pos_matrix = self.get_strength_item_embedding(interaction, pos_item)
        neg_matrix = self.get_strength_item_embedding(interaction, neg_item, prefix=self.NEG_PREFIX)    
        # pos_matrix, neg_matrix = torch.split(input_matrix, [pos_item.shape[0], neg_item.shape[0]], dim=0)

        user_e, pos_e = self.forward(user, pos_item, custom_item_matrix=pos_matrix)
        neg_e = neg_matrix if neg_matrix is not None else self.get_item_embedding(neg_item)

        batch_item_embeddings = torch.cat([pos_e, neg_e], dim=0)
        out, rq_loss_total, indices, residual = forward_rq_item_epoch(self.rq_model_item, batch_item_embeddings)
        pos_out, neg_out = torch.split(out, [pos_e.shape[0], neg_e.shape[0]], dim = 0)
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(
            user_e, neg_e
        ).sum(dim=1)

        pos_rq_score, neg_rq_score = torch.mul(user_e, pos_out).sum(dim=1), torch.mul(
            user_e, neg_out
        ).sum(dim=1)
        loss = self.loss(pos_item_score, neg_item_score)
        content_loss = self.loss(pos_rq_score, neg_rq_score)
        loss = loss + self.item_rq_loss_rate * rq_loss_total + self.content_bpr_loss_rate * content_loss
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        input_matrix = self.get_strength_item_embedding(interaction, item)
        user_e, item_e = self.forward(user, item, custom_item_matrix=input_matrix)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        all_item_e = self.get_strength_item_embedding(interaction, interaction[self.ITEM_ID])
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)