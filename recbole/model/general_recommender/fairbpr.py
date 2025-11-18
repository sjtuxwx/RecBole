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
import torch.nn.functional as F
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType
from recbole.fairness.pop_utils import InfoNCE, split_by_pop

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
        # self.pop_embedding = nn.Embedding(max_pop, self.embedding_size)
        self.item_pop_factor = nn.Embedding(self.n_items, self.embedding_size)
        self.loss = BPRLoss()
        self.user_mlp = nn.Linear(self.embedding_size, self.embedding_size)
        self.item_mlp = nn.Linear(self.embedding_size, self.embedding_size)
        self.pop_mlp = nn.Linear(self.embedding_size, self.embedding_size)

        # parameters initialization
        self.apply(xavier_normal_initialization)

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
    
    def get_pop_embedding(self, pop):
        return self.popularity_positional_encoding_transformer_style(pop)

    def forward(self, user, item, pp):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        p_e = self.get_pop_embedding(pp)
        user_e = self.user_mlp(user_e)
        item_e = self.item_mlp(item_e)
        p_e = self.pop_mlp(p_e)
        return user_e, item_e, p_e
    
    def cl_loss(self, user, item, itempop):
        G1, G2 = split_by_pop(item, itempop)
        _, item_view1 = self.forward(user, item, perturbed=True)
        _, item_view2 = self.forward(user, item, perturbed=True)
        clss = InfoNCE(item_view1, item_view2)
        
        
        return clss
    
    def forward(self, user, item, perturbed=False):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        if perturbed:
            random_noise = torch.rand_like(item_e).to(self.device)
            item_e = item_e + torch.sign(item_e) * F.normalize(random_noise, dim=1) * 0.2
            
            random_noise = torch.rand_like(user_e).to(self.device)
            user_e = user_e + torch.sign(user_e) * F.normalize(random_noise, dim=1) * 0.2
            
        # user_e = self.user_mlp(user_e)
        # item_e = self.item_mlp(item_e)
        
        return user_e, item_e,


    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        
        context = interaction.context
        context.itempop = context.itempop.to(self.device)
        # context.user2item = context.user2item.to(self.device)
        
        clss = self.cl_loss(user, pos_item, context.itempop)

        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.get_item_embedding(neg_item)
        # pop_e = self.popularity_positional_encoding_transformer_style(interaction.itempop, self.embedding_size)
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(
            user_e, neg_e
        ).sum(dim=1)
        loss = self.loss(pos_item_score, neg_item_score)
        return loss + 0.15 * clss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e, item_e = self.forward(user, item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        # all_item_e = self.item_mlp(self.item_embedding.weight)
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)

    def popularity_positional_encoding(self, popularity, embedding_size, min_freq=1.0, max_freq=10000.0):
        """
        将标量流行度映射为 embedding_size 维的正弦/余弦位置编码向量。

        参数:
            popularity (torch.Tensor): 形状为 [N] 或 [N, 1] 的流行度标量张量，必须为非负。
            embedding_size (int): 目标嵌入维度，>=1。
            min_freq (float): 最小频率，>0。
            max_freq (float): 最大频率，>min_freq。

        返回:
            torch.Tensor: 形状为 [N, embedding_size] 的编码向量。
        """

        if popularity.dim() == 2 and popularity.size(-1) == 1:
            popularity = popularity.squeeze(-1)
        popularity = popularity.to(dtype=torch.float32)

        N = popularity.shape[0]
        d_model = int(embedding_size)
        device = popularity.device

        if d_model <= 0:
            raise ValueError("embedding_size must be positive")
        if min_freq <= 0 or max_freq <= min_freq:
            raise ValueError("frequency range must satisfy: 0 < min_freq < max_freq")

        half = d_model // 2
        if half == 0:
            scales = torch.tensor([min_freq], dtype=torch.float32, device=device)
        else:
            scales = torch.logspace(
                start=torch.log10(torch.tensor(min_freq)),
                end=torch.log10(torch.tensor(max_freq)),
                steps=half,
                base=10.0,
                device=device,
                dtype=torch.float32,
            )

        phase = popularity.view(N, 1) * scales.view(1, -1)
        sin_part = torch.sin(phase)
        cos_part = torch.cos(phase)
        enc = torch.cat([sin_part, cos_part], dim=-1)

        if enc.shape[1] < d_model:
            extra = torch.sin(popularity.view(N, 1))
            enc = torch.cat([enc, extra], dim=-1)
        elif enc.shape[1] > d_model:
            enc = enc[:, :d_model]

        return enc
    
    def popularity_positional_encoding_transformer_style(self, popularity, embedding_size, scale_factor=10000.0):
        """
        使用Transformer原始位置编码风格实现流行度编码
        
        参数:
            popularity (torch.Tensor): 形状为[N]或[N,1]的流行度标量张量
            embedding_size (int): 目标嵌入维度
            scale_factor (float): 缩放因子，控制频率衰减速度
        
        返回:
            torch.Tensor: 形状为[N, embedding_size]的编码向量
        """
        
        # 输入验证和预处理
        if popularity.dim() == 2 and popularity.size(-1) == 1:
            popularity = popularity.squeeze(-1)
        popularity = popularity.to(dtype=torch.float32)
        
        N = popularity.shape[0]
        d_model = int(embedding_size)
        device = popularity.device
        
        if d_model <= 0:
            raise ValueError("embedding_size must be positive")
        
        # 创建位置编码矩阵
        pe = torch.zeros(N, d_model, device=device)
        
        # 生成维度索引
        position = popularity.unsqueeze(1)  # [N, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32, device=device) * 
            -torch.log(torch.tensor(scale_factor)) / d_model
        )  # [d_model/2]
        
        # 应用正弦余弦编码（Transformer原始风格）
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度：正弦
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度：余弦
        
        return pe

    def align_loss(self, train_dataset, batch_size):
        """
        计算对齐损失，用于FairBPR模型
        
        参数:
            train_dataset (Dataset): 训练数据集
            batch_size (int): 批次大小
        
        返回:
            torch.Tensor: 对齐损失标量
        """
        pass