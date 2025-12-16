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
from recbole.model.loss import BPRLoss
from recbole.utils import InputType
from recbole.utils.fair_utils import args2class, forward_rq_item_epoch
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
            in_dim=self.latent_dim,
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
            pop_dim=self.latent_dim
        )

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

    def forward(self, user, item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        return user_e, item_e

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.get_item_embedding(neg_item)

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
        user_e, item_e = self.forward(user, item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)
