# -*- coding: utf-8 -*-
# @Time   : 2020/8/31
# @Author : Changxin Tian
# @Email  : cx.tian@outlook.com

# UPDATE:
# @Time   : 2020/9/16, 2021/12/22
# @Author : Shanlei Mu, Gaowei Zhang
# @Email  : slmu@ruc.edu.cn, 1462034631@qq.com

r"""
LightGCN
################################################

Reference:
    Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." in SIGIR 2020.

Reference code:
    https://github.com/kuandeng/LightGCN
"""
import copy

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.ndimage import label
import torch.nn as nn
from torch.nn.functional import embedding

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.layers import MLPLayers
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
from recbole.fairness.pop_utils import InfoNCE, split_by_pop, InfoNCE_i
from recbole.rq.models.rqvae import RQVAE

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)

class FairLightGCN(GeneralRecommender):
    r"""LightGCN is a GCN-based recommender model.

    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.

    We implement the model following the original author with a pairwise training mode.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(FairLightGCN, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)

        
        # load parameters info
        self.latent_dim = config[
            "embedding_size"
        ]  # int type:the embedding size of lightGCN
        self.n_layers = config["n_layers"]  # int type:the layer num of lightGCN
        self.reg_weight = config[
            "reg_weight"
        ]  # float32 type: the weight decay for l2 normalization
        self.require_pow = config["require_pow"]

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim
        )
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

        if config['eps'] is None:
            self.eps = 0.2
        else:
            self.eps = config['eps']

        if config['gama'] is None:
            self.gama = 0.2
        else:
            self.gama = config['gama']

        if config['beta'] is None:
            self.beta = 0.2
        else:
            self.beta = config['beta']

        if config['cl_rate'] is None:
            self.cl_rate = 0.2
        else:
            self.cl_rate = config['cl_rate']

        if config['item_rq_loss_rate'] is None:
            self.item_rq_loss_rate = 0.2
        else:
            self.item_rq_loss_rate = config['item_rq_loss_rate']
        if config['pop_rate'] is None:
            self.pop_rate = 0.2
        else:
            self.pop_rate = config['pop_rate']

        if config['item_loss_type'] is None:
            self.item_loss_type = 'full'
        else:
            self.item_loss_type = config['item_loss_type']
            
        if config['pop_loss_rate'] is None:
            self.pop_loss_rate = 0.2
        else:
            self.pop_loss_rate = config['pop_loss_rate']

        if config['enable_user_loss'] is None:
            self.enable_user_loss = False
        else:
            self.enable_user_loss = config['enable_user_loss']
        # if "eps" in config.keys():
        #     self.eps = config["eps"]
        # else:
        #     self.eps = 0.2
        self.gen_extra_embedding()
        self.item_strong_dim = self.latent_dim * ((len(self.eInfo['item']))  - 1)
        self.item_strong_info = nn.Parameter()

        
        # 这里需要去掉 popularity 这个 extra_info
        self.rq_model_item = RQVAE(in_dim=self.latent_dim,
                  num_emb_list=[32, 32, 32],
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
        self.fusion_side_info = MLPLayers(
            [self.item_strong_dim, self.latent_dim],
            dropout=0.2, activation="sigmoid", last_activation=False
        )
        self.side_info_ln = nn.LayerNorm(self.latent_dimS)
        self.fusion_side_info = nn.Sequential(
            self.fusion_side_info,
            self.side_info_ln
        )
        
        # ============================================================
        # 2. 定义 Target MLP (师父 - 用于生成稳定缓存)
        # ============================================================
        # 深拷贝 Online MLP，保证结构初始参数一致
        self.fusion_side_info_target = copy.deepcopy(self.fusion_side_info)

        # 核心：完全冻结 Target MLP，不接受梯度，只接受动量更新
        for param in self.fusion_side_info_target.parameters():
            param.requires_grad = False
        
        self.alpha = 0.10

        self.register_buffer(
            'side_info_cache',
            torch.zeros(self.n_items, self.latent_dim)
        )
        # 动量系数 (建议 0.99 或 0.999)
        self.momentum = config['momentum'] if 'momentum' in config else 0.995

        self.fusion_gate_layer = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),  # 输入维度减半
            nn.Tanh(),
            nn.Linear(self.latent_dim, 1),
            nn.Sigmoid()
        )

        self.pop_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim // 2),
            nn.Tanh(),
            nn.Linear(self.latent_dim // 2, 1),
            nn.Sigmoid()
        )

        # self.rq_model_user = RQVAE(in_dim=self.latent_dim * (len(self.eInfo['user']) + 1),
        #           num_emb_list=[8, 8, 8],
        #           e_dim=16,
        #           layers=[64, 32],
        #           dropout_prob=0,
        #           bn=False,
        #           loss_type='mse',
        #           quant_loss_weight=1,
        #           beta=0.25,
        #           kmeans_init=True,
        #           kmeans_iters=100,
        #           sk_epsilons=[0.0, 0.0, 0.0],
        #           sk_iters=50,
        #           )

    @torch.no_grad()
    def _update_target_network(self):
        """
        Momentum update: theta_target = m * theta_target + (1 - m) * theta_online
        """
        for param_o, param_t in zip(self.fusion_side_info.parameters(), self.fusion_side_info_target.parameters()):
            param_t.data = param_t.data * self.momentum + param_o.data * (1.0 - self.momentum)

    def gen_extra_embedding(self):
        self.item_extra_embedding = torch.nn.ModuleDict()
        self.user_extra_embedding = torch.nn.ModuleDict()
        for extra_info in self.eInfo['item']:
            aa = torch.nn.Embedding(
                num_embeddings=self.eInfo['item'][extra_info], embedding_dim=self.latent_dim, device=self.device
            )
            self.item_extra_embedding[extra_info] =  aa
        for extra_info in self.eInfo['user']:
            aa = torch.nn.Embedding(
                num_embeddings=self.eInfo['user'][extra_info], embedding_dim=self.latent_dim, device=self.device
            )
            self.user_extra_embedding[extra_info] =  aa
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
    def extra_embedding_for_specified(self, extra_info, item):
        embedding_layer = self.item_extra_embedding[extra_info]
        if extra_info == 'popularity':
            item = item.long()
        emb = embedding_layer(item)
        return emb

    def extra_embedding_forward_user(self, extra_info, user):
        embedding_layer = self.user_extra_embedding[extra_info]
        self.user_embedding_name = []
        emb = embedding_layer(user)
        if user.dim() == 2:
            mask = (user != 0).unsqueeze(-1)  # [batch_size, seq_len, 1]
            emb_masked = emb * mask  # [batch_size, seq_len, embedding_dim]
            valid_count = mask.sum(dim=1)  # [batch_size, 1]
            mean_emb = emb_masked.sum(dim=1) / valid_count.clamp(min=1)  # [batch_size, embedding_dim]
            return mean_emb
        else:
            # 一维时直接返回 embedding
            return emb
            
    def get_user_cl_loss_fn(self, user_view1, user_view2):
        if self.enable_user_loss in [None, False]:
            return 0
        else:
            return InfoNCE(user_view1, user_view2)
    def get_item_cl_loss_fn(self, item_view1, item_view2, item_view1_pop, item_view2_pop, item_view1_unpop, item_view2_unpop):
        if self.item_loss_type is None:
            return 0
        if self.item_loss_type == 'full':
            return InfoNCE(item_view1, item_view2)
        if self.item_loss_type == 'loss1':
            return InfoNCE_i(item_view1_unpop, item_view2_unpop, item_view1_pop, gama=self.beta)
        if self.item_loss_type == 'loss2':
            return InfoNCE_i(item_view1_pop, item_view2_pop, item_view1_unpop, gama=self.beta)
        if self.item_loss_type == 'loss12':
            return (self.gama * (InfoNCE_i(item_view1_unpop, item_view2_unpop, item_view1_pop, gama=self.beta)) +
                    (1 - self.gama) * InfoNCE_i(item_view1_pop, item_view2_pop, item_view1_unpop, gama=self.beta))

    def forward_rq_item_epoch(self, rq_model, data):
        out, rq_loss, indices, residual = rq_model(data)
        rq_loss_total, rq_rec = rq_model.compute_loss(out, rq_loss, xs=data)

        return out, rq_loss_total, indices, residual
    def forward_rq_user_epoch(self, rq_model, data):
        out, rq_loss, indices = rq_model(data)
        rq_loss_total, rq_rec = rq_model.compute_loss(out, rq_loss, xs=data)

        return out, rq_loss_total, indices

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

    def process_user_side_info(self, interaction):
        res = []
        for extra_info in self.eInfo['user']:
            user = interaction[extra_info]
            emb = self.extra_embedding_forward_user(extra_info, user)
            res.append(emb)
        res.append(self.user_embedding(interaction[self.USER_ID]))
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


    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        inter_M = self.interaction_matrix
        row = np.concatenate([inter_M.row, inter_M.col + self.n_users])
        col = np.concatenate([inter_M.col + self.n_users, inter_M.row])
        data = np.ones(row.shape[0], dtype=np.float32)
        A = sp.coo_matrix((data, (row, col)), shape=(self.n_users + self.n_items, self.n_users + self.n_items))
        A.sum_duplicates()
        deg = np.array(A.tocsr().sum(axis=1)).flatten() + 1e-7
        D = sp.diags(np.power(deg, -0.5))
        L = D @ A @ D
        L = sp.coo_matrix(L)
        indices = torch.LongTensor(np.vstack([L.row, L.col]))
        values = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(indices, values, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self, use_strong_info: bool = True):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self, custom_item_matrix=None, perturbed=False):
        # 1. User 还是原来的 User ID Embedding
        user_all_embeddings = self.user_embedding.weight
        if self.alpha == 0:
            item_all_embeddings = self.item_embedding.weight
        else:

            # 2. Item 矩阵逻辑
            if custom_item_matrix is not None:
                # 训练时：使用我们拼装好的“弗兰肯斯坦”矩阵
                item_all_embeddings = custom_item_matrix
            else:
                # 推理/验证时：使用 ID Emb + 缓存的 Side Emb
                # 或者是 ID Emb + Target MLP 算出来的 Side Emb
                # item_all_embeddings = torch.cat([
                #     self.item_embedding.weight,
                #     self.side_info_cache
                # ], dim=1)  # 假设是拼接

                # 如果你是相加融合：
                item_all_embeddings = self.get_fused_embeddings(self.item_embedding.weight, self.side_info_cache)
                # item_all_embeddings = (1-self.alpha)*self.item_embedding.weight + self.alpha*self.side_info_cache

        # 3. 构造图卷积的初始 Ego Embedding
        # 注意维度：User 也是 latent_dim，但 Item 现在可能是 2*latent_dim (如果concat)
        # 如果维度不匹配，你可能需要一个线性层把 Item 降维，或者 User 也做对应拼接
        # 这里假设维度已经对齐，或者 Item ID Emb 和 Side Emb 是相加关系
        all_embeddings = torch.cat([user_all_embeddings, item_all_embeddings], dim=0)

        embeddings_list = []

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            if perturbed:
                random_noise = torch.rand_like(all_embeddings).to(self.device)
                all_embeddings = all_embeddings + torch.sign(all_embeddings) * F.normalize(random_noise,
                                                                                           dim=1) * self.eps
            embeddings_list.append(all_embeddings)

        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )

        return user_all_embeddings, item_all_embeddings

    # def forward(self, perturbed=False):
    #     all_embeddings = self.get_ego_embeddings()
    #     embeddings_list = []
    #
    #     for layer_idx in range(self.n_layers):
    #         all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
    #         if perturbed:
    #             random_noise = torch.rand_like(all_embeddings).to(self.device)
    #             all_embeddings = all_embeddings + torch.sign(all_embeddings) * F.normalize(random_noise,
    #                                                                                        dim=1) * self.eps
    #         embeddings_list.append(all_embeddings)
    #
    #     lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
    #     lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
    #
    #     user_all_embeddings, item_all_embeddings = torch.split(
    #         lightgcn_all_embeddings, [self.n_users, self.n_items]
    #     )
    #
    #     return user_all_embeddings, item_all_embeddings

    def fusion_cl_loss(self, user_loss, item_loss):
        # print(user_loss, item_loss)
        fusion_loss = user_loss
        if self.enable_user_loss == False:
            fusion_loss = item_loss
        else:
            fusion_loss = (item_loss + user_loss) / 2
        return self.cl_rate * fusion_loss

    def cl_loss(self, user, item, itempop, cl_rate=0.2, gama=0.2, beta=0.2):
        if self.cl_rate == 0.0:
            return 0.0
        G1, G2 = split_by_pop(item, itempop)
        user_view1, item_view1 = self.forward(perturbed=True)
        # item_view1 = item_view1[item]
        
        user_view2, item_view2 = self.forward(perturbed=True)
        # item_view2 = item_view2[item]   
        # user_loss = InfoNCE(user_view1[user], user_view2[user])
        user_loss = self.get_user_cl_loss_fn(user_view1[user], user_view2[user])
        item_loss = self.get_item_cl_loss_fn(item_view1[item], item_view2[item], item_view1[G2], item_view2[G2], item_view1[G1], item_view2[G1])


        return self.fusion_cl_loss(user_loss, item_loss)
    def calculate_loss(self, interaction):
        # clear the storage variable when training

        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        # --------------------------------------------------------
        # Step 1: 准备当前 Batch Item 的 MLP 输入
        # --------------------------------------------------------
        # 获取 pos_item 对应的原始 Side Info Embedding 拼接结果
        # input shape: [batch_size, 3 * dim]
        batch_mlp_input = self.get_batch_mlp_input(interaction, pos_item)

        # --------------------------------------------------------
        # Step 2: Online MLP 计算 (热计算 -> 算梯度)
        # --------------------------------------------------------
        # 这部分带有梯度，反向传播会更新 MLP 和 Side Info 的 Embedding 表
        batch_online_emb = self.fusion_side_info(batch_mlp_input)

        # --------------------------------------------------------
        # Step 3: Target MLP 计算 & 刷新缓存 (冷计算 -> 存背景)
        # --------------------------------------------------------
        with torch.no_grad():  # 绝对不要梯度
            # a. 动量更新参数
            self._update_target_network()

            # b. 用 Target 网络算一遍
            batch_target_emb = self.fusion_side_info_target(batch_mlp_input)

            # c. 写入全局缓存
            # 注意：这里我们只更新 pos_item 对应的行
            # .detach() 双重保险，确保不带计算图
            self.side_info_cache[pos_item] = batch_target_emb.detach()

        # --------------------------------------------------------
        # Step 4: 组装 "弗兰肯斯坦" 矩阵
        # --------------------------------------------------------
        # a. 复制一份缓存 (全是 Target MLP 的结果)
        global_side_emb = self.side_info_cache.clone()

        # b. 把当前 Batch 的位置挖掉，填入 Online MLP 的结果 (带梯度!)
        # 这是梯度回传的唯一通道
        global_side_emb.index_put_((pos_item,), batch_online_emb)

        # c. 与 ID Embedding 结合 (Concat 或 Add)
        # 假设我们使用 Concat 方式融合
        # global_input_matrix = torch.cat([
        #     self.item_embedding.weight,  # ID Embedding (始终可训练)
        #     global_side_emb  # 混合 Side Embedding
        # ], dim=1)
        global_input_matrix = self.get_fused_embeddings(self.item_embedding.weight, global_side_emb)
        user_all_embeddings, item_all_embeddings = self.forward(custom_item_matrix=global_input_matrix)
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        batch_item_embeddings = torch.cat([pos_embeddings, neg_embeddings], dim=0)
        out, rq_loss, indices, residual = self.forward_rq_item_epoch(self.rq_model_item, batch_item_embeddings)
        codebook = self.rq_model_item.rq.get_codebook()
        pop_book = codebook[0][indices[0]]
        unpop_book = codebook[1][indices[1]] + codebook[2][indices[2]]




        context = interaction.context
        context.itempop = context.itempop.to(self.device)
        

        cl_loss = self.cl_loss(user, pos_item, context.itempop, cl_rate=self.cl_rate, gama=self.gama, beta=self.beta)



        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate regularization Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            # global_side_emb[pos_item],
            require_pow=self.require_pow,
        )

        loss = mf_loss + self.reg_weight * reg_loss + cl_loss

        # item_side_info = self.process_item_side_info(interaction, excluded_info=['popularity'])
        # user_side_info = self.process_user_side_info(interaction)
        # out, rq_loss, indices, pop_out = self.forward_rq_item_epoch(self.rq_model_item, item_side_info)
        # # out, rq_loss_user, indices = self.forward_rq_user_epoch(self.rq_model_user, user_side_info)
        # # return loss + 0.5 * (rq_loss + rq_loss_user) / 2
        # item_popularity = interaction['popularity'][:, 1]
        # item_embedding = self.extra_embedding_for_specified('popularity', item_popularity)
        # pop_loss = self.pop_recontruct_loss(pop_out, item_embedding)
        # align_loss = self.pop_item_align_loss(pop_out, item_all_embeddings[pos_item])
        # return loss + self.item_rq_loss_rate * rq_loss + self.pop_loss_rate * (pop_loss + align_loss) / 2
        return loss
    
    def pop_item_align_loss(self, pop_out, item_embedding):
        # 计算 pop_out 与 item_embedding 的余弦相似度损失
        cos_sim = F.cosine_similarity(pop_out, item_embedding, dim=-1)  # [batch_size]
        # 正交约束
        loss = cos_sim ** 2
        return loss.mean()
     
    def pop_recontruct_loss(self, pop_out, pop_embedding):
        # 计算 pop_out 与 pop_embedding 的余弦相似度损失
        cos_sim = F.cosine_similarity(pop_out, pop_embedding, dim=-1)  # [batch_size]
        # 最大化余弦相似度 -> 最小化 1 - cos_sim
        loss = 1 - cos_sim
        return loss.mean()

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)

    def get_batch_mlp_input(self, interaction, item_indices):
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
            feature_val = interaction[extra_info]

            # 如果 feature_val 已经是 batch 后的数据，直接用
            emb = self.extra_embedding_forward(extra_info, feature_val)
            res.append(emb)

        # 拼接: [batch_size, 3 * emb_dim]
        return torch.concat(res, dim=-1)

    def