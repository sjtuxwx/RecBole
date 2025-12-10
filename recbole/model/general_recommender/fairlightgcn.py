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

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.ndimage import label
import torch.nn as nn

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
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
        self.e_dim = 16
        # 这里需要去掉 popularity 这个 extra_info
        self.rq_model_item = RQVAE(in_dim=self.latent_dim * (len(self.eInfo['item']) + 1 - 1),
                  num_emb_list=[512, 16, 16],
                  e_dim=self.e_dim,
                  layers=[32],
                  dropout_prob=0,
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
        
        # Pop Predictor (Positive Alignment)
        self.pop_predictor = nn.Sequential(
            nn.Linear(self.e_dim, self.e_dim),
            nn.ReLU(),
            nn.Linear(self.e_dim, 1)
        )

        # Pop Discriminator (Negative Alignment / Adversarial)
        self.content_decoder = torch.nn.Sequential(
            torch.nn.Linear(self.e_dim, 2*self.e_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2*self.e_dim, self.e_dim),
            torch.nn.ReLU()
        )
        self.content_pop_decoder = torch.nn.Sequential(
            self.content_decoder,
            torch.nn.Linear(self.e_dim, 1)
        )
        self.content_info_decoder = torch.nn.Sequential(
            self.content_decoder,
            torch.nn.Linear(self.e_dim, self.latent_dim)
        )


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
        out, rq_loss, indices, z_pop, z_content = rq_model(data)
        rq_loss_total, rq_rec = rq_model.compute_loss(out, rq_loss, xs=data)

        return out, rq_loss_total, indices, z_pop, z_content
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

    def get_ego_embeddings(self, unique_item_id, item_strong_info):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        """
        user_embeddings = self.user_embedding.weight
        
        # [修改点] 添加 .clone()
        # 原因：self.item_embedding.weight 是叶子节点，不能直接在上面做 in-place 修改。
        # clone() 产生一个中间变量，修改中间变量是合法的，梯度可以正常回传。
        item_embeddings = self.item_embedding.weight.clone()
        
        if unique_item_id is not None and item_strong_info is not None:
            # 现在的 item_embeddings 是一个 clone 的副本，在这里修改它是安全的
            # 注意：这里计算均值时，右边的 item_embeddings[unique_item_id] 依然使用的是修改前的值（正确逻辑）
            item_embeddings[unique_item_id] = (item_strong_info + item_embeddings[unique_item_id]) / 2
            
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings
    def forward(self, unique_item_id=None, item_strong_info=None, perturbed=False):
        all_embeddings = self.get_ego_embeddings(unique_item_id, item_strong_info)
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

    def fusion_cl_loss(self, user_loss, item_loss):
        # print(user_loss, item_loss)
        fusion_loss = user_loss
        if self.enable_user_loss == False:
            fusion_loss = item_loss
        else:
            fusion_loss = (item_loss + user_loss) / 2
        return self.cl_rate * fusion_loss

    def cl_loss(self, user, item, itempop, cl_rate=0.2, gama=0.2, beta=0.2):
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

        # RQ-VAE & Hierarchical Disentanglement
        item_side_info = self.process_item_side_info(interaction, excluded_info=['popularity'])
        unique_item_id, unique_idx = self.unique_with_order(interaction[self.ITEM_ID])
        item_side_info = item_side_info[unique_idx]
        out, rq_loss, indices, z_pop, z_content = self.forward_rq_item_epoch(self.rq_model_item, item_side_info)

        # Ground Truth Popularity
        # User confirmed interaction['popularity'] is already normalized [0, 1]
        gtd_pop = interaction['popularity'].float().unsqueeze(-1)  # [B, 1]
        gtd_pop = gtd_pop[unique_idx]

        # Positive Alignment: z_pop should predict popularity
        pred_pos = self.pop_predictor(z_pop)
        # pos_align_loss = F.binary_cross_entropy_with_logits(pred_pos, gtd_pop)
        pos_align_loss = self.pop_item_align_loss(torch.sigmoid(pred_pos), gtd_pop)

        # Negative Alignment (Adversarial): z_content should NOT predict popularity
        # Apply Gradient Reversal Layer to invert gradient
        z_content_grl = grad_reverse(z_content, alpha=1.0)
        pred_neg = self.content_pop_decoder(z_content_grl)
        # neg_align_loss = F.binary_cross_entropy_with_logits(pred_neg, gtd_pop)
        neg_align_loss = self.pop_item_align_loss(torch.sigmoid(pred_neg), gtd_pop)

        item_strong_info = self.content_info_decoder(z_content)

        user_all_embeddings, item_all_embeddings = self.forward(unique_item_id, item_strong_info)
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]


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
            require_pow=self.require_pow,
        )

        loss = mf_loss + self.reg_weight * reg_loss + cl_loss


        # neg_align_loss = 0
        return loss + self.item_rq_loss_rate * rq_loss + self.pop_loss_rate * (pos_align_loss + neg_align_loss)
    
    def pop_item_align_loss(self, pop_out, gtd_pop):
        # 计算 pop_out 与 item_embedding 的余弦相似度损失
        return (((pop_out - gtd_pop) ** 2) / 2).mean()
     
    def pop_recontruct_loss(self, pop_out, gtd_pop):
        # 计算 pop_out 与 pop_embedding 的余弦相似度损失
        return (((pop_out - gtd_pop) ** 2) / 2).mean()

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def unique_with_order(self, x):
        """
        输入: x (1D tensor)
        输出: 
            unique_values: 保持原顺序的唯一值
            first_indices: 这些值第一次出现的索引
        """
        # 1. 获取唯一值(自动排序)和逆向索引
        # sorted=True 是默认的，返回的 sorted_uniques 是按数值大小排序的 (如 [1, 2, 3])
        # inverse_indices 是原始 x 中的元素对应 sorted_uniques 的下标
        sorted_uniques, inverse_indices = torch.unique(x, sorted=True, return_inverse=True)
        
        # 2. 准备一个容器来存储每个唯一值第一次出现的索引
        # 初始化为由 x 的长度填充的一个大数（为了后面取最小值做准备）
        # 长度等于唯一值的数量
        first_indices_placeholder = torch.zeros(
            sorted_uniques.size(0), 
            dtype=torch.long, 
            device=x.device
        ).fill_(x.size(0))
        
        # 3. 生成原始位置的索引序列 [0, 1, 2, ...]
        perm = torch.arange(x.size(0), dtype=torch.long, device=x.device)
        
        # 4. 使用 scatter_reduce 找到每个唯一值对应的最小索引
        # 逻辑解释：
        # inverse_indices 告诉我们 x 的第 i 个元素属于哪个“唯一值类别”。
        # 我们把 perm (即原始索引) 按照 inverse_indices 撒(scatter)到 placeholder 里。
        # reduce="amin" 表示：如果同一个位置被撒入多个索引，保留最小的那个（即第一次出现的那个）。
        # 注意：scatter_reduce_ 需要 PyTorch 1.12+ 版本
        first_indices_placeholder.scatter_reduce_(
            0, 
            inverse_indices, 
            perm, 
            reduce="amin", 
            include_self=False
        )
        
        # 此时，first_indices_placeholder 里存的是 sorted_uniques 对应的第一次出现索引。
        # 但因为 sorted_uniques 是按值排序的，所以这里的索引顺序是乱的（不是按出现顺序）。
        
        # 5. 对索引进行排序，恢复“按出现顺序”排列
        # sort_indices 是排序后的索引，indices_order 是原本的排序位置
        sorted_first_indices, sort_order = first_indices_placeholder.sort()
        
        # 6. 使用排序后的索引从原始 x 中提取值，或者重新排列 sorted_uniques
        # 推荐直接用 sorted_first_indices 从 x 取值，这样最直观
        final_unique_values = x[sorted_first_indices]
        
        return final_unique_values, sorted_first_indices

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)
