import numpy as np
import torch
from mpmath import residual
from numpy.core.numeric import indices
from torch import nn
from torch.cuda.nccl import init_rank
from torch.nn import functional as F
from torch.nn.functional import l1_loss

from .layers import MLPLayers
from .rq import ResidualVectorQuantizer
from .rqvae import RQVAE

class Shared_Encoder_RQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 # num_emb_list=[256,256,256,256],
                 num_emb_list=None,
                 e_dim=64,
                 # layers=[512,256,128],
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 beta=0.25,
                 kmeans_init=False,
                 kmeans_iters=100,
                 # sk_epsilons=[0,0,0.003,0.01]],
                 sk_epsilons=None,
                 sk_iters=100,
                 pop_dim=64,
                 decoder_num: int = 1
        ):
        super(Shared_Encoder_RQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim

        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight=quant_loss_weight
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters

        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn,
                                 activation='sigmoid'
                                 )

        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim,
                                          beta=self.beta,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,)
        self.rq = nn.ModuleList([ResidualVectorQuantizer(num_emb_list, e_dim,
                                          beta=self.beta,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,) for _  in range(decoder_num)])


        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = nn.ModuleList([MLPLayers(layers=self.decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn,
                                       activation='sigmoid'
                                       ) for _ in range(decoder_num)])

        self.pop_decode_layer_dims = self.encode_layer_dims[::-1] + [pop_dim]
        self.pop_decoder = MLPLayers(layers=self.pop_decode_layer_dims,
                                 dropout=self.dropout_prob, bn=self.bn,
                                 activation='sigmoid'
                                 )
        self.W_gate = nn.Linear(self.in_dim, self.in_dim, bias=True)
    def forward(self, x, use_sk=True):
        ipt = x
        x = self.encoder(x) # 这个是最初始的输入
        x_q_, rq_loss_, indices_, residual_, out_ = [], [], [], [], []
        for i in range(self.decoder_num):
            x_q, rq_loss, indices, residual = self.rq(x,use_sk=use_sk)
            out = self.decoder(x_q)
            x_q_.append(out)
            rq_loss_.append(rq_loss)
            indices_.append(indices)
            residual_.append(residual)
            out_.append(out)
        # pop_out = self.pop_decoder(residual)
        

        return out_, rq_loss_, indices_, residual_

    @torch.no_grad()
    def get_indices(self, xs, idx, use_sk=False):
        x_e = self.encoder(xs)
        _, _, indices = self.rq[idx](x_e, use_sk=use_sk)
        return indices

    def compute_loss(self, out, quant_loss, xs=None):

        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')
        # gate = torch.sigmoid(self.W_gate(xs))
        # if self.loss_type == 'mse':
        #     loss_recon = gate * ((out - xs) ** 2) * (1/2)
        #     loss_recon = loss_recon.mean()
        # if self.loss_type == 'l1':
        #     loss_recon = gate * (out - xs).abs()
        #     loss_recon = loss_recon.mean()
        loss_total = loss_recon + self.quant_loss_weight * quant_loss

        return loss_total, loss_recon
    def compute_multi_loss(self, out, quant_loss, xs=None):
        loss_total = 0
        loss_recon = 0
        for ot, qls in zip(out, quant_loss):
            ll, lc = self.compute_loss(ot, qls, xs)
            loss_total += ll
            loss_recon += lc
        return loss_total, loss_recon