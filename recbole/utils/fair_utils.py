import torch

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

def args2class(self, config):
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
    if config['content_bpr_loss_rate'] is None:
        self.content_bpr_loss_rate = 0.0
    else:
        self.content_bpr_loss_rate = config['content_bpr_loss_rate']
    if config['enable_side_fuse'] is None:
        self.enable_side_fuse = False
    else:
        self.enable_side_fuse = config['enable_side_fuse']
    if config['task'] is None:
        self.task = 'xwx'
    else:
        self.task = config['task']


def gen_extra_embedding(eInfo, latent_dim, device):
    item_extra_embedding = torch.nn.ModuleDict()
    user_extra_embedding = torch.nn.ModuleDict()
    for extra_info in eInfo['item']:
        aa = torch.nn.Embedding(
            num_embeddings=eInfo['item'][extra_info], embedding_dim=latent_dim, device=device
        )
        item_extra_embedding[extra_info] =  aa
    for extra_info in eInfo['user']:
        aa = torch.nn.Embedding(
            num_embeddings=eInfo['user'][extra_info], embedding_dim=latent_dim, device=device
        )
        user_extra_embedding[extra_info] =  aa
    return item_extra_embedding, user_extra_embedding

def forward_rq_item_epoch(rq_model, data):
    out, rq_loss, indices, residual = rq_model(data)
    rq_loss_total, rq_rec = rq_model.compute_loss(out, rq_loss, xs=data)
    return out, rq_loss_total, indices, residual

def forward_mutil_decoder_rq_item_epoch(rq_model, data):
    out, rq_loss, indices, residual = rq_model(data)

    rq_loss_total, rq_rec = rq_model.compute_multi_loss(out, rq_loss, xs=data)

    return out, rq_loss_total, indices, residual

