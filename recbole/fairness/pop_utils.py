import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType

def split_by_pop(items, itempop):
        G1, G2 = [], []
        # user_interact 与 itempop 均为 torch.Tensor
        
        # 根据 itempop 值对物品排序
        pop_vals = itempop[items]  # torch.Tensor
        sorted_idx = pop_vals.argsort()
        items_sorted = items[sorted_idx]  # 去掉最冷门的1个
        # 若为奇数，随机去掉1个
        if items_sorted.shape[0] % 2 != 0:
            del_idx = torch.randint(items_sorted.shape[0], (1,)).item()
            items_sorted = torch.cat(
                [items_sorted[:del_idx], items_sorted[del_idx + 1:]]
            )
        half = items_sorted.shape[0] // 2
        G1.append(items_sorted[:half])
        G2.append(items_sorted[half:])
       
        return G1[0], G2[0]
    
def InfoNCE(view1, view2, temperature=0.2):
        view1, view2 = torch.nn.functional.normalize(
            view1, dim=1), torch.nn.functional.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

def InfoNCE_i(view1, view2, view3,temperature=0.2,gama=0.2):
    view1, view2,view3 = torch.nn.functional.normalize(
        view1, dim=1), torch.nn.functional.normalize(view2, dim=1), torch.nn.functional.normalize(view3, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score_1 = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score_1 = torch.exp(ttl_score_1 / temperature).sum(dim=1)
    ttl_score_2 = torch.matmul(view1, view3.transpose(0, 1))
    ttl_score_2 = torch.exp(ttl_score_2 / temperature).sum(dim=1)

    cl_loss = -torch.log(pos_score / (gama*ttl_score_2+ttl_score_1+pos_score))
    return torch.mean(cl_loss)
