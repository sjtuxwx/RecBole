
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType
from recbole.fairness.pop_utils import InfoNCE, split_by_pop
from .pop_utils import split_by_pop_non_del

class Context(object):
    def __init__(self, n_users, n_items):
        self.n_users = n_users
        self.n_items = n_items
        self.itempop = None
        self.user_interact = None
    def to(self, device):
        # 将所有self.属性搬到指定device
        if self.itempop is not None:
            self.itempop = self.itempop.to(device)
        if self.user_interact is not None:
            self.user_interact = self.user_interact.to(device)
        return self
    
    def add_user_interaction(self, user_interact):
        self.user_interact = user_interact
    def add_item_popularity(self, itempop):
        self.itempop = itempop
        self.unpop_item, self.pop_item = split_by_pop_non_del(
            torch.arange(self.n_items), self.itempop, with_zero=True
        )
