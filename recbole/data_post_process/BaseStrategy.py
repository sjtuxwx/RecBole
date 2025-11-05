import pandas as pd
import numpy as np
from functools import wraps

def basic_process_users(func):
    """装饰器：对用户数据进行基础处理"""
    @wraps(func)
    def wrapper(self, users, items, interactions):
        
        
        # 然后执行基础处理逻辑
        if self.config['attr']['user']['need_inter_num']:
            # 计算每个用户的交互数量，并重置索引以便合并
            interaction_counts = interactions.groupby(self.config['USER_ID_FIELD']).size().reset_index(name='interaction_nums')
            # 使用外连接将交互数量合并到 users 表中
            users = users.merge(interaction_counts, on=self.config['USER_ID_FIELD'], how='left')
        if self.config['attr']['user']['need_inter_hour']:
            # 从 timestamp 中提取小时（24小时制）
            interactions['hour'] = pd.to_datetime(interactions['timestamp'], unit=self.config['attr']['interaction']['timestamp_unit']).dt.hour
            # 按用户分组，统计每个小时的出现次数，取出现次数最多的小时
            user_top_hour = (interactions
                             .groupby([self.config['USER_ID_FIELD'], 'hour'])
                             .size()
                             .reset_index(name='count')
                             .sort_values([self.config['USER_ID_FIELD'], 'count'], ascending=[True, False])
                             .drop_duplicates(subset=[self.config['USER_ID_FIELD']], keep='first')
                             .rename(columns={'hour': 'interaction_hours'})
                             [[self.config['USER_ID_FIELD'], 'interaction_hours']]) // 4
            # 合并到 users 表中
            users = users.merge(user_top_hour, on=self.config['USER_ID_FIELD'], how='left')
        if self.config['attr']['user']['need_rating']:
            # 计算每个用户的交互数量，并重置索引以便合并
            interaction_counts = interactions.groupby(self.config['USER_ID_FIELD'])['rating'].mean().reset_index(name='mean_rating')
            # 使用外连接将交互数量合并到 users 表中
            users = users.merge(interaction_counts, on=self.config['USER_ID_FIELD'], how='left')
            
        # 先执行被装饰的函数
        users, items, interactions = func(self, users, items, interactions)
        return users, items, interactions
    return wrapper

def basic_process_items(func):
    """装饰器：对物品数据进行基础处理"""
    @wraps(func)
    def wrapper(self, users, items, interactions):
       
        
        # 然后执行基础处理逻辑
        if self.config['attr']['item']['need_inter_num']:
            # 计算每个物品的交互数量，并重置索引以便合并
            interaction_counts = interactions.groupby(self.config['ITEM_ID_FIELD']).size().reset_index(name='interaction_nums')
            # 使用外连接将交互数量合并到 items 表中
            items = items.merge(interaction_counts, on=self.config['ITEM_ID_FIELD'], how='left')
        if self.config['attr']['item']['need_rating']:
            # 计算每个物品的交互数量，并重置索引以便合并
            interaction_counts = interactions.groupby(self.config['ITEM_ID_FIELD'])['rating'].mean().reset_index(name='mean_rating')
            # 使用外连接将交互数量合并到 items 表中
            items = items.merge(interaction_counts, on=self.config['ITEM_ID_FIELD'], how='left')
        if self.config['attr']['item']['need_inter_hour']:
            # 从 timestamp 中提取小时（24小时制）
            interactions['hour'] = pd.to_datetime(interactions['timestamp'], unit=self.config['attr']['interaction']['timestamp_unit']).dt.hour
            # 按物品分组，统计每个小时的出现次数，取出现次数最多的小时
            item_top_hour = (interactions
                             .groupby([self.config['ITEM_ID_FIELD'], 'hour'])
                             .size()
                             .reset_index(name='count')
                             .sort_values([self.config['ITEM_ID_FIELD'], 'count'], ascending=[True, False])
                             .drop_duplicates(subset=[self.config['ITEM_ID_FIELD']], keep='first')
                             .rename(columns={'hour': 'interaction_hours'})
                             [[self.config['ITEM_ID_FIELD'], 'interaction_hours']]) // 4
            # 合并到 items 表中
            items = items.merge(item_top_hour, on=self.config['ITEM_ID_FIELD'], how='left')
            
         # 先执行被装饰的函数
        users, items, interactions = func(self, users, items, interactions)
        return users, items, interactions
    return wrapper

class BaseStrategy(object):
    def __init__(self, config):
        self.config = config
    
    @basic_process_users
    def special_process_users(self, users, items, interactions):
        pass
    
    @basic_process_items  
    def special_process_items(self, users, items, interactions):
        pass
            
