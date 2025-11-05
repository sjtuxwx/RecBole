import pandas as pd
import numpy as np
import logging
import sys
import torch.distributed as dist
from collections.abc import MutableMapping
from logging import getLogger
from collections import OrderedDict
from ray import tune

from recbole.config import Config
from recbole.data import (
    create_dataset,
    data_preparation,
)
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)
import os
from pathlib import Path
from recbole.data_post_process.BaseStrategy import BaseStrategy
from recbole.data_post_process.special_strategy.ml_1m import Ml_1MStrategy
def run_post(
    model=None,
    dataset=None,
    config_file_list=None,
    config_dict=None,
    saved=True,
    queue=None,
):
    r"""A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
        queue (torch.multiprocessing.Queue, optional): The queue used to pass the result to the main process. Defaults to ``None``.
    """
    # configurations initialization
    config = Config(
        model=model,
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    users, items, iteractions= parse_basic_dataset(config)
    strategy = Ml_1MStrategy(config)
    users, items, interactions = strategy.special_process_users(users, items, iteractions)
    users, items, interactions = strategy.special_process_items(users, items, interactions)
    
    print("happy")
    
def parse_basic_dataset(config):
    base_path = os.path.join(config["data_path"])
    user_file_name = config["user_file"] if config["user_file"] else f"{config['dataset']}.user"
    user_file_name = os.path.join(base_path, user_file_name)
    item_file_name = config["item_file"] if config["item_file"] else f"{config['dataset']}.item"
    item_file_name = os.path.join(base_path, item_file_name)
    interaction_file_name = config["interaction_file"] if config["interaction_file"] else f"{config['dataset']}.inter"
    interaction_file_name = os.path.join(base_path, interaction_file_name)
    
    interaction_df = pd.read_csv(interaction_file_name, sep='\t')
    interaction_df.columns = del_postfix(interaction_df.columns, sep=":")
    
    users = interaction_df[config["USER_ID_FIELD"]].unique()
    items = interaction_df[config["ITEM_ID_FIELD"]].unique()
    
    if Path(user_file_name).exists():
        user_df = pd.read_csv(user_file_name, sep='\t')
        user_df.columns = del_postfix(user_df.columns, sep=":")
    else:
        user_df = pd.DataFrame(users, columns=[config["USER_ID_FIELD"]])
    
    if Path(item_file_name).exists():
        item_df = pd.read_csv(item_file_name, sep='\t')
        item_df.columns = del_postfix(item_df.columns, sep=":")
    else:
        item_df = pd.DataFrame(items, columns=[config["ITEM_ID_FIELD"]])
        
    return user_df, item_df, interaction_df
    
    
def del_postfix(p_l, sep=":"):
    rr = []
    for i in p_l:
        rr.append(i.split(sep)[0])
    return rr