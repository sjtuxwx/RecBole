# @Time   : 2020/7/20
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2022/7/8, 2020/10/3, 2020/10/1
# @Author : Zhen Tian, Yupeng Hou, Zihan Lin
# @Email  : chenyuwuxinn@gmail.com, houyupeng@ruc.edu.cn, zhlin@ruc.edu.cn

import argparse
import logging


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# logging.disable(logging.WARNING)

# warnings.filterwarnings("ignore")

from recbole.quick_start import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="LightGCN", help="name of models")
    parser.add_argument(
        "--dataset", "-d", type=str, default="ml-1m", help="name of datasets"
    )
    parser.add_argument("--config_files", type=str, default=None, help="config files")
    parser.add_argument(
        "--nproc", type=int, default=1, help="the number of process in this group"
    )
    parser.add_argument(
        "--ip", type=str, default="localhost", help="the ip of master node"
    )
    parser.add_argument(
        "--port", type=str, default="5678", help="the port of master node"
    )
    parser.add_argument(
        "--world_size", type=int, default=-1, help="total number of jobs"
    )
    parser.add_argument(
        "--group_offset",
        type=int,
        default=0,
        help="the global rank offset of this group",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        help="the global rank offset of this group",
    )
    parser.add_argument(
        "--cl_rate",
        type=float,
        default=0.2,
        help="cl learning rate"
    )
    parser.add_argument(
        "--enable_user_loss",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--item_loss_type",
        type=str,
        default=None
    )
    parser.add_argument(
        "--item_rq_loss_rate",
        type=float,
        default=0
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.2
    )
    parser.add_argument(
        "--log_type",
        type=str,
        default="fair"
    )
    parser.add_argument(
        "--pop_rate",
        type=float,
        default=0.2
    )
    parser.add_argument(
        "--gama",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=None
    )
    
    parser.add_argument(
        "--description",
        type=str,
        default=None
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="train"
    )

    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="checkpoint file to resume training"
    )
    
    parser.add_argument(
        "--pop_loss_rate",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--content_bpr_loss_rate",
        type=float,
        default=0.0
    )

    args, _ = parser.parse_known_args()

    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )

    run(
        args.model,
        args.dataset,
        config_file_list=config_file_list,
        nproc=args.nproc,
        world_size=args.world_size,
        ip=args.ip,
        port=args.port,
        group_offset=args.group_offset,
        devices=args.gpu_ids,
        args=vars(args)
    )
