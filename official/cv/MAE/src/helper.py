# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os
import sys
from multiprocessing import Process

import yaml
import numpy as np


from mindspore import context
from mindspore.common import set_seed
from mindspore.communication.management import init, get_group_size, get_rank


PARALLEL_MODE = {"DATA_PARALLEL": context.ParallelMode.DATA_PARALLEL,
                 "SEMI_AUTO_PARALLEL": context.ParallelMode.SEMI_AUTO_PARALLEL,
                 "AUTO_PARALLEL": context.ParallelMode.AUTO_PARALLEL,
                 "HYBRID_PARALLEL": context.ParallelMode.HYBRID_PARALLEL}
MODE = {"PYNATIVE_MODE": context.PYNATIVE_MODE,
        "GRAPH_MODE": context.GRAPH_MODE}


def cloud_context_init(seed=0,
                       use_parallel=True,
                       context_config=None,
                       parallel_config=None):
    np.random.seed(seed)
    set_seed(seed)
    device_num = 1
    rank_id = 0
    context_config["mode"] = MODE[context_config["mode"]]
    if use_parallel:
        init()
        device_id = int(os.getenv('DEVICE_ID'))  # 0 ~ 7
        rank_id = get_rank()  # local_rank
        device_num = get_group_size()  # world_size
        context_config["device_id"] = device_id
        parallel_config["parallel_mode"] = PARALLEL_MODE[parallel_config["parallel_mode"]]
        context.set_context(**context_config)
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            device_num=device_num, **parallel_config)
    else:
        context.set_context(**context_config)
    os.environ['MOX_SILENT_MODE'] = '1'
    return rank_id, device_num


def sync_trans(f):
    try:
        def wrapper(*args, **kwargs):
            pro = Process(target=f, args=args, kwargs=kwargs)
            pro.start()
            return pro
        return wrapper
    except Exception as e:
        raise e


def check_obs_url(url: str):
    if url.startswith("s3") or url.startswith("obs"):
        return True
    return False


def str2bool(b):
    if b.lower() in ["false"]:
        output = False
    elif b.lower() in ["true"]:
        output = True
    else:
        raise Exception("Invalid Bool Value")
    return output


def parse_with_config(parser):
    """Parse With Config"""
    args = parser.parse_args()
    if args.config is not None:
        config_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
        override_keys = {arg[2:].split('=')[0] for arg in sys.argv[1:]
                         if arg.startswith('--')}
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
    del args.config
    return args

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# MoCo v3: https://github.com/facebookresearch/moco-v3

# --------------------------------------------------------


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
