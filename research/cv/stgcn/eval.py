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
"""
testing network performance.
"""

import os
import pandas as pd
from sklearn import preprocessing

from mindspore import context, Tensor
from mindspore.common import set_seed, dtype as mstype
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.argparser import arg_parser
from src.model import models
from src import dataloader, utility, config

os.system("export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python")


def get_config(args):
    """return config based on selected n_pred and graph_conv_type"""
    if args.graph_conv_type == "chebconv":
        if args.n_pred == 9:
            cfg = config.stgcn_chebconv_45min_cfg
        elif args.n_pred == 6:
            cfg = config.stgcn_chebconv_30min_cfg
        elif args.n_pred == 3:
            cfg = config.stgcn_chebconv_15min_cfg
        else:
            raise ValueError("Unsupported n_pred.")
    elif args.graph_conv_type == "gcnconv":
        if args.n_pred == 9:
            cfg = config.stgcn_gcnconv_45min_cfg
        elif args.n_pred == 6:
            cfg = config.stgcn_gcnconv_30min_cfg
        elif args.n_pred == 3:
            cfg = config.stgcn_gcnconv_15min_cfg
        else:
            raise ValueError("Unsupported pred.")
    else:
        raise ValueError("Unsupported graph_conv_type.")

    return cfg


def get_params():
    """get and preprocess parameters"""
    args = arg_parser()
    cfg = get_config(args)

    if ((cfg.Kt - 1) * 2 * cfg.stblock_num > cfg.n_his) or ((cfg.Kt - 1) * 2 * cfg.stblock_num <= 0):
        raise ValueError(f'ERROR: {cfg.Kt} and {cfg.stblock_num} are unacceptable.')
    Ko = cfg.n_his - (cfg.Kt - 1) * 2 * cfg.stblock_num
    if (cfg.graph_conv_type != "chebconv") and (cfg.graph_conv_type != "gcnconv"):
        raise NotImplementedError(f'ERROR: {cfg.graph_conv_type} is not implemented.')

    if (cfg.graph_conv_type == 'gcnconv') and (cfg.Ks != 2):
        cfg.Ks = 2

    target = args.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)

    # blocks: settings of channel size in st_conv_blocks and output layer,
    # using the bottleneck design in st_conv_blocks
    blocks = [[1]]
    for _ in range(cfg.stblock_num):
        blocks.append([64, 16, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([1])

    cfg.n_pred = cfg.n_pred

    return args, cfg, blocks


def run_eval(args, cfg, blocks):
    """evaluate stgcn model"""
    if args.run_modelarts:
        import moxing as mox
        device_num = 1
        cfg.batch_size = cfg.batch_size*int(8/device_num)
        local_data_url = '/cache/data'
        local_ckpt_url = '/cache/ckpt'
        mox.file.copy_parallel(args.data_url, local_data_url)
        mox.file.copy_parallel(args.ckpt_url, local_ckpt_url)
        data_dir = local_data_url + '/'
        local_ckpt_url = local_ckpt_url + '/'
    else:
        context.set_context(device_id=args.device_id)
        data_dir = args.data_url + '/'
        local_ckpt_url = args.ckpt_url

    adj_mat = dataloader.load_weighted_adjacency_matrix(data_dir+args.wam_path)

    n_vertex_vel = pd.read_csv(data_dir+args.data_path, header=None).shape[1]
    n_vertex_adj = pd.read_csv(data_dir+args.wam_path, header=None).shape[1]
    if n_vertex_vel == n_vertex_adj:
        n_vertex = n_vertex_vel
    else:
        raise ValueError(f'ERROR: number of vertices in dataset is not equal to \
         number of vertices in weighted adjacency matrix.')

    mat = utility.calculate_laplacian_matrix(adj_mat, cfg.mat_type)
    conv_matrix = Tensor(Tensor.from_numpy(mat), mstype.float32)
    if cfg.graph_conv_type == "chebconv":
        if (cfg.mat_type != "wid_sym_normd_lap_mat") and (cfg.mat_type != "wid_rw_normd_lap_mat"):
            raise ValueError(f'ERROR: {cfg.mat_type} is wrong.')
    elif cfg.graph_conv_type == "gcnconv":
        if (cfg.mat_type != "hat_sym_normd_lap_mat") and (cfg.mat_type != "hat_rw_normd_lap_mat"):
            raise ValueError(f'ERROR: {cfg.mat_type} is wrong.')

    net = models.STGCN_Conv(cfg.Kt, cfg.Ks, blocks, cfg.n_his, n_vertex, cfg.gated_act_func,
                            cfg.graph_conv_type, conv_matrix, cfg.drop_rate)

    # start evaluation
    zscore = preprocessing.StandardScaler()
    dataset = dataloader.create_dataset(data_dir+args.data_path, cfg.batch_size, cfg.n_his, cfg.n_pred, zscore, mode=2)

    param_dict = load_checkpoint(local_ckpt_url)
    load_param_into_net(net, param_dict)

    test_MAE, test_RMSE, test_MAPE = utility.evaluate_metric(net, dataset, zscore)
    print(f'MAE {test_MAE:.2f} | MAPE {test_MAPE*100:.2f} | RMSE {test_RMSE:.2f}')


if __name__ == "__main__":
    set_seed(1)
    run_eval(*get_params())
