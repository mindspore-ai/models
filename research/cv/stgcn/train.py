# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
train network.
"""

import os
import pandas as pd
from sklearn import preprocessing

import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.common import set_seed, dtype as mstype
from mindspore.train.model import Model
from mindspore.train.callback import CheckpointConfig, LossMonitor, ModelCheckpoint, TimeMonitor
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size

from src.argparser import arg_parser
from src import dataloader, utility, config
from src.model import models, metric


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
            raise ValueError("Unsupported n_pred.")
    else:
        raise ValueError("Unsupported graph_conv_type.")

    return cfg


def get_params(args):
    """get and preprocess parameters"""
    cfg = get_config(args)

    if (cfg.graph_conv_type == 'gcnconv') and (cfg.Ks != 2):
        cfg.Ks = 2
    Ko = cfg.n_his - (cfg.Kt - 1) * 2 * cfg.stblock_num

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

    time_pred = cfg.n_pred * cfg.time_intvl
    time_pred_str = str(time_pred) + '_mins'

    if cfg.graph_conv_type == "chebconv":
        if (cfg.mat_type != "wid_sym_normd_lap_mat") and (cfg.mat_type != "wid_rw_normd_lap_mat"):
            raise ValueError(f'ERROR: {cfg.mat_type} is wrong.')
    elif cfg.graph_conv_type == "gcnconv":
        if (cfg.mat_type != "hat_sym_normd_lap_mat") and (cfg.mat_type != "hat_rw_normd_lap_mat"):
            raise ValueError(f'ERROR: {cfg.mat_type} is wrong.')

    return args, cfg, blocks, time_pred_str


def run_train(args, cfg, blocks, time_pred_str):
    """train stgcn model"""
    target = args.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
    if args.run_modelarts:
        import moxing as mox
        device_id = int(os.getenv('DEVICE_ID'))
        device_num = int(os.getenv('RANK_SIZE'))
        cfg.batch_size = cfg.batch_size * int(8/device_num)
        context.set_context(device_id=device_id)
        local_data_url = '/cache/data'
        local_train_url = '/cache/train'
        mox.file.copy_parallel(args.data_url, local_data_url)
        if device_num > 1:
            init()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
        data_dir = local_data_url + '/'
    else:
        if target == "Ascend":
            device_id = 0
            device_num = 1
            context.set_context(device_id=args.device_id)
            if args.run_distribute:
                device_id = int(os.getenv('DEVICE_ID'))
                device_num = int(os.getenv('RANK_SIZE'))
                context.set_context(device_id=device_id)
                init()
                context.reset_auto_parallel_context()
                # context.set_auto_parallel_context(parameter_broadcast=True)
                context.set_auto_parallel_context(device_num=device_num,
                                                  parallel_mode=ParallelMode.DATA_PARALLEL,
                                                  gradients_mean=True)
        elif target == "GPU":
            device_id = args.device_id
            device_num = 1
            if args.run_distribute:
                init()
                device_id = get_rank()
                device_num = get_group_size()
                context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                                  device_num=device_num)
        else:
            raise ValueError("Unsupported platform, only GPU or Ascend is supported.")

        cfg.batch_size = cfg.batch_size * int(8 / device_num)
        data_dir = args.data_url + '/'
        model_save_path = args.train_url + cfg.graph_conv_type + '_' + time_pred_str
        if args.device_target == "GPU" and args.run_distribute:
            model_save_path = os.path.join(model_save_path, "ckpt_" + str(device_id) + "/")

    adj_mat = dataloader.load_weighted_adjacency_matrix(os.path.join(data_dir, args.wam_path))
    n_vertex_vel = pd.read_csv(os.path.join(data_dir, args.data_path), header=None).shape[1]
    n_vertex_adj = pd.read_csv(os.path.join(data_dir, args.wam_path), header=None).shape[1]
    if n_vertex_vel == n_vertex_adj:
        n_vertex = n_vertex_vel
    else:
        raise ValueError(f"ERROR: number of vertices in dataset is not equal to number \
            of vertices in weighted adjacency matrix.")
    mat = utility.calculate_laplacian_matrix(adj_mat, cfg.mat_type)
    conv_matrix = Tensor(Tensor.from_numpy(mat), mstype.float32)

    net = models.STGCN_Conv(cfg.Kt, cfg.Ks, blocks, cfg.n_his, n_vertex, cfg.gated_act_func,
                            cfg.graph_conv_type, conv_matrix, cfg.drop_rate)

    # start training
    zscore = preprocessing.StandardScaler()
    dataset = dataloader.create_dataset(os.path.join(data_dir, args.data_path), cfg.batch_size, cfg.n_his, cfg.n_pred,
                                        zscore, device_num, device_id, mode=0)
    dataset_size = dataset.get_dataset_size()

    learning_rate = nn.exponential_decay_lr(learning_rate=cfg.learning_rate, decay_rate=cfg.gamma,
                                            total_step=dataset_size*args.epochs, step_per_epoch=dataset_size,
                                            decay_epoch=cfg.decay_epoch)

    if cfg.opt == "RMSProp":
        optimizer = nn.RMSProp(net.trainable_params(), learning_rate=learning_rate)
    elif cfg.opt == "Adam":
        optimizer = nn.Adam(net.trainable_params(), learning_rate=learning_rate, weight_decay=cfg.weight_decay_rate)
    elif cfg.opt == "AdamW":
        optimizer = nn.AdamWeightDecay(net.trainable_params(), learning_rate=learning_rate,
                                       weight_decay=cfg.weight_decay_rate)
    else:
        raise ValueError(f'ERROR: optimizer {cfg.opt} is undefined.')

    loss_cb = LossMonitor(per_print_times=dataset_size)
    time_cb = TimeMonitor()
    callbacks = [time_cb, loss_cb]

    # save training results
    if args.save_checkpoint and (device_num == 1 or device_id == 0):
        ckpt_config = CheckpointConfig(save_checkpoint_steps=dataset_size * args.epochs,
                                       keep_checkpoint_max=args.epochs)
        if args.run_modelarts:
            ckpt_cb = ModelCheckpoint(prefix='STGCN' + cfg.graph_conv_type + str(cfg.n_pred) + '-',
                                      directory=local_train_url, config=ckpt_config)
        else:
            ckpt_cb = ModelCheckpoint(prefix='STGCN', directory=model_save_path, config=ckpt_config)
        callbacks += [ckpt_cb]

    net = metric.LossCellWithNetwork(net)
    model = Model(net, optimizer=optimizer)

    model.train(args.epochs, dataset, callbacks=callbacks, dataset_sink_mode=False)
    if args.run_modelarts:
        mox.file.copy_parallel(src_url=local_train_url, dst_url=args.train_url)


if __name__ == "__main__":
    set_seed(1)
    run_train(*get_params(arg_parser()))
