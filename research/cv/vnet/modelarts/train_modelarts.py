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
"""vnet train."""

import os
import argparse
import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import Model
from mindspore import context, Tensor, export, load_checkpoint, load_param_into_net
from mindspore import dtype as mstype
from mindspore.train.model import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.common import set_seed
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.communication.management import init, get_rank, get_group_size

from src.config import vnet_cfg as cfg
from src.dataset import create_dataset
from src.vnet import VNet
from src.utils import dynamic_lr, get_rank_id



parser = argparse.ArgumentParser(description='Vnet train running')
parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'],
                    help='device where the code will be implemented (default: Ascend)')
parser.add_argument('--run_distribute', type=int, default=0, help='0 -- run standalone, 1 -- run distribute')
parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
parser.add_argument("--epochs", type=int, default=500, help="epochs, default is 500.")
parser.add_argument("--data_url", type=str, default="./promise", help="Path of dataset, default is ./promise")
parser.add_argument("--train_url", type=str, default="", help="Path of outputs")



def main():
    """Main entrance for training"""
    args = parser.parse_args()
    train_split_file_path = args.data_url + '/train.csv'
    results_dir = args.train_url
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    base_lr = cfg.lr
    if args.device_target == "Ascend":
        context.set_context(enable_auto_mixed_precision=True, device_id=args.device_id)
        rank_id = 0
        if args.run_distribute:
            init()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=args.device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            rank_id = get_rank_id()
    elif args.device_target == "GPU":
        base_lr = 0.0005
        rank_id = 0
        context.set_context(enable_graph_kernel=True)
        if args.run_distribute:
            init()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=args.device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            rank_id = get_rank()

    rank_save_flag = False
    if rank_id == 0 or args.device_num == 1:
        rank_save_flag = True

    ckpt_dir = results_dir

    if args.run_distribute:
        ds_train = create_dataset(mode='Train',
                                  parameters=cfg,
                                  data_path=args.data_url,
                                  split_file_path=train_split_file_path,
                                  batch_size=cfg.batch_size,
                                  num_of_workers=8,
                                  num_of_epoch=1,
                                  is_distributed=True,
                                  rank=get_rank(),
                                  group_size=get_group_size(),
                                  seed=0)
    else:
        ds_train = create_dataset(mode='Train',
                                  parameters=cfg,
                                  data_path=args.data_url,
                                  split_file_path=train_split_file_path,
                                  batch_size=cfg.batch_size,
                                  num_of_workers=8,
                                  num_of_epoch=1)

    network = VNet()
    network.set_train()
    net_loss = nn.DiceLoss()
    lr = Tensor(dynamic_lr(cfg, base_lr, ds_train.get_dataset_size()), mstype.float32)
    net_opt = nn.Adam(network.trainable_params(), lr)
    scale_manager = FixedLossScaleManager(256, drop_overflow_update=False)
    model = Model(network, net_loss, net_opt, loss_scale_manager=scale_manager)
    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
    cb = [time_cb, LossMonitor()]
    if args.run_distribute:
        ckpt_dir = os.path.join(ckpt_dir, str(rank_id))
    if rank_save_flag:
        config_ck = CheckpointConfig(keep_checkpoint_max=10, saved_network=network)
        ckpoint_cb = ModelCheckpoint(prefix="checkpoint_vnet", directory=ckpt_dir, config=config_ck)
        cb.append(ckpoint_cb)
    print("============== Starting Training fold {} ==============".format(str(cfg.fold)))
    model.train(args.epochs, ds_train, callbacks=cb)

    # export network


    def construct(self, x):
        """vnet construct"""
        B, X, Y, Z = x.shape
        x = x.view(B, 1, X, Y, Z)
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out

    VNet.construct = construct
    network = VNet(dropout=False)

    # load network checkpoint
    checkpoint_path = sorted([file for file in os.listdir(ckpt_dir) if '.ckpt' in file])[-1]
    print('export ckpt:', ckpt_dir + '/' + checkpoint_path)
    param_dict = load_checkpoint(ckpt_dir + '/' + checkpoint_path)
    load_param_into_net(network, param_dict)
    inputs = Tensor(np.ones([1, cfg.VolSize[0], cfg.VolSize[1], cfg.VolSize[2]]), mindspore.float32)
    export(network, inputs, file_name=ckpt_dir + '/vnet', file_format='AIR')

if __name__ == '__main__':
    set_seed(1)
    main()
