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
"""OSVOS train."""
import os
import time
import argparse
import math

import mindspore.nn as nn
from mindspore import Model
from mindspore import context, load_param_into_net, load_checkpoint
from mindspore.train.model import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.common import set_seed
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.communication.management import init, get_rank, get_group_size

from src.config import osvos_cfg
from src.vgg_osvos import OSVOS
from src.utils import ClassBalancedCrossEntropyLoss
from src.dataset import create_dataset

parser = argparse.ArgumentParser(description='OSVOS train running')
parser.add_argument('--device_target', type=str, default="GPU", choices=['Ascend', 'GPU', 'CPU'],
                    help='device where the code will be implemented (default: GPU)')
parser.add_argument('--run_distribute', type=int, default=0, help='0 -- run standalone, 1 -- run distribute')
parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
parser.add_argument("--stage", type=int, default=1, choices=[1, 2], help="choose train stage, default is 1.")
parser.add_argument("--seq_name", type=str, default='blackswan', help="the sequence name for stage 2.")
parser.add_argument("--data_path", type=str, default="./DAVIS", help="the dataset path, default is ./DAVIS")
parser.add_argument("--parent_ckpt_path", type=str, default=None, help="the parent ckpt path, default is None")
parser.add_argument("--vgg_features_ckpt", type=str, default='./models/vgg16_features.ckpt',
                    help="Path to the vgg pretrain model.")

def train_parent(args, cfg):
    """train stage 1, train parent network."""
    data_path = args.data_path

    lr = cfg.tp_lr
    epoch_size = cfg.tp_epoch_size
    batch_size = cfg.tp_batch_size

    rank_id = 0
    ckpt_dir = cfg.dirResult + '/parent'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    if args.run_distribute:
        init()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=args.device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        rank_id = get_rank()
        dataset_train = create_dataset(mode="Train",
                                       data_path=data_path,
                                       batch_size=batch_size,
                                       num_of_workers=4,
                                       num_of_epoch=1,
                                       is_distributed=args.run_distribute,
                                       rank=get_rank(),
                                       group_size=get_group_size(),
                                       seed=0)
    else:
        context.set_context(device_id=args.device_id)
        dataset_train = create_dataset(mode="Train",
                                       data_path=data_path,
                                       batch_size=batch_size,
                                       num_of_workers=4,
                                       num_of_epoch=1)

    rank_sava_flag = False
    if rank_id == 1 or args.device_num == 1:
        rank_sava_flag = True

    batch_num = dataset_train.get_dataset_size()
    print(f'batch_num:{batch_num}')
    print(f'lr:{lr}')
    print(f'epoch_size:{epoch_size}')

    learning_rate = []
    warm_up = [lr / math.floor(epoch_size / 5) * (i + 1) for _ in range(batch_num) for i in
               range(math.floor(epoch_size / 5))]
    shrink = [lr / (16 * (i + 1)) for _ in range(batch_num)
              for i in range(math.floor(epoch_size * 2 / 5))]
    normal_run = [lr for _ in range(batch_num) for i in
                  range(epoch_size - math.floor(epoch_size / 5) - math.floor(epoch_size * 2 / 5))]
    learning_rate = learning_rate + warm_up + normal_run + shrink

    net = OSVOS(args.vgg_features_ckpt)
    net.set_train()
    net_loss = ClassBalancedCrossEntropyLoss()

    opt = nn.Adam(net.trainable_params(),
                  learning_rate=learning_rate, use_nesterov=True, weight_decay=1e-5)

    loss_scale_manager = FixedLossScaleManager(1024, drop_overflow_update=False)
    model = Model(net, loss_fn=net_loss, optimizer=opt, loss_scale_manager=loss_scale_manager)
    time_cb = TimeMonitor(data_size=batch_num)
    loss_cb = LossMonitor(per_print_times=batch_num)
    cb = [time_cb, loss_cb]
    if args.run_distribute:
        ckpt_dir = os.path.join(ckpt_dir, str(rank_id))
    if rank_sava_flag:
        config_ck = CheckpointConfig(keep_checkpoint_max=10, saved_network=net)
        ckpoint_cb = ModelCheckpoint(prefix='checkpoint_parent', directory=ckpt_dir, config=config_ck)
        cb.append(ckpoint_cb)

    print("start train...")
    start = time.time()
    model.train(epoch_size, dataset_train, callbacks=cb)
    end = time.time()
    print(f"train success, use time {(end-start)/60} minutes")


def train_online(args, cfg):
    """train stage 2, train online network."""
    data_path = args.data_path
    seq_name = args.seq_name
    seq_name_list = {
        'blackswan': 1e-4,
        'goat': 1e-4,
        'car-shadow': 5e-6,
        'cows': 5e-5,
        'car-roundabout': 1e-5,
        'paragliding-launch': 1e-4,
        'horsejump-high': 1e-4,
        'dance-twirl': 7e-6,
        'drift-straight': 5e-9,
        'motocross-jump': 7e-7,
        'parkour': 1e-5,
        'soapbox': 5e-6,
        'camel': 7e-5,
        'kite-surf': 1e-5,
        'dog': 5e-7,
        'libby': 1e-5,
        'bmx-trees': 7e-5,
        'breakdance': 5e-5,
        'drift-chicane': 5e-7,
        'scooter-black': 5e-8,
    }
    print("Start of Online Training, sequence: " + seq_name)

    context.set_context(device_id=args.device_id)
    lr = seq_name_list[seq_name]
    epoch_size = cfg.to_epoch_size
    batch_size = cfg.to_batch_size

    print(f'lr:{lr}')

    save_dir = cfg.dirResult + '/online/' + args.seq_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)


    dataset_train = create_dataset(mode="Train",
                                   data_path=data_path,
                                   batch_size=batch_size,
                                   seq_name=seq_name,
                                   num_of_workers=4,
                                   num_of_epoch=1)

    batch_num = dataset_train.get_dataset_size()
    print(f'batch_num:{batch_num}')

    net = OSVOS()
    param_dict = load_checkpoint(args.parent_ckpt_path)
    load_param_into_net(net, param_dict)
    net.set_train()

    learning_rate = []
    warm_up = [lr/ math.floor(epoch_size / 5) * (i + 1) for _ in range(batch_num) for i in
               range(math.floor(epoch_size / 5))]
    shrink = [lr / (16 * (i + 1)) for _ in range(batch_num)
              for i in range(math.floor(epoch_size * 2 / 5))]
    normal_run = [lr for _ in range(batch_num) for i in
                  range(epoch_size - math.floor(epoch_size / 5) - math.floor(epoch_size * 2 / 5))]
    learning_rate = learning_rate + warm_up + normal_run + shrink
    opt = nn.Adam(net.trainable_params(),
                  learning_rate=learning_rate, use_nesterov=True, weight_decay=1e-5)

    net_loss = ClassBalancedCrossEntropyLoss(online=True)
    loss_scale_manager = FixedLossScaleManager(1024, drop_overflow_update=False)
    model = Model(net, loss_fn=net_loss, optimizer=opt, loss_scale_manager=loss_scale_manager)
    time_cb = TimeMonitor(data_size=batch_num)
    loss_cb = LossMonitor(per_print_times=batch_num)
    cb = [time_cb, loss_cb]

    config_ck = CheckpointConfig(keep_checkpoint_max=10, saved_network=net)
    ckpoint_cb = ModelCheckpoint(prefix='checkpoint_online', directory=save_dir, config=config_ck)
    cb.append(ckpoint_cb)

    print("start train...")
    start = time.time()
    model.train(epoch_size, dataset_train, callbacks=cb)
    end = time.time()
    print(f"train success, use time {(end-start)/60} minutes")


def main():
    """Main entrance for training"""
    args = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

    if args.stage == 1:
        train_parent(args, osvos_cfg)
    else:
        train_online(args, osvos_cfg)

if __name__ == '__main__':
    set_seed(1)
    main()
