# Copyright 2021 Huawei Technologies Co., Ltd
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
# ===========================================================================
"""Train Auto-DeepLab"""

import os
import math
import numpy as np

import mindspore
import mindspore.nn as nn
from mindspore import context
from mindspore.context import ParallelMode
from mindspore import Model
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train.callback import TimeMonitor, LossMonitor, ModelCheckpoint, CheckpointConfig
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore.communication import init


from src.utils.loss import build_criterion
from src.core.model import AutoDeepLab
from src.utils.cityscapes import CityScapesDataset
from src.config import obtain_autodeeplab_args
from src.utils.dynamic_lr import warmup_poly_lr
from src.utils.utils import prepare_seed

device_id = int(os.getenv('DEVICE_ID'))
device_num = int(os.getenv('RANK_SIZE'))
local_data_url = "/cache/data"
local_train_url = "/cache/train"


def train():
    """train"""
    args = obtain_autodeeplab_args()
    prepare_seed(args.seed)

    context.set_context(mode=context.GRAPH_MODE,
                        save_graphs=False,
                        device_target=args.device_target,
                        device_id=int(os.getenv('DEVICE_ID')))
    ckpt_file = args.ckpt_name
    if args.modelArts:
        import moxing as mox
        init()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        shard_id = device_id
        num_shards = device_num
        device_data_url = os.path.join(local_data_url, "device{0}".format(device_id))
        device_train_url = os.path.join(local_train_url, "device{0}".format(device_id))
        local_train_file = os.path.join(device_data_url, 'cityscapes_train.mindrecord')
        local_val_file = os.path.join(device_data_url, 'cityscapes_val.mindrecord')
        if args.ckpt_name is not None:
            ckpt_file = os.path.join(device_data_url, args.ckpt_name)

        mox.file.make_dirs(local_data_url)
        mox.file.make_dirs(local_train_url)
        mox.file.make_dirs(device_data_url)
        mox.file.make_dirs(device_train_url)
        mox.file.copy_parallel(src_url=args.data_url, dst_url=device_data_url)
    else:
        if args.parallel:
            rank_id = int(os.getenv('RANK_ID'))
            init()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            shard_id = rank_id
            num_shards = device_num
            device_train_url = os.path.join(args.out_path, "device{0}".format(device_id))
        else:
            shard_id = None
            num_shards = None
            device_train_url = os.path.join(args.out_path, "device")
        local_train_file = os.path.join(args.data_path, 'cityscapes_train.mindrecord')
        local_val_file = os.path.join(args.data_path, 'cityscapes_val.mindrecord')

    # define dataset
    batch_size = int(args.batch_size // device_num) if args.parallel else args.batch_size
    crop = args.crop_size

    train_ds = CityScapesDataset(local_train_file, 'train', args.ignore_label, (crop, crop),
                                 num_shards, shard_id, shuffle=True)
    train_ds = train_ds.batch(batch_size=batch_size, drop_remainder=True)

    eval_ds = CityScapesDataset(local_val_file, 'eval', args.ignore_label, None,
                                num_shards, shard_id, shuffle=False)
    eval_ds = eval_ds.batch(batch_size=batch_size)

    # counting steps
    iter_per_epoch = train_ds.get_dataset_size()
    total_iters = iter_per_epoch * args.epochs
    args.total_iters = total_iters

    # define model
    print(args)
    net = AutoDeepLab(args)

    # loading checkpoints
    if args.ckpt_name is not None:
        print('=> loading checkpoint {0}'.format(args.ckpt_name))
        param_dict = load_checkpoint(ckpt_file)
        load_param_into_net(net, param_dict)
        print('=> successfully loaded checkpoint {0}'.format(args.ckpt_name))

    # loss
    if args.criterion == 'ohemce':
        args.thresh = -math.log(args.ohem_thresh)
        args.n_min = int((batch_size * args.crop_size * args.crop_size) // 16)
    criterion = build_criterion(args)

    # learning rate
    min_lr = args.min_lr if args.min_lr is not None else 0.0
    lr = warmup_poly_lr(args.warmup_start_lr, args.base_lr, min_lr, args.warmup_iters, total_iters)

    # recover training
    current_iter = args.start_epoch * iter_per_epoch
    epochs = args.epochs - args.start_epoch
    lr = lr[current_iter::]
    lr = mindspore.Tensor(np.array(lr).astype(np.float32))

    # optimizer
    bias_params = list(filter(lambda x: ('gamma' in x.name) or ('beta' in x.name), net.trainable_params()))
    no_bias_params = list(filter(lambda x: ('gamma' not in x.name) and ('beta' not in x.name), net.trainable_params()))
    group_params = [{'params': bias_params, 'weight_decay': 0.0},
                    {'params': no_bias_params, 'weight_decay': args.weight_decay},
                    {'order_params': net.trainable_params()}]
    optimizer = nn.Momentum(params=group_params,
                            learning_rate=lr,
                            momentum=0.9,
                            weight_decay=args.weight_decay)
    loss_scale_manager = DynamicLossScaleManager()

    # model
    model = Model(net, criterion, loss_scale_manager=loss_scale_manager, optimizer=optimizer)

    # callback for loss & time cost
    loss_cb = LossMonitor()
    time_cb = TimeMonitor(data_size=iter_per_epoch)
    cbs = [loss_cb, time_cb]

    # callback for saving ckpt
    config_ckpt = CheckpointConfig(save_checkpoint_steps=args.save_epochs * iter_per_epoch, keep_checkpoint_max=40)
    ckpt_cb = ModelCheckpoint(prefix='autodeeplab-paral', directory=device_train_url, config=config_ckpt)
    cbs += [ckpt_cb]

    model.train(epochs, train_ds, callbacks=cbs, dataset_sink_mode=True)

    if args.modelArts:
        mox.file.copy_parallel(src_url="/cache/train", dst_url=args.train_url)
        mox.file.copy_parallel(src_url='/tmp', dst_url=args.train_url)

    return 0


if __name__ == "__main__":
    train()
