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
# ===========================================================================
"""Train Auto-DeepLab"""
import sys
import os

import math
import numpy as np

import moxing as mox

import mindspore
import mindspore.nn as nn
from mindspore import context
from mindspore.context import ParallelMode
from mindspore import Model
from mindspore import load_checkpoint, load_param_into_net, export
from mindspore.train.callback import TimeMonitor, LossMonitor, ModelCheckpoint, CheckpointConfig
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore.communication import init


device_id = int(os.getenv('DEVICE_ID'))
device_num = int(os.getenv('RANK_SIZE'))
local_data_url = "/cache/data"
local_train_url = "/cache/train"


def get_last_ckpt(ckpt_dir):
    """get_last_ckpt"""
    ckpt_files = [ckpt_file for ckpt_file in os.listdir(ckpt_dir)
                  if ckpt_file.endswith('.ckpt')]
    if not ckpt_files:
        print("No ckpt file found.")
        return None

    return os.path.join(ckpt_dir, sorted(ckpt_files)[-1])


def export_air(args, ckpt_dir):
    """export_air"""
    args.parallel = False
    ckpt_file = get_last_ckpt(ckpt_dir)
    air_name = os.path.join(ckpt_dir, 'Auto-DeepLab-s_NHWC_BGR')
    if not ckpt_file:
        return 1

    # net
    autodeeplab = AutoDeepLab(args)

    # load checkpoint
    print('start export ', ckpt_file)
    param_dict = load_checkpoint(ckpt_file)

    # load the parameter into net
    load_param_into_net(autodeeplab, param_dict)
    network = InferWithFlipNetwork(autodeeplab, flip=args.infer_flip, input_format=args.input_format)

    input_data = np.random.uniform(0.0, 1.0, size=[1, 1024, 2048, 3]).astype(np.float32)
    export(network, mindspore.Tensor(input_data), file_name=air_name, file_format=args.file_format)

    return 0


def filter_weight(param_dict):
    """filter_weight"""
    for key in list(param_dict.keys()):
        if 'decoder.last_conv.6' in key:
            print('filter {}'.format(key))
            del param_dict[key]
    return  param_dict


def train():
    """train"""
    args = obtain_autodeeplab_args()
    prepare_seed(args.seed)

    context.set_context(mode=context.GRAPH_MODE,
                        save_graphs=False,
                        device_target=args.device_target,
                        device_id=int(os.getenv('DEVICE_ID')))
    ckpt_file = args.ckpt_name

    init()
    context.set_auto_parallel_context(device_num=device_num,
                                      parallel_mode=ParallelMode.DATA_PARALLEL,
                                      gradients_mean=True)
    shard_id = device_id
    num_shards = device_num
    device_data_url = os.path.join(local_data_url, "device{0}".format(device_id))
    device_train_url = os.path.join(local_train_url, "device{0}".format(device_id))
    local_train_file = os.path.join(device_data_url, 'cityscapes_train.mindrecord')
    if args.ckpt_name is not None:
        ckpt_file = os.path.join(device_data_url, args.ckpt_name)

    mox.file.make_dirs(local_data_url)
    mox.file.make_dirs(local_train_url)
    mox.file.make_dirs(device_data_url)
    mox.file.make_dirs(device_train_url)
    mox.file.copy_parallel(src_url=args.data_url, dst_url=device_data_url)

    # define dataset
    batch_size = int(args.batch_size // device_num) if args.parallel else args.batch_size
    crop = args.crop_size

    train_ds = CityScapesDataset(local_train_file, 'train', args.ignore_label, (crop, crop),
                                 num_shards, shard_id, shuffle=True)
    train_ds = train_ds.batch(batch_size=batch_size, drop_remainder=True)

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
        param_dict = filter_weight(param_dict) if args.filter_weight else param_dict
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

    export_air(args, device_train_url)

    if args.modelArts and device_id == 0:
        mox.file.copy_parallel(src_url="/cache/train", dst_url=args.train_url)
        mox.file.copy_parallel(src_url='/tmp', dst_url=args.train_url)

    return 0


if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from src.utils.loss import build_criterion
    from src.core.model import AutoDeepLab
    from src.utils.cityscapes import CityScapesDataset
    from src.utils.dynamic_lr import warmup_poly_lr
    from src.utils.utils import prepare_seed
    from infer.util.mindx_config import obtain_autodeeplab_args
    from infer.util.mindx_utils import InferWithFlipNetwork

    train()
