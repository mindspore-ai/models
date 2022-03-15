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
run model train
"""
import math
import os
import numpy as np
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import Model
from mindspore.common import set_seed
import mindspore.ops as ops
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore import Tensor, context, export, load_checkpoint, load_param_into_net
from mindspore.train.callback import (CheckpointConfig, LossMonitor,
                                      ModelCheckpoint, TimeMonitor)
from src.dataset.DatasetGenerator import DatasetGenerator
from src.dataset.MPIIDataLoader import MPII
from src.models.loss import HeatmapLoss
from src.models.StackedHourglassNet import StackedHourglassNet
from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper

set_seed(1)

args = config


@moxing_wrapper()
def run_train():
    """
    run_train
    """
    if not os.path.exists(args.img_dir) or not os.path.exists(args.annot_dir):
        print("Dataset not found.")
        exit()

    # Set context mode
    if args.context_mode == "GRAPH":
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, save_graphs=False)
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)

    if args.parallel:
        # Parallel mode
        context.reset_auto_parallel_context()
        init()
        context.set_auto_parallel_context(parallel_mode=context.ParallelMode.DATA_PARALLEL, gradients_mean=True)
        args.rank_id = get_rank()
        args.group_size = get_group_size()
    else:
        args.rank_id = 0
        args.group_size = 1

    net = StackedHourglassNet(args.nstack, args.inp_dim, args.oup_dim)

    # Process dataset
    mpii = MPII()
    train, _ = mpii.setup_val_split()

    train_generator = DatasetGenerator(args.input_res, args.output_res, mpii, train)
    train_size = len(train_generator)
    train_sampler = ds.DistributedSampler(num_shards=args.group_size, shard_id=args.rank_id, shuffle=True)
    train_data = ds.GeneratorDataset(train_generator, ["data", "label"], sampler=train_sampler)
    train_data = train_data.batch(args.batch_size, True, args.group_size)

    print("train data size:", train_size)
    step_per_epoch = math.ceil(train_size / args.batch_size / args.group_size)

    # Define loss function
    loss_func = HeatmapLoss()
    # Define optimizer
    lr_decay = nn.exponential_decay_lr(
        args.initial_lr, args.decay_rate, args.num_epoch * step_per_epoch, step_per_epoch, args.decay_epoch
    )
    optimizer = nn.Adam(net.trainable_params(), lr_decay)

    # Define model
    model = Model(net, loss_func, optimizer, amp_level=args.amp_level, keep_batchnorm_fp32=False)

    # Define callback functions
    callbacks = []
    callbacks.append(LossMonitor(args.loss_log_interval))
    callbacks.append(TimeMonitor(train_size))

    # Save checkpoint file
    if args.rank_id == 0:
        config_ck = CheckpointConfig(
            save_checkpoint_steps=args.save_checkpoint_epochs * step_per_epoch,
            keep_checkpoint_max=args.keep_checkpoint_max,
        )
        ckpoint = ModelCheckpoint("ckpt", directory=args.output_path, config=config_ck)
        callbacks.append(ckpoint)

    model.train(args.num_epoch, train_data, callbacks=callbacks, dataset_sink_mode=True)

    ckpt_file = _get_last_ckpt(args.output_path)
    air_file = os.path.join(args.output_path, args.file_name)
    print('air_file: ', air_file)
    run_export(ckpt_file, air_file)


flipped_parts = {"mpii": [5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10]}


class MaxPool2dFilter(nn.Cell):
    """
    maxpool 2d for filter
    """

    def __init__(self):
        super(MaxPool2dFilter, self).__init__()
        self.pool = nn.MaxPool2d(3, 1, "same")
        self.eq = ops.Equal()

    def construct(self, x):
        """
        forward
        """
        maxm = self.pool(x)
        return self.eq(maxm, x)


class Hourglass(nn.Cell):
    """
        Hourglass
    """

    def __init__(self, net):
        super(Hourglass, self).__init__(auto_prefix=False)
        self.net = net
        self.pool = nn.MaxPool2d(3, 1, "same")
        self.eq = ops.Equal()
        self.topk = ops.TopK(sorted=True)
        self.stack = ops.Stack(axis=3)
        self.reshape = ops.Reshape()

    def construct(self, input_arr, input_arr2):
        """
        forward
        """
        tmp1 = self.net(input_arr)
        tmp2 = self.net(input_arr2)
        tmp = ops.Concat(0)((tmp1, tmp2))

        det = tmp[0, -1] + tmp[1, -1, :, :, ::-1][flipped_parts["mpii"]]

        det = det / 2

        det = ops.minimum(det, 1)
        det0 = det
        det = ops.expand_dims(det, 0)
        maxm = self.pool(det)
        maxm = self.eq(maxm, det)
        maxm = det * maxm

        w = maxm.shape[3]
        det1 = self.reshape(maxm, (maxm.shape[0], maxm.shape[1], -1))
        val_k, ind = self.topk(det1, 1)
        x = (ind % w).astype(np.float32)
        y = (ind.astype(np.float32) / w)
        ind_k = self.stack((x, y))
        loc = ind_k[0, :, 0, :]
        val = val_k[0, :, :]
        ans = ops.Concat(1)((loc, val))
        ans = ops.expand_dims(ans, 0)
        ans = ops.expand_dims(ans, 0)

        return det0, ans


def _get_last_ckpt(ckpt_dir):
    """
    _get_last_ckpt
    """
    file_dict = {}
    lists = os.listdir(ckpt_dir)
    for name in lists:
        if name.endswith('.ckpt'):
            i = name.index("-")
            j = name.index("_")
            ctime = int(name[i + 1:j])
            file_dict[ctime] = name
    max_ctime = max(file_dict.keys())
    ckpt_file = os.path.join(ckpt_dir, file_dict[max_ctime])
    print('ckpt_file: ', ckpt_file)
    if not ckpt_file:
        print("No ckpt file found.")
        return None
    return ckpt_file


def run_export(ckpt_file, air_file):
    """
    run_export
    """
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, save_graphs=False)
    network = StackedHourglassNet(args.nstack, args.inp_dim, args.oup_dim)
    param_dict = load_checkpoint(ckpt_file)
    load_param_into_net(network, param_dict)
    input_arr = Tensor(np.zeros([1, args.input_res, args.input_res, 3], np.float32))
    input_arr2 = Tensor(np.zeros([1, args.input_res, args.input_res, 3], np.float32))
    network = Hourglass(network)
    export(network, input_arr, input_arr2, file_name=air_file, file_format=args.file_format)


if __name__ == "__main__":
    run_train()
