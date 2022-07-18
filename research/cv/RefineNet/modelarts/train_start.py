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
""" train Refinenet """
import argparse
import math
import os

import moxing as mox
import numpy as np

from mindspore import Parameter, context, Tensor, export
from mindspore.train.model import Model
import mindspore.nn as nn
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.common import set_seed
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.common.initializer import initializer, HeUniform
from mindspore.context import ParallelMode

from src import dataset as data_generator
from src import loss, learning_rates
from src.refinenet import RefineNet
from src.refinenet import Bottleneck

set_seed(1)


def parse_args():
    """parse_args"""
    parser = argparse.ArgumentParser('MindSpore Refinet training')
    # dataset
    parser.add_argument('--data_file', type=str, default='', help='path and Name of the first MindRecord file')
    parser.add_argument('--data_file2', type=str, default='', help='path and Name of the second MindRecord file')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--crop_size', type=int, default=513, help='crop size')
    parser.add_argument('--image_mean', type=list, default=[123.675, 116.28, 103.53], help='image mean')# rgb
    parser.add_argument('--image_std', type=list, default=[58.395, 57.120, 57.375], help='image std') #rgb
    parser.add_argument('--min_scale', type=float, default=0.5, help='minimum scale of data argumentation')
    parser.add_argument('--max_scale', type=float, default=3.0, help='maximum scale of data argumentation')
    parser.add_argument('--ignore_label', type=int, default=255, help='ignore label')
    parser.add_argument('--num_classes', type=int, default=21, help='number of classes')

    # optimizer
    parser.add_argument('--train_epochs', type=int, default=10, help='epoch')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='warmup_epoch')
    parser.add_argument('--lr_type', type=str, default='cos', help='type of learning rate')
    parser.add_argument('--lr_decay_step', type=int, default=450, help='learning rate decay step')
    parser.add_argument('--lr_decay_rate', type=float, default=0.8, help='learning rate decay rate')
    parser.add_argument('--loss_scale', type=float, default=1024.0, help='loss scale')

    # model
    parser.add_argument('--model', type=str, default='refinenet', help='select model')
    parser.add_argument('--freeze_bn', action='store_true', help='freeze bn')
    parser.add_argument('--ckpt_pre_trained', type=str, default='', help='PreTrained model')

    # train
    parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'CPU', 'GPU'],
                        help='device where the code will be implemented. (Default: Ascend)')
    parser.add_argument('--is_distributed', action='store_true', help='distributed training')
    parser.add_argument('--rank', type=int, default=0, help='local rank of distributed')
    parser.add_argument('--group_size', type=int, default=1, help='world size of distributed')
    parser.add_argument('--save_epochs', type=int, default=5, help='steps interval for saving')
    parser.add_argument('--keep_checkpoint_max', type=int, default=200, help='max checkpoint for saving')
    parser.add_argument('--data_lst', type=str, default='', help='list of val data')
    parser.add_argument('--output_path', type=str, default='', help='path to save ckpt file')
    arg, _ = parser.parse_known_args()
    return arg


def weights_init(net):
    for _, cell in net.cells_and_names():
        if isinstance(cell, nn.Conv2d):
            cell.weight = Parameter(initializer(HeUniform(negative_slope=math.sqrt(5)), cell.weight.shape,
                                                cell.weight.dtype), name=cell.weight.name)


def get_device_id0():
    device_id = os.getenv('DEVICE_ID', '0')
    return int(device_id)


def train(data_file, ckpt_pre_trained, base_lr, stage):
    """train"""

    if args.is_distributed:
        init()
        args.rank = get_rank()
        args.group_size = get_group_size()
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True,
                                          device_num=args.group_size)

    # dataset
    dataset = data_generator.SegDataset(image_mean=args.image_mean,
                                        image_std=args.image_std,
                                        data_file=data_file,
                                        batch_size=args.batch_size,
                                        crop_size=args.crop_size,
                                        max_scale=args.max_scale,
                                        min_scale=args.min_scale,
                                        ignore_label=args.ignore_label,
                                        num_classes=args.num_classes,
                                        num_readers=2,
                                        num_parallel_calls=4,
                                        shard_id=args.rank,
                                        shard_num=args.group_size,
                                        )
    dataset = dataset.get_dataset1()
    network = RefineNet(Bottleneck, [3, 4, 23, 3], args.num_classes)

    # loss
    loss_ = loss.SoftmaxCrossEntropyLoss(args.num_classes, args.ignore_label)
    weights_init(network)
    if ckpt_pre_trained:
        param_dict = load_checkpoint(ckpt_pre_trained)
        load_param_into_net(network, param_dict)

    # optimizer
    iters_per_epoch = dataset.get_dataset_size()
    total_train_steps = iters_per_epoch * args.train_epochs
    if args.lr_type == 'cos':
        lr_iter = learning_rates.cosine_lr(base_lr, total_train_steps, total_train_steps)
    elif args.lr_type == 'poly':
        lr_iter = learning_rates.poly_lr(base_lr, total_train_steps, total_train_steps, end_lr=0.0, power=0.9)
    elif args.lr_type == 'exp':
        lr_iter = learning_rates.exponential_lr(base_lr, args.lr_decay_step, args.lr_decay_rate,
                                                total_train_steps, staircase=True)
    elif args.lr_type == 'cos_warmup':
        lr_iter = learning_rates.warmup_cosine_annealing_lr(base_lr, iters_per_epoch,
                                                            args.warmup_epochs, args.train_epochs)
    else:
        raise ValueError('unknown learning rate type')
    opt = nn.Momentum(params=network.trainable_params(), learning_rate=lr_iter, momentum=0.9, weight_decay=0.0005,
                      loss_scale=args.loss_scale)

    # loss scale
    manager_loss_scale = FixedLossScaleManager(args.loss_scale, drop_overflow_update=False)
    if args.device_target == "CPU":
        amp_level = "O0"
    elif args.device_target == "GPU":
        amp_level = "O2"
    elif args.device_target == "Ascend":
        amp_level = "O3"
    model = Model(network, loss_, optimizer=opt, amp_level=amp_level, loss_scale_manager=manager_loss_scale)

    # callback for saving ckpts
    time_cb = TimeMonitor(data_size=iters_per_epoch)
    loss_cb = LossMonitor()
    cbs = [time_cb, loss_cb]
    if args.rank == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=args.save_epochs*iters_per_epoch,
                                     keep_checkpoint_max=args.keep_checkpoint_max)
        ckpoint_cb = ModelCheckpoint(prefix=args.model, directory="./ckpt_"+str(args.rank), config=config_ck)
        cbs.append(ckpoint_cb)
    model.train(args.train_epochs, dataset, callbacks=cbs, dataset_sink_mode=(args.device_target != "CPU"))
    return iters_per_epoch


def export_models(step):
    print("exporting model....")
    network = RefineNet(Bottleneck, [3, 4, 23, 3], args.num_classes)
    ckptfile = "./ckpt_0/refinenet_1-{}_{}.ckpt".format(args.train_epochs, step)
    param_dict = load_checkpoint(ckptfile)
    load_param_into_net(network, param_dict)
    image_shape = [args.batch_size, 3, args.crop_size, args.crop_size]
    input_data = np.random.uniform(0.0, 1.0, size=image_shape).astype(np.float32)
    export(network, Tensor(input_data), file_name="refinenet.air", file_format="AIR")
    print("export model finished....")


if __name__ == '__main__':
    args = parse_args()
    context.set_context(mode=context.GRAPH_MODE, enable_auto_mixed_precision=True, save_graphs=False,
                        device_target=args.device_target, device_id=int(get_device_id0()))
    iters = 0
    args.data_file = os.path.join(args.data_file, "sbdonly0")
    args.data_file2 = os.path.join(args.data_file2, "mindrecord0")
    args.ckpt_pre_trained = os.path.join(args.ckpt_pre_trained, "resnet-101.ckpt")
    for i in [1, 2]:
        if i == 1:
            lr = 0.0015
            iters = train(args.data_file, args.ckpt_pre_trained, lr, i)
        else:
            steps = int(args.save_epochs * iters / 5)
            ckpt_pretrained = "./ckpt_0/refinenet-{}_{}.ckpt".format(args.train_epochs, steps)
            lr = 0.00015
            iters = train(args.data_file2, ckpt_pretrained, lr, i)

    steps2 = int(args.save_epochs * iters / 5)
    export_models(steps2)
    mox.file.copy_parallel("./", args.output_path)
