# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""train the PDarts model"""
import os
import time
import logging
import argparse

import numpy as np

from mindspore import context
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.callback._time_monitor import TimeMonitor
from mindspore.train.callback._loss_monitor import LossMonitor
from mindspore.train import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication import init
from mindspore.communication.management import get_rank
from mindspore.train.model import ParallelMode

import src.my_utils as my_utils
import src.genotypes as genotypes
from src.loss import SoftmaxCrossEntropyLoss
from src.model import NetworkCIFAR as Network
from src.dataset import create_cifar10_dataset
from src.call_backs import Val_Callback, Set_Attr_CallBack

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU'],
                    help='device where the code will be implemented (default: CPU)')
parser.add_argument('--local_data_root', default='/cache/',
                    help='a directory used for transfer data between local path and OBS path')
parser.add_argument('--data_url', type=str,
                    default="cifar-10-binary", help='the training data path')
parser.add_argument('--train_url', type=str, default="",
                    help='the path to save training outputs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--load_weight', type=str, default='',
                    help='load ckpt file path')
parser.add_argument('--no_top', type=str, default='True',
                    help='whether contains the top fc layer weights')
parser.add_argument('--learning_rate', type=float,
                    default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float,
                    default=3e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=600,
                    help='num of training epochs')
parser.add_argument('--init_channels', type=int,
                    default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20,
                    help='total number of layers')
parser.add_argument('--auxiliary', action='store_true',
                    default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float,
                    default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float,
                    default=0.3, help='drop path probability')
parser.add_argument('--arch', type=str, default='PDARTS',
                    help='which architecture to use')
parser.add_argument('--amp_level', type=str, default='O3', help='')
parser.add_argument('--optimizer', type=str,
                    default='Momentum', help='SGD or Momentum')
parser.add_argument('--cutout_length', default=16, help='use cutout')

args = parser.parse_args()

CIFAR_CLASSES = 10


def cosine_lr(base_lr, decay_steps, total_steps):
    lr_each_step = []
    for i in range(total_steps):
        step_ = min(i, decay_steps)
        new_lr = base_lr * 0.5 * (1 + np.cos(np.pi * step_ / decay_steps))
        lr_each_step.append(new_lr)
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    return lr_each_step


def main():
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=args.device_target)
    context.set_context(enable_graph_kernel=True)
    rank_size = int(os.getenv('RANK_SIZE', '1'))
    rank_id = 0
    if rank_size > 1:
        init()
        rank_id = get_rank()
        context.set_auto_parallel_context(device_num=rank_size,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)

    print(genotypes.Genotype)
    if args.arch == 'PDARTS':
        genotype = genotypes.PDARTS
    print('---------Genotype---------')
    logging.info(genotype)
    print('--------------------------')
    network = Network(args.init_channels, CIFAR_CLASSES,
                      args.layers, args.auxiliary, genotype)
    my_utils.print_trainable_params_count(network)

    if args.load_weight != '' and args.load_weight != 'None' and args.load_weight is not None:
        param_dict = load_checkpoint(args.load_weight)
        if args.no_top == 'True':
            print('remove top fc layer weights...')
            param_dict.pop('auxiliary_head.classifier.weight')
            param_dict.pop('auxiliary_head.classifier.bias')
            param_dict.pop('classifier.weight')
            param_dict.pop('classifier.bias')
            param_dict.pop('moments.auxiliary_head.classifier.weight')
            param_dict.pop('moments.auxiliary_head.classifier.bias')
            param_dict.pop('moments.classifier.weight')
            param_dict.pop('moments.classifier.bias')
        load_param_into_net(network, param_dict)

    if args.data_url.startswith('s3://') or args.data_url.startswith('obs://'):
        from moxing.framework import file
        data_url_cache = os.path.join(args.local_data_root, 'data')
        file.copy_parallel(args.data_url, data_url_cache)
        args.data_url = data_url_cache

    train_path = os.path.join(args.data_url, 'train')
    train_dataset = create_cifar10_dataset(
        train_path, True, batch_size=args.batch_size, shuffle=True, cutout_length=args.cutout_length,
        rank_id=rank_id, rank_size=rank_size)
    val_path = os.path.join(args.data_url, 'val')
    val_dataset = create_cifar10_dataset(
        val_path, False, batch_size=128, shuffle=False, rank_id=rank_id, rank_size=rank_size)

    # learning rate setting
    step_size = train_dataset.get_dataset_size()
    print(f'step_size:{step_size}')
    lr = args.learning_rate
    lr = cosine_lr(lr, args.epochs * step_size, args.epochs * step_size)
    lr = Tensor(lr)

    if args.optimizer == 'SGD':
        net_opt = nn.SGD(
            network.trainable_params(),
            lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
            loss_scale=1024
        )
    elif args.optimizer == 'Momentum':
        net_opt = nn.Momentum(
            network.trainable_params(),
            lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            use_nesterov=True,
            loss_scale=1024
        )

    net_loss = SoftmaxCrossEntropyLoss(args.auxiliary, args.auxiliary_weight)
    loss_scale = FixedLossScaleManager(1024, drop_overflow_update=False)
    model = Model(network, net_loss, net_opt, metrics={'loss', 'top_1_accuracy', 'top_5_accuracy'},
                  amp_level=args.amp_level, loss_scale_manager=loss_scale)

    set_attr_cb = Set_Attr_CallBack(
        network, args.drop_path_prob, args.epochs, args.layers, args.batch_size)

    loss_cb = LossMonitor()
    time_cb = TimeMonitor()
    val_callback = Val_Callback(model, train_dataset, val_dataset, args.train_url,
                                prefix='PDarts', network=network, img_size=32,
                                rank_id=rank_id, is_eval_train_dataset=True)
    callbacks = [loss_cb, time_cb, val_callback, set_attr_cb]

    model.train(args.epochs, train_dataset,
                callbacks=callbacks, dataset_sink_mode=True)


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Total time: %ds.', duration)
