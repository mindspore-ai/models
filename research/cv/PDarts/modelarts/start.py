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
try:
    from moxing.framework import file
    print("import moxing success")
except ModuleNotFoundError as e:
    print(f'not modelarts env, error={e}')

import os
import time
import logging
import argparse
import glob
import datetime

import moxing as mox
import numpy as np
import mindspore
from mindspore import context
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore import export
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.callback._time_monitor import TimeMonitor
from mindspore.train.callback._loss_monitor import LossMonitor
from mindspore.train import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication import init
from mindspore.train.model import ParallelMode

import src.my_utils as my_utils
import src.genotypes as genotypes
from src.loss import SoftmaxCrossEntropyLoss
from src.model import NetworkCIFAR as Network
from src.dataset import create_cifar10_dataset
from src.call_backs import Val_Callback, Set_Attr_CallBack

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend'],
                    help='device where the code will be implemented (default: CPU)')
parser.add_argument('--local_data_root', default='/cache/',
                    help='a directory used for transfer data between local path and OBS path')
parser.add_argument("--modelarts_result_dir", type=str, default="/cache/save_model/")
parser.add_argument('--data_url', type=str,
                    default="cifar-10-binary", help='the training data path')
parser.add_argument('--train_url', type=str, default="./output",
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
parser.add_argument("--modelarts_ckpt_dir", type=str, default="/cache/ckpt")

args = parser.parse_args()

CIFAR_CLASSES = 10

def modelarts_result2obs(FLAGS):
    """
    Copy debug data from modelarts to obs.
    According to the switch flags, the debug data may contains auto tune repository,
    dump data for precision comparison, even the computation graph and profiling data.
    """
    mox.file.copy_parallel(src_url=FLAGS.local_data_root, dst_url=FLAGS.train_url)
    print("===>>>Copy Event or Checkpoint from modelarts dir:{} to obs:{}".format(FLAGS.local_data_root,
                                                                                  FLAGS.train_url))
    files = os.listdir()
    print("===>>>current Files:", files)
    mox.file.copy(src_url='pdarts.air', dst_url=FLAGS.train_url + '/pdarts.air')


def export_AIR(opt):
    """start modelarts export"""
    ckpt_list = glob.glob(opt.train_url + "/*" + "/checkpoint" + "/model*.ckpt")
    if not ckpt_list:
        print("ckpt file not generated.")

    ckpt_list.sort(key=os.path.getmtime)
    ckpt_model = ckpt_list[-1]
    print("checkpoint path", ckpt_model)
    if opt.arch == 'PDARTS':
        genotype = genotypes.PDARTS
    print('---------Genotype---------')
    print(genotype)
    print('--------------------------')
    network = Network(opt.init_channels, CIFAR_CLASSES,
                      opt.layers, opt.auxiliary, genotype)
    network.training = False
    network.drop_path_prob = opt.drop_path_prob * 300 / opt.epochs
    keep_prob = 1. - network.drop_path_prob
    epoch_mask = []
    for dummy_i in range(opt.layers):
        layer_mask = []
        for dummy_j in range(5 * 2):
            mask = np.array([np.random.binomial(1, p=keep_prob)
                             for k in range(opt.batch_size)])
            mask = mask[:, np.newaxis, np.newaxis, np.newaxis]
            mask = Tensor(mask, mindspore.float16)
            layer_mask.append(mask)
        epoch_mask.append(layer_mask)
    network.epoch_mask = epoch_mask
    param_dict = load_checkpoint(ckpt_model)
    load_param_into_net(network, param_dict)
    input_arr = Tensor(np.zeros([1, 3, 32, 32], np.float32))
    export(network, input_arr, file_name="pdarts", file_format='AIR')
    print(os.path.dirname("pdarts.air"))


def cosine_lr(base_lr, decay_steps, total_steps):
    lr_each_step = []
    for i in range(total_steps):
        step_ = min(i, decay_steps)
        new_lr = base_lr * 0.5 * (1 + np.cos(np.pi * step_ / decay_steps))
        lr_each_step.append(new_lr)
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    return lr_each_step


def main():
    device_id = int(os.getenv('DEVICE_ID'))
    device_num = int(os.getenv('RANK_SIZE'))
    print(f'device_id:{device_id}')
    print(f'device_num:{device_num}')
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=args.device_target)
    context.set_context(device_id=device_id)
    context.set_context(enable_graph_kernel=True)
    if device_num > 1:
        init()
        context.set_auto_parallel_context(device_num=device_num,
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
        data_url_cache = os.path.join(args.local_data_root, 'data')
        file.copy_parallel(args.data_url, data_url_cache)
        args.data_url = data_url_cache

    train_path = os.path.join(args.data_url, 'train')
    train_dataset = create_cifar10_dataset(
        train_path, True, batch_size=args.batch_size, shuffle=True, cutout_length=args.cutout_length)
    val_path = os.path.join(args.data_url, 'val')
    val_dataset = create_cifar10_dataset(
        val_path, False, batch_size=128, shuffle=False)

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
    if device_num == 1 or device_id == 0:
        loss_cb = LossMonitor()
        time_cb = TimeMonitor()
        val_callback = Val_Callback(model, train_dataset, val_dataset, args.train_url,
                                    prefix='PDarts', network=network, img_size=32, is_eval_train_dataset=True)
        callbacks = [loss_cb, time_cb, val_callback, set_attr_cb]
    else:
        callbacks = [set_attr_cb]
    model.train(args.epochs, train_dataset,
                callbacks=callbacks, dataset_sink_mode=True)

    # start export air
    if device_id == 0:
        print("start to export air model")
        start = datetime.datetime.now()
        export_AIR(args)
        end = datetime.datetime.now()
        print("===>>end up exporting air model, time use:{}(s)".format((end - start).seconds))

    # copy result from modelarts to obs
    modelarts_result2obs(args)


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Total time: %ds.', duration)
