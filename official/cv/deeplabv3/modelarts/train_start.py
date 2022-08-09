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
"""train deeplabv3."""

import os
import glob
import argparse
import moxing as mox
import numpy as np
from mindspore import context, export, Tensor
from mindspore.train.model import Model
from mindspore.context import ParallelMode
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.common import set_seed
from src.data import dataset as data_generator
from src.loss import loss
from src.nets import net_factory
from src.utils import learning_rates

set_seed(1)


_CACHE_DATA_URL = "/cache/data_url"
_CACHE_TRAIN_URL = "/cache/train_url"


class BuildTrainNetwork(nn.Cell):
    def __init__(self, network, criterion):
        super(BuildTrainNetwork, self).__init__()
        self.network = network
        self.criterion = criterion

    def construct(self, input_data, label):
        output = self.network(input_data)
        net_loss = self.criterion(output, label)
        return net_loss


class BuildEvalNetwork(nn.Cell):
    def __init__(self, net, input_format="NCHW"):
        super(BuildEvalNetwork, self).__init__()
        self.network = net
        self.softmax = nn.Softmax(axis=1)
        self.transpose = ops.Transpose()
        self.format = input_format

    def construct(self, x):
        if self.format == "NHWC":
            x = self.transpose(x, (0, 3, 1, 2))
        output = self.network(x)
        output = self.softmax(output)
        return output


def _parse_args():
    parser = argparse.ArgumentParser('mindspore deeplabv3 training')
    # dataset
    parser.add_argument('--train_url', type=str, default='',
                        help='where training log and ckpts saved')
    parser.add_argument('--data_url', type=str, default='',
                        help='path of dataset')
    parser.add_argument('--file_name', type=str, default='vocaug_mindrecord0',
                        help='mindrecord file name of dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--crop_size', type=int, default=513, help='crop size')
    parser.add_argument('--min_scale', type=float, default=0.5,
                        help='minimum scale of data argumentation')
    parser.add_argument('--max_scale', type=float, default=2.0,
                        help='maximum scale of data argumentation')
    parser.add_argument('--ignore_label', type=int, default=255,
                        help='ignore label')
    parser.add_argument('--num_classes', type=int, default=21,
                        help='number of classes')
    parser.add_argument('--image_mean', type=int, default=(103.53, 116.28, 123.675),
                        help='image mean')
    parser.add_argument('--image_std', type=int, default=(57.375, 57.120, 58.395),
                        help='image std')


    # optimizer
    parser.add_argument('--train_epochs', type=int, default=300, help='epoch')
    parser.add_argument('--lr_type', type=str, default='cos',
                        help='type of learning rate')
    parser.add_argument('--base_lr', type=float, default=0.015,
                        help='base learning rate')
    parser.add_argument('--lr_decay_step', type=int, default=40000,
                        help='learning rate decay step')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='learning rate decay rate')
    parser.add_argument('--loss_scale', type=float, default=3072.0,
                        help='loss scale')

    # model
    parser.add_argument('--model', type=str, default='deeplab_v3_s16',
                        help='select model')
    parser.add_argument('--export_model', type=str, default='deeplab_v3_s16',
                        help='choices in [deeplab_v3_s16, deeplab_v3_s8]')
    parser.add_argument('--freeze_bn', action='store_false', help='freeze bn')
    parser.add_argument('--ckpt_pre_trained', type=str, default='',
                        help='pretrained model')
    parser.add_argument('--input_format', type=str, default='NCHW',
                        help='NCHW or NHWC')
    parser.add_argument('--export_batch_size', type=int, default=1,
                        help='batch size for export')
    parser.add_argument('--input_size', type=int, default=513,
                        help='input size')
    parser.add_argument('--export_name', type=str, default='deeplabv3',
                        help='output file name')
    parser.add_argument('--file_format', type=str, default='AIR',
                        help='file format, choices in [AIR, MINDIR]')

    # train
    parser.add_argument('--device_target', type=str, default='Ascend',
                        choices=['Ascend', 'CPU'],
                        help='device where the code will be implemented. '
                             '(Default: Ascend)')
    parser.add_argument('--is_distributed', action='store_false',
                        help='distributed training')
    parser.add_argument('--rank', type=int, default=0,
                        help='local rank of distributed')
    parser.add_argument('--group_size', type=int, default=1,
                        help='world size of distributed')
    parser.add_argument('--save_steps', type=int, default=1500,
                        help='steps interval for saving')
    parser.add_argument('--keep_checkpoint_max', type=int, default=200,
                        help='max checkpoint for saving')
    parser.add_argument('--filter_weight', type=str, default="",
                        help="filter weight")

    args, _ = parser.parse_known_args()
    return args


def get_device_id():
    device_id = os.getenv('DEVICE_ID', '0')
    return int(device_id)


def train(args, train_url, data_file, ckpt_pre_trained):
    if args.device_target == "CPU":
        context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="CPU")
    else:
        context.set_context(mode=context.GRAPH_MODE, save_graphs=False,
                            device_target="Ascend", device_id=get_device_id())

    # init multicards training
    if args.is_distributed:
        init()
        args.rank = get_rank()
        args.group_size = get_group_size()

        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=args.group_size)

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
                                        shard_num=args.group_size)
    dataset = dataset.get_dataset(repeat=1)

    # network
    if args.model == 'deeplab_v3_s16':
        network = net_factory.nets_map[args.model]('train', args.num_classes, 16, args.freeze_bn)
    elif args.model == 'deeplab_v3_s8':
        network = net_factory.nets_map[args.model]('train', args.num_classes, 8, args.freeze_bn)
    else:
        raise NotImplementedError('model [{:s}] not recognized'.format(args.model))

    # loss
    loss_ = loss.SoftmaxCrossEntropyLoss(args.num_classes, args.ignore_label)
    loss_.add_flags_recursive(fp32=True)
    train_net = BuildTrainNetwork(network, loss_)

    # load pretrained model
    if args.ckpt_pre_trained:
        param_dict = load_checkpoint(ckpt_pre_trained)
        if args.filter_weight:
            filter_list = ["network.aspp.conv2.weight", "network.aspp.conv2.bias"]
            for key in list(param_dict.keys()):
                for filter_key in filter_list:
                    if filter_key not in key:
                        continue
                    print('filter {}'.format(key))
                    del param_dict[key]
            load_param_into_net(train_net, param_dict)
            print('load_model {} success'.format(args.ckpt_pre_trained))
        else:
            trans_param_dict = {}
            for key, val in param_dict.items():
                key = key.replace("down_sample_layer", "downsample")
                trans_param_dict[f"network.resnet.{key}"] = val
            load_param_into_net(train_net, trans_param_dict)
            print('load_model {} success'.format(args.ckpt_pre_trained))

    # optimizer
    iters_per_epoch = dataset.get_dataset_size()
    total_train_steps = iters_per_epoch * args.train_epochs
    if args.lr_type == 'cos':
        lr_iter = learning_rates.cosine_lr(args.base_lr, total_train_steps, total_train_steps)
    elif args.lr_type == 'poly':
        lr_iter = learning_rates.poly_lr(args.base_lr, total_train_steps, total_train_steps, end_lr=0.0, power=0.9)
    elif args.lr_type == 'exp':
        lr_iter = learning_rates.exponential_lr(args.base_lr, args.lr_decay_step, args.lr_decay_rate,
                                                total_train_steps, staircase=True)
    else:
        raise ValueError('unknown learning rate type')
    opt = nn.Momentum(params=train_net.trainable_params(), learning_rate=lr_iter, momentum=0.9, weight_decay=0.0001,
                      loss_scale=args.loss_scale)

    # loss scale
    manager_loss_scale = FixedLossScaleManager(args.loss_scale, drop_overflow_update=False)
    amp_level = "O0" if args.device_target == "CPU" else "O3"
    model = Model(train_net, optimizer=opt, amp_level=amp_level, loss_scale_manager=manager_loss_scale)

    # callback for saving ckpts
    time_cb = TimeMonitor(data_size=iters_per_epoch)
    loss_cb = LossMonitor()
    cbs = [time_cb, loss_cb]

    if args.rank == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=args.save_steps,
                                     keep_checkpoint_max=args.keep_checkpoint_max)
        ckpoint_cb = ModelCheckpoint(prefix=args.model, directory=train_url, config=config_ck)
        cbs.append(ckpoint_cb)

    model.train(args.train_epochs, dataset, callbacks=cbs, dataset_sink_mode=(args.device_target != "CPU"))


def export_air(args, train_url):
    '''run export.'''
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    ckpt_list = glob.glob(train_url + "/*.ckpt")
    ckpt_list.sort(key=os.path.getmtime)
    ckpt_model = ckpt_list[-1]

    if args.export_model == 'deeplab_v3_s16':
        network = net_factory.nets_map['deeplab_v3_s16']('eval', args.num_classes, 16, True)
    else:
        network = net_factory.nets_map['deeplab_v3_s8']('eval', args.num_classes, 8, True)
    network = BuildEvalNetwork(network, args.input_format)
    param_dict = load_checkpoint(ckpt_model)

    # load the parameter into net
    load_param_into_net(network, param_dict)
    if args.input_format == "NHWC":
        input_data = Tensor(
            np.ones([args.export_batch_size, args.input_size, args.input_size, 3]).astype(np.float32))
    else:
        input_data = Tensor(
            np.ones([args.export_batch_size, 3, args.input_size, args.input_size]).astype(np.float32))
    export(network, input_data, file_name=args.export_name, file_format=args.file_format)
    export_file = args.export_name+"."+args.file_format.lower()
    mox.file.copy(export_file, os.path.join(train_url, export_file))


def main():
    args = _parse_args()
    os.makedirs(_CACHE_TRAIN_URL, exist_ok=True)
    os.makedirs(_CACHE_DATA_URL, exist_ok=True)
    mox.file.copy_parallel(args.data_url, _CACHE_DATA_URL)
    train_url = _CACHE_TRAIN_URL
    data_url = _CACHE_DATA_URL
    ckpt_pre_trained = os.path.join(_CACHE_DATA_URL,
                                    args.ckpt_pre_trained) \
        if args.ckpt_pre_trained else ""
    data_file = os.path.join(data_url, args.file_name)
    train(args, train_url, data_file, ckpt_pre_trained)
    export_air(args, train_url)
    mox.file.copy_parallel(_CACHE_TRAIN_URL, args.train_url)


if __name__ == '__main__':
    main()
