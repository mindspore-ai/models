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

# from __future__ import absolute_imports
import sys
import time
import datetime
import argparse
import os
import os.path as osp
import numpy as np

import mindspore
from mindspore.common import set_seed
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor, LossMonitor
from mindspore.train import Model
from mindspore import context
from mindspore import Tensor
from mindspore.nn import TrainOneStepCell
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication import init
from mindspore.communication import get_rank, get_group_size
from mindspore.context import ParallelMode
# from mindspore import export
import moxing as mox

from src.losses2 import AllLoss, CustomWithLossCell
from src.utils import Logger
from src import init_model
from src.dataset_loader import create_train_dataset
from src.lr_scheduler import get_lr

parser = argparse.ArgumentParser(description='Train AlignedReID with cross entropy loss and triplet hard loss')

parser.add_argument('--data_url', type=str, default='/opt_data/lh', help="root path to data directory")
parser.add_argument('--dataset', type=str, default='market1501', help="market1501 or other")
parser.add_argument('--train_url', type=str, default='output')
parser.add_argument('--pre_trained', type=str,
                    default='resnet50_ascend_v130_imagenet2012_official_cv_bs256_top1acc76.97__top5acc_93.44.ckpt')

parser.add_argument('--lr_init', type=float, default=0, help="the init learning rate")
parser.add_argument('--optim', type=str, default='momentum', help="momentum or adam")
parser.add_argument('--lr_decay_mode', type=str, default='cosine', help="steps or cosine or poly")
parser.add_argument('--lr_max', type=float, default=1e-2, help="the max learning rate")
parser.add_argument('--warmup_epochs', type=int, default=20, help="the warm up epochs")
parser.add_argument('--max_epoch', default=2, type=int, help="maximum epochs to run")
parser.add_argument('--class_num', type=int, default=751, help="the class num of dataset")
parser.add_argument('--num_instances', type=int, default=4)
parser.add_argument('--workers', default=4, type=int, help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=256, help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128, help="width of an image (default: 128)")
parser.add_argument('--filter_weight', type=lambda x: x.lower() == 'true', default=True, help="filter weight")
parser.add_argument('--labelsmooth', action='store_false', help="label smooth")
parser.add_argument('--weight-decay', default=5e-04, type=float, help="weight decay (default: 5e-04)")
parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")
parser.add_argument('--htri_only', action='store_true', default=False,
                    help="if this is True, only htri loss is used in training")
parser.add_argument('--arch', type=str, default='resnet50')
parser.add_argument('--unaligned', action='store_true', help='test local feature with unalignment')

parser.add_argument("--file_name", type=str, default="AlignedReID", help="ckpt2air output file name.")
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="AIR", help="file format")
# parser.add_argument("--convert_ckpt_file", type=str, default='11-25thebest/resnet50-300_93.ckpt',help="teh ckpt2air Checkpoint file path.")

args = parser.parse_args()
set_seed(1)

device_id = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)
context.set_context(device_id=device_id)
init()

context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                  gradients_mean=True)  # ParallelMode.STAND_ALONE, AUTO_PARALLEL, DATA_PARALLEL

rank_id = get_rank()
rank_size = get_group_size()


def get_last_ckpt(ckpt_dir):
    ckpt_files = [ckpt_file for ckpt_file in os.listdir(ckpt_dir)
                  if ckpt_file.endswith('.ckpt')]
    if not ckpt_files:
        print("No ckpt file found.")
        return None
    return os.path.join(ckpt_dir, sorted(ckpt_files)[-1])


if __name__ == '__main__':

    start_time = time.time()
    sys.stdout = Logger(osp.join(args.train_url, 'log_train.txt'))
    print("==========\nArgs:{}\n==========".format(args))
    print("Initializing dataset {}".format(args.dataset))

    real_path_data = '/cache/datapath' + str(rank_id) + '/'
    os.system("rm -rf {0}".format(real_path_data))
    os.system("mkdir {0}".format(real_path_data))

    mox.file.copy_parallel(args.data_url, real_path_data)
    print("training data finish copy to %s." % real_path_data)

    trainloader, num_train_pids = create_train_dataset(real_path_data, args, rank_id, rank_size)
    steps_per_epoch = trainloader.get_dataset_size()

    net = init_model(name=args.arch, num_classes=num_train_pids, loss='softmax and metric', aligned=True, is_train=True)
    lr = get_lr(lr_init=0, lr_end=3.5e-6, lr_max=args.lr_max, warmup_epochs=args.warmup_epochs,
                total_epochs=args.max_epoch, steps_per_epoch=steps_per_epoch, lr_decay_mode=args.lr_decay_mode)
    lr = Tensor(lr)
    lossfunction = AllLoss(num_classes=num_train_pids, margin=args.margin, labelsmoth=args.labelsmooth)
    if args.optim == 'momentum':
        optimizer = mindspore.nn.Momentum(params=net.trainable_params(), learning_rate=lr, momentum=0.9,
                                          weight_decay=args.weight_decay, loss_scale=1.0, use_nesterov=True)
    elif args.optim == 'adam':
        optimizer = mindspore.nn.Adam(params=net.trainable_params(), learning_rate=lr, weight_decay=args.weight_decay)
    else:
        print("NO optimizer")

    time_cb = TimeMonitor(data_size=steps_per_epoch * 10)
    loss_cb = LossMonitor(per_print_times=1)
    config_ck = CheckpointConfig(save_checkpoint_steps=steps_per_epoch * 1, keep_checkpoint_max=2)
    ckpoint_cb = ModelCheckpoint(prefix='lhresnet50', directory=args.train_url, config=config_ck)
    cb = [ckpoint_cb, loss_cb]

    loss_net = CustomWithLossCell(net, lossfunction)
    grad_net = TrainOneStepCell(loss_net, optimizer)

    load_path = args.pre_trained
    if load_path != "":
        param_dict = load_checkpoint(load_path)
        # mindspore model
        if 'ascend' in load_path:
            if args.filter_weight:
                for oldkey in list(param_dict.keys()):
                    if oldkey.startswith(('global_step', 'learning_rate', 'moments', 'momentum', 'step')):
                        data = param_dict.pop(oldkey)
                    if not oldkey.startswith(('global_step', 'learning_rate', 'moments', 'momentum', 'step')):
                        data = param_dict.pop(oldkey)
                        newkey = '_backbone.base.' + oldkey
                        param_dict[newkey] = data
                        oldkey = newkey
                    if oldkey in ('_backbone.base.end_point.weight', '_backbone.base.end_point.bias'):
                        data = param_dict.pop(oldkey)
        # self model
        else:
            if args.filter_weight:
                for oldkey in list(param_dict.keys()):
                    if oldkey.startswith(('global_step', 'learning_rate', 'moments', 'momentum', 'step')):
                        data = param_dict.pop(oldkey)
                    if oldkey in ('_backbone.end_point.weight', '_backbone.end_point.bias'):
                        data = param_dict.pop(oldkey)
        load_param_into_net(net=grad_net, parameter_dict=param_dict)

    model = Model(grad_net)
    model.train(args.max_epoch, trainloader, callbacks=cb, dataset_sink_mode=False)

    train_time = time.time() - start_time
    total_time = str(datetime.timedelta(seconds=train_time))
    print("total_time", total_time)

    convert_net = init_model(name=args.arch, num_classes=num_train_pids, loss='softmax and metric', aligned=True,
                             is_train=True)
    convert_net.set_train(False)
    ckpt_file = get_last_ckpt(args.train_url)
    param_dict = load_checkpoint(ckpt_file)
    load_param_into_net(convert_net, param_dict)
    input_arr = Tensor(np.zeros([1, 3, args.height, args.width], np.float32))
    airfilename = os.path.join(args.train_url, args.file_name)
    print("export success")
