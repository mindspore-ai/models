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
# ============================================================================
"""train tsn"""
import os
import glob
import ast
import argparse
import random
import zipfile
import numpy as np

import moxing as mox
import mindspore.nn as nn
import mindspore.dataset as ds

from mindspore import dtype as mstype
from mindspore.common import set_seed
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.train.callback import LossMonitor
from mindspore.communication.management import init
from mindspore import context, Tensor, load_checkpoint, load_param_into_net, export
from mindspore.train.callback import TimeMonitor, ModelCheckpoint, CheckpointConfig

from src.models import TSN
from src.dataset import create_dataset
from src.tsn_for_train import TrainOneStepCellWithGradClip
from src.util import process_trainable_params, get_lr
from src.transforms import GroupMultiScaleCrop, GroupRandomHorizontalFlip, Stack,\
     ToTorchFormatTensor, GroupNormalize, IdentityTransform

set_seed(1234)
random.seed(1234)
np.random.seed(1234)
ds.config.set_seed(1234)


def filter_checkpoint_parameter_by_list(orgin_dict, param_filter):
    """remove useless parameters according to filter_list"""
    for key in list(orgin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del orgin_dict[key]
                break


def load_tsn_params(filter_weight, pretrained_checkpoint, load_net):
    """Load parameter from checkpoint."""
    param_dict = load_checkpoint(pretrained_checkpoint)
    if filter_weight:
        filter_list = [x.name for x in load_net.base_model.fc.new_fc.get_parameters()]
        filter_checkpoint_parameter_by_list(param_dict, filter_list)
    load_param_into_net(load_net, param_dict)
    return load_net


def export_tsn(checkpoint_path, s_prefix, file_name, file_format, num_classes,\
     modality, arch, consensus_type, input_size):
    """ export_tsn """
    dropout = 0.2
    if modality == 'RGB':
        length = 3
    elif modality == 'Flow':
        length = 10
        dropout = 0.3
    elif modality == 'RGBDiff':
        length = 18
    # load checkpoint
    net_export = TSN(num_classes, 1, modality, base_model=arch,\
         consensus_type=consensus_type, dropout=dropout)
    prob_ckpt_list = os.path.join(checkpoint_path, "{}*.ckpt".format(s_prefix))
    ckpt_list = glob.glob(prob_ckpt_list)
    if not ckpt_list:
        print('Freezing model failed!')
        print("can not find ckpt files. ")
    else:
        ckpt_list.sort(key=os.path.getmtime)
        ckpt_name = ckpt_list[-1]
        print("checkpoint file name", ckpt_name)
        param_dict = load_checkpoint(ckpt_name)
        load_param_into_net(net_export, param_dict)

        input_arr = Tensor(np.zeros([1, length, input_size, input_size]), mstype.float32)

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)
        file_name = os.path.join(checkpoint_path, file_name)
        export(net_export, input_arr, file_name=file_name+modality, file_format=file_format)
        print('Freezing model success!')
    return 0

parser = argparse.ArgumentParser(description="MindSpore implementation of Temporal Segment Networks")
parser.add_argument('--platform', type=str, default='Ascend', choices=['Ascend'],
                    help='Running platform, only support Ascend now. Default is Ascend.')
parser.add_argument('--data_url', type=str, default='')
parser.add_argument('--file_name', type=str, default='')
parser.add_argument('--directory', type=str, default='')
parser.add_argument('--train_url', type=str, default='')
parser.add_argument('--dataset', type=str, default='ucf101', choices=['ucf101', 'hmdb51', 'kinetics'])
parser.add_argument('--modality', type=str, default='Flow', choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('--train_list_path', type=str, default="")
parser.add_argument('--train_list', type=str, default="")
parser.add_argument('--pretrained_path', type=str, default="")
parser.add_argument('--pre_trained_name', type=str, default="")
parser.add_argument('--device_id', default=0, type=int)

# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="BNInception")
parser.add_argument('--num_segments', type=int, default=3)
parser.add_argument('--consensus_type', type=str, default='avg', choices=['avg', 'max', 'topk'])
parser.add_argument('--dropout', default=0.3, type=float, help='dropout ratio (default: 0.5)')
parser.add_argument('--num_class', default=101, type=int, help='number class')
parser.add_argument('--filter_weight', default=False, type=ast.literal_eval, help='filter weight')

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=3, type=int, help='number of total epochs to run')
parser.add_argument('--batch_size', default=16, type=int, help='mini-batch size (default: 256)')
parser.add_argument('--learning_rate', default=0.005, type=float, help='initial learning rate')
parser.add_argument('--gamma', default=0.1, type=float, help='dacay rate of learning rate')
parser.add_argument('--loss_scale', default=1.0, type=float, help='')
parser.add_argument('--lr_steps', default=70, type=int, help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--dataset_sink_mode', default=False, type=ast.literal_eval, help='')

# ========================= Monitor Configs ==========================
parser.add_argument('--workers', default=16, type=int, help='number of data loading workers')
parser.add_argument('--save_check_point', type=ast.literal_eval, default=True)
parser.add_argument('--ckpt_save_dir', type=str, default="./checkpoint")
parser.add_argument('--snapshot_pref', type=str, default="ucf101_bninception_")
parser.add_argument('--flow_prefix', default="flow_", type=str)

# ========================= export model =============================
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="AIR", help="file format")
parser.add_argument('--export_name', default="tsn_", type=str)
parser.add_argument('--test_segments', default=25, type=int)
parser.add_argument('--test_crops', default=10, type=int)
parser.add_argument('--input_size', default=224, type=int)

if __name__ == '__main__':
    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE,\
         device_target=args.platform, save_graphs=False)

    device_id = int(os.getenv('DEVICE_ID'))
    device_num = int(os.getenv('RANK_SIZE'))
    rank = device_id
    context.set_context(device_id=device_id)
    local_data_url = '/cache/data'
    local_train_url = '/cache/train'
    local_list_url = '/cache/list'
    local_pre_train_url = '/cache/resume'

    mox.file.make_dirs(local_data_url)
    mox.file.make_dirs(local_train_url)
    mox.file.make_dirs(local_list_url)
    mox.file.make_dirs(local_pre_train_url)
    if args.train_list_path != args.data_url:
        mox.file.copy_parallel(args.train_list_path, local_list_url)
    else:
        local_list_url = local_data_url
    mox.file.copy_parallel(args.pretrained_path, local_pre_train_url)
    mox.file.copy_parallel(args.data_url, local_data_url)

    data_dir = local_data_url + '/'
    zFile = zipfile.ZipFile(data_dir+args.file_name, "r")
    for fileM in zFile.namelist():
        zFile.extract(fileM, local_data_url)
    zFile.close()

    if device_num > 1:
        init()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num,\
            parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
    data_url = local_data_url + '/' + args.directory + '/'
    pre_trained = local_pre_train_url + '/'
    train_list = local_list_url + '/' + args.train_list

    if args.dataset == 'ucf101':
        num_class = 101
    elif args.dataset == 'hmdb51':
        num_class = 51
    elif args.dataset == 'kinetics':
        num_class = 400

    if args.num_class:
        num_class = args.num_class

    net = TSN(num_class, args.num_segments, args.modality, base_model=args.arch,\
         consensus_type=args.consensus_type, dropout=args.dropout)

    net = load_tsn_params(args.filter_weight, pre_trained+args.pre_trained_name, net)

    crop_size = net.crop_size
    scale_size = net.scale_size
    input_mean = net.input_mean
    input_std = net.input_std

    if args.modality == 'RGB':
        data_length = 1
        scale = [1, .875, .75, .66]
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5
        scale = [1, .875, .75]

    transform = [GroupMultiScaleCrop(crop_size, scale),\
         GroupRandomHorizontalFlip(is_flow=args.modality == 'Flow')]
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    transform += [Stack(roll=args.arch == 'BNInception'),\
         ToTorchFormatTensor(div=args.arch != 'BNInception'), normalize]
    run_distribute = device_num > 1
    image_tmpl = "img_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg"
    train_loader = create_dataset(root_path=data_url, list_file=train_list,\
         batch_size=args.batch_size, num_segments=args.num_segments, new_length=data_length,\
              modality=args.modality, image_tmpl=image_tmpl, transform=transform, test_mode=0,\
                   run_distribute=run_distribute, worker=args.workers,\
                             num_shards=device_num, shard_id=rank)
    data_size = train_loader.get_dataset_size()

    lr = get_lr(learning_rate=args.learning_rate, gamma=args.gamma, epochs=args.epochs,\
         steps_per_epoch=data_size, lr_steps=args.lr_steps)
    base_lr = 5 if args.modality == 'Flow' else 1
    group1, group2, group3, group4, group5 = process_trainable_params(net.trainable_params())
    group_params = [{'params': group1, 'lr': lr*base_lr, 'weight_decay': args.weight_decay},\
         {'params': group2, 'lr': lr*base_lr*2, 'weight_decay': 0},\
              {'params': group3, 'lr': lr*1, 'weight_decay': args.weight_decay},\
                   {'params': group4, 'lr': lr*2, 'weight_decay': 0},\
                        {'params': group5, 'lr': lr*1, 'weight_decay': 0}]

    # define loss function (criterion) and optimizer
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    #define callbacks
    loss_cb = LossMonitor()
    time_cb = TimeMonitor(data_size=data_size)
    callbacks = [loss_cb, time_cb]

    #process net for train
    optimizer = nn.SGD(group_params, momentum=args.momentum, loss_scale=args.loss_scale)
    network = nn.WithLossCell(net, criterion)
    network = TrainOneStepCellWithGradClip(network, optimizer, args.loss_scale)

    model = Model(network)

    if args.save_check_point and (device_num == 1 or device_id == 0):
        config_ck = CheckpointConfig(save_checkpoint_steps=data_size*args.epochs, keep_checkpoint_max=1)
        ckpoint_cb = ModelCheckpoint(prefix=args.snapshot_pref+args.modality,\
                 directory=local_train_url, config=config_ck)
        callbacks += [ckpoint_cb]

    model.train(args.epochs, train_loader, callbacks=callbacks, dataset_sink_mode=args.dataset_sink_mode)

    export_tsn(local_train_url, args.snapshot_pref+args.modality, args.export_name,\
         args.file_format, num_class, args.modality, args.arch, args.consensus_type, args.input_size)
    mox.file.copy_parallel(src_url=local_train_url, dst_url=args.train_url)
