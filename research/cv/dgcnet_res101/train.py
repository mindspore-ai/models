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
"""DGCNet(res101) train."""
import argparse
import os.path as osp
import timeit
import numpy as np

import mindspore
import mindspore.nn as nn
from mindspore.communication import init, get_rank, get_group_size
from mindspore import load_checkpoint, load_param_into_net
from mindspore import context
from mindspore.common import set_seed
from mindspore.context import ParallelMode

from src.cityscapes import create_dataset
from src.loss import CriterionOhemDSN

set_seed(1)

def str2bool(v):
    """str2bool"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        result = True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        result = False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    return result


def get_arguments():
    """
    Parse all the arguments
    Returns: args
    A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DGCNet-ResNet101 Network")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data_dir", type=str, default="./dataset",
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data_list", type=str, default="./src/data/cityscapes/train.txt",
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--data_set", type=str, default="cityscapes", help="dataset to train")
    parser.add_argument("--ignore_label", type=int, default=255,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input_size", type=int, default=832,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning_rate", type=float, default=1e-2,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num_classes", type=int, default=19,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num_steps", type=int, default=60000,
                        help="Number of training steps.")
    parser.add_argument("--multiple", type=int, default=8,
                        help="Multiple number of training steps.")
    parser.add_argument("--power", type=float, default=0.9,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--rgb", type=int, default=1)

    # ***** Params for save and load ******
    parser.add_argument("--restore_from", type=str, default=None,
                        help="Where restore models parameters from.")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Where to save snapshots of the models.")

    # **** Params for OHEM Loss**** #
    parser.add_argument("--ohem_thres", type=float, default=0.7,
                        help="choose the samples with correct probability underthe threshold.")
    parser.add_argument("--ohem_keep", type=int, default=100000,
                        help="choose the samples with correct probability underthe threshold.")

    # ***** Params for Distributed Traning ***** #
    parser.add_argument('--run_distribute', type=int, default=0, help='Run distribute')
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
    parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
    args = parser.parse_args()
    return args


def train():
    """Train start"""
    args = get_arguments()

    if args.run_distribute:
        init("nccl")
        target = 'GPU'
        device_num = get_group_size()
        device_id = get_rank()
        print("rank_id is {}, device_num is {}".format(device_id, device_num))
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
        context.set_context(mode=context.GRAPH_MODE, device_target=target, device_id=device_id)
    else:
        target = 'GPU'
        device_num = args.device_num
        device_id = args.device_id
        print("rank_id is {}, device_num is {}".format(device_id, device_num))
        context.set_context(mode=context.GRAPH_MODE, device_target=target, device_id=device_id)

    # RGB input
    h, w = args.input_size, args.input_size
    input_size = (h, w)
    IMG_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
    IMG_VARS = np.array((0.229, 0.224, 0.225), dtype=np.float32)

    # set max_iters
    max_iters = args.num_steps

    # set data loader
    train_ds = create_dataset(args, crop_size=input_size, max_iters=int(max_iters), mean=IMG_MEAN, vari=IMG_VARS,
                              scale=True, mirror=True, device_num=device_num, device_id=device_id)
    train_data_loader = train_ds.create_dict_iterator()
    print("Create train dataset done!")
    net_with_loss = CriterionOhemDSN(args)

    # set optimizer
    polynomial_decay_lr = nn.PolynomialDecayLR(learning_rate=args.learning_rate, end_learning_rate=0.0,
                                               decay_steps=train_ds.get_dataset_size(), power=args.power)
    optim = nn.SGD(params=net_with_loss.trainable_params(), learning_rate=polynomial_decay_lr, momentum=args.momentum,
                   weight_decay=args.weight_decay)

    # load pretrain params
    if args.restore_from is not None:
        saved_state_dict = load_checkpoint(args.restore_from)
        print("load pretrined models")
        load_param_into_net(net_with_loss, saved_state_dict, strict_load=False)
    else:
        print("train from stracth")

    manager = nn.DynamicLossScaleUpdateCell(loss_scale_value=2 ** 12, scale_factor=2, scale_window=1000)
    train_net = nn.TrainOneStepWithLossScaleCell(net_with_loss, optim, scale_sense=manager)
    train_net.set_train()
    start_train = timeit.default_timer()

    for i_iter, item in enumerate(train_data_loader):
        start_perstep = timeit.default_timer()
        image = item["image"]
        label = item["label"].astype("int64")
        iter_loss, _, _ = train_net(image, label)
        end_perstep = timeit.default_timer()
        print('iter = {} of {} completed, loss = {}, lr={}, time={}seconds'.format(i_iter,
                                                                                   train_ds.get_dataset_size(),
                                                                                   iter_loss,
                                                                                   polynomial_decay_lr(i_iter),
                                                                                   end_perstep - start_perstep))

    end_train = timeit.default_timer()
    if device_id == 0:
        print('save final checkpoint ...')
        mindspore.save_checkpoint(train_net, osp.join(args.save_dir, 'dgcnet_res101_final.ckpt'))
        print("Training cost: " + str(end_train - start_train) + 'seconds')


if __name__ == '__main__':
    train()
