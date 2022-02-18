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
"""train MultiTaskNet"""
import os
import ast
import argparse
import glob
import numpy as np

import mindspore
from mindspore import context, Tensor, export
from mindspore.train.model import Model
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.model.DenseNet import DenseNet121
from src.loss.loss import netWithLossCell
from src.lr_scheduler.lr_scheduler import get_lr
from src.dataset.dataset import create_dataset
from src.optimizers.optimizers import init_optim

parser = argparse.ArgumentParser(description='Train MultiTaskNet')

parser.add_argument('--device_target', type=str, default="Ascend", help='Device target')
parser.add_argument('--pretrained_url', type=str, default='densenet121_pretrain.ckpt')
parser.add_argument('--isModelArts', type=ast.literal_eval, default=True)
parser.add_argument('--train_url', type=str)
parser.add_argument('--data_url', type=str)

# Datasets
parser.add_argument('--root', type=str, default='', help="root path to data directory")
parser.add_argument('-d', '--dataset', type=str, default='veri', help="name of the dataset")
parser.add_argument('--height', type=int, default=256, help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=256, help="width of an image (default: 256)")

# Optimization options
parser.add_argument('--optim', type=str, default='adam', help="optimization algorithm")
parser.add_argument('--max_epoch', default=1, type=int, help="maximum epochs to run")
parser.add_argument('--train_batch', default=32, type=int, help="train batch size")
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float, help="initial learning rate")
parser.add_argument('--stepsize', default=[30, 60], nargs='+', type=int, help="stepsize to decay learning rate")
parser.add_argument('--gamma', default=0.1, type=float, help="learning rate decay")
parser.add_argument('--weight_decay', default=5e-04, type=float, help="weight decay (default: 5e-04)")
parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")
parser.add_argument('--htri_only', type=ast.literal_eval, default=False, help="whether use only htri loss in training")
parser.add_argument('--lambda_xent', type=float, default=1, help="cross entropy loss weight")
parser.add_argument('--lambda_htri', type=float, default=1, help="hard triplet loss weight")
parser.add_argument('--lambda_vcolor', type=float, default=0.125, help="vehicle color classification weight")
parser.add_argument('--lambda_vtype', type=float, default=0.125, help="vehicle type classification weight")
parser.add_argument('--label_smooth', type=ast.literal_eval, default=False, help="label smoothing")

# Architecture
parser.add_argument('--heatmapaware', type=ast.literal_eval, default=False, help="embed heatmaps to images")
parser.add_argument('--segmentaware', type=ast.literal_eval, default=False, help="embed segments to images")

args = parser.parse_args()


def frozen_to_air(frozen_net, frozen_args):
    ''' frozen '''
    if frozen_args.get("ckpt_file") is not None:
        param_dict = load_checkpoint(frozen_args.get("ckpt_file"))
        load_param_into_net(frozen_net, param_dict)
    batch_size = frozen_args.get("batch_size")
    height = frozen_args.get("height")
    width = frozen_args.get("width")
    if frozen_args.get("heatmapaware") and frozen_args.get("segmentaware"):
        input_img = np.zeros((batch_size, 52, height, width))
    elif frozen_args.get("heatmapaware"):
        input_img = np.zeros((batch_size, 39, height, width))
    elif frozen_args.get("segmentaware"):
        input_img = np.zeros((batch_size, 16, height, width))
    else:
        input_img = np.zeros((batch_size, 3, height, width))
    input_img = Tensor(input_img, mindspore.float32)
    input_vkpt = Tensor(np.zeros((batch_size, 108)), mindspore.float32)
    input_arr = (input_img, input_vkpt)
    export(frozen_net, *input_arr, file_name=frozen_args.get("file_name"), file_format=frozen_args.get("file_format"))

if __name__ == '__main__':
    target = args.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
    train_dataset_path = args.data_url

    # dataset
    dataset, num_train_vids, num_train_vcolors, num_train_vtypes = create_dataset(dataset_dir=args.dataset,
                                                                                  root=train_dataset_path,
                                                                                  width=args.width,
                                                                                  height=args.height,
                                                                                  keyptaware=True,
                                                                                  heatmapaware=args.heatmapaware,
                                                                                  segmentaware=args.segmentaware,
                                                                                  train_batch=args.train_batch)

    step_size = dataset.get_dataset_size()

    # model
    net = DenseNet121(pretrain_path=args.pretrained_url,
                      num_vids=num_train_vids,
                      num_vcolors=num_train_vcolors,
                      num_vtypes=num_train_vtypes,
                      keyptaware=True,
                      heatmapaware=args.heatmapaware,
                      segmentaware=args.segmentaware,
                      multitask=True)

    net_with_loss = netWithLossCell(net,
                                    label_smooth=args.label_smooth,
                                    keyptaware=True,
                                    multitask=True,
                                    htri_only=args.htri_only,
                                    lambda_xent=args.lambda_xent,
                                    lambda_htri=args.lambda_htri,
                                    lambda_vcolor=args.lambda_vcolor,
                                    lambda_vtype=args.lambda_vtype,
                                    margin=args.margin,
                                    num_train_vids=num_train_vids,
                                    num_train_vcolors=num_train_vcolors,
                                    num_train_vtypes=num_train_vtypes,
                                    batch_size=args.train_batch)

    lr = get_lr(lr=args.lr,
                total_epochs=args.max_epoch,
                steps_per_epoch=step_size,
                lr_step=args.stepsize,
                gamma=args.gamma)
    lr = Tensor(lr, mindspore.float32)

    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)
    group_params = [{'params': decayed_params, 'weight_decay': args.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]

    manager = FixedLossScaleManager(64, drop_overflow_update=False)
    optimizer = init_optim(args.optim, group_params, lr, args.weight_decay)

    model = Model(net_with_loss, optimizer=optimizer, loss_scale_manager=manager, amp_level="O3")

    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]

    config_ck = CheckpointConfig(save_checkpoint_steps=step_size, keep_checkpoint_max=args.max_epoch)

    save_checkpoint_path = args.train_url

    ckpt_cb = ModelCheckpoint(prefix="MultipleNet",
                              directory=save_checkpoint_path,
                              config=config_ck)
    cb += [ckpt_cb]

    print("##################### heatmapaware is #####################:{}".format(args.heatmapaware))
    print("##################### segmentaware is #####################:{}".format(args.segmentaware))
    print("##################### batch size is #####################:{}".format(args.train_batch))
    print("######################## start train ########################")

    model.train(args.max_epoch, dataset, callbacks=cb, dataset_sink_mode=True)
    ckpt_list = glob.glob(args.train_url + "/*.ckpt")
    if not ckpt_list:
        print("ckpt file not generated.")

    ckpt_list.sort(key=os.path.getmtime)
    ckpt_model = ckpt_list[-1]
    print("checkpoint path", ckpt_model)
    frozen_to_air_args = {'ckpt_file': None,
                          'batch_size': args.train_batch,
                          'height': args.height,
                          'width': args.width,
                          'heatmapaware': args.heatmapaware,
                          'segmentaware': args.segmentaware,
                          'file_name': args.train_url + "/MultiTaskNet",
                          'file_format': 'AIR'}
    frozen_to_air(net, frozen_to_air_args)
