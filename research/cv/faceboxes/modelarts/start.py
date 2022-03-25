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
"""Train FaceBoxes."""
from __future__ import print_function
import os
import math
import argparse
import glob
import numpy as np
import mindspore
from mindspore import context, Tensor, export
from mindspore.context import ParallelMode
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.communication.management import init, get_rank
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.config import faceboxes_config
from src.network import FaceBoxes, FaceBoxesWithLossCell, TrainingWrapper
from src.loss import MultiBoxLoss
from src.dataset import create_dataset
from src.lr_schedule import adjust_learning_rate
from src.utils import prior_box

import moxing

parser = argparse.ArgumentParser(description='FaceBoxes: Face Detection')
parser.add_argument('--resume', type=str, default=None, help='resume training')
parser.add_argument('--device_target', type=str, default="Ascend", help='run device_target')
parser.add_argument('--batch_size', type=int, default=8, help='size of batch')
parser.add_argument('--max_epoch', type=int, default=300, help='maximum of epoch')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--initial_lr', type=float, default=0.001)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--negative_ratio', type=int, default=7)
parser.add_argument('--decay1', type=int, default=200)
parser.add_argument('--decay2', type=int, default=250)

parser.add_argument('--data_url', default=None, help='Location of data.')
parser.add_argument('--train_url', default='', help='Location of training outputs.')

args_opt = parser.parse_args()

if __name__ == '__main__':
    moxing.file.copy_parallel(src_url=args_opt.data_url, dst_url='/cache/data')
    config = faceboxes_config
    mindspore.common.seed.set_seed(config['seed'])
    print('train config:\n', config)

    # set context and device init
    if args_opt.device_target == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', device_id=config['device_id'],
                            save_graphs=False)
        if int(os.getenv('RANK_SIZE', '1')) > 1:
            context.set_auto_parallel_context(device_num=config['rank_size'], parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()
    else:
        raise ValueError("Unsupported device_target.")

    # set parameters
    batch_size = args_opt.batch_size
    max_epoch = args_opt.max_epoch
    momentum = args_opt.momentum
    weight_decay = args_opt.weight_decay
    initial_lr = args_opt.initial_lr
    gamma = args_opt.gamma
    num_classes = args_opt.num_classes
    negative_ratio = args_opt.negative_ratio
    stepvalues = (args_opt.decay1, args_opt.decay2)

    ds_train = create_dataset('/cache/data', config, batch_size, multiprocessing=True,
                              num_worker=config["num_worker"])
    print('dataset size is : \n', ds_train.get_dataset_size())

    steps_per_epoch = math.ceil(ds_train.get_dataset_size())

    # define loss
    anchors_num = prior_box(config['image_size'], config['min_sizes'], config['steps'], config['clip']).shape[0]
    multibox_loss = MultiBoxLoss(num_classes, anchors_num, negative_ratio, config['batch_size'])

    # define net
    net = FaceBoxes(phase='train')
    net.set_train(True)
    # resume
    if args_opt.resume:
        param_dict = load_checkpoint(args_opt.resume)
        load_param_into_net(net, param_dict)
    net = FaceBoxesWithLossCell(net, multibox_loss, config)

    # define optimizer
    lr = adjust_learning_rate(initial_lr, gamma, stepvalues, steps_per_epoch, max_epoch,
                              warmup_epoch=config['warmup_epoch'])
    opt = mindspore.nn.SGD(params=net.trainable_params(), learning_rate=lr, momentum=momentum,
                           weight_decay=weight_decay, loss_scale=1)

    # define model
    net = TrainingWrapper(net, opt)
    model = Model(net)

    # save model
    rank = 0
    if int(os.getenv('RANK_SIZE', '1')) > 1:
        rank = get_rank()
    ckpt_save_dir = "/cache/train/checkpoint"
    config_ck = CheckpointConfig(save_checkpoint_steps=config['save_checkpoint_epochs'],
                                 keep_checkpoint_max=config['keep_checkpoint_max'])
    ckpt_cb = ModelCheckpoint(prefix="FaceBoxes", directory=ckpt_save_dir, config=config_ck)

    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
    callback_list = [LossMonitor(), time_cb, ckpt_cb]

    # training
    print("============== Starting Training ==============")
    model.train(max_epoch, ds_train, callbacks=callback_list, dataset_sink_mode=True)
    print("============== End Training ==============")

    cfg = None
    if args_opt.device_target == "Ascend":
        cfg = faceboxes_config
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    else:
        raise ValueError("Unsupported device_target.")

    net_t = FaceBoxes(phase='test')

    ckpt_pattern = os.path.join(ckpt_save_dir, '*.ckpt')
    ckpt_list = glob.glob(ckpt_pattern)
    if not ckpt_list:
        print(f"Cant't found ckpt in {ckpt_save_dir}")
        exit()
    ckpt_list.sort(key=os.path.getmtime)
    print("====================%s" % ckpt_list[-1])

    param_dict = load_checkpoint(os.path.join(ckpt_list[-1]))
    load_param_into_net(net_t, param_dict)
    input_shp = [1, 3, 2496, 1056]
    input_array = Tensor(np.random.uniform(-1.0, 1.0, size=input_shp).astype(np.float32))
    export(net_t, input_array, file_name="/cache/train/checkpoint/" + 'FaceBoxes', file_format='AIR')
    moxing.file.copy_parallel(src_url='/cache/train/', dst_url=args_opt.train_url)
