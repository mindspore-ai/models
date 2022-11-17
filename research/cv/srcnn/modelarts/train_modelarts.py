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
"""srcnn training"""

import os
import ast
import argparse
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import context, Tensor, export
from mindspore.common import set_seed
from mindspore.train.model import Model
from mindspore.train.callback import TimeMonitor, LossMonitor, CheckpointConfig, ModelCheckpoint
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.dataset import create_train_dataset
from src.srcnn import SRCNN

from src.model_utils.moxing_adapter import sync_data
import moxing

set_seed(1)

parser = argparse.ArgumentParser(description='srcnn', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_url', type=str, default=None, help='Location of Data')
parser.add_argument('--train_url', type=str, default='', help='Location of training outputs')
parser.add_argument('--image_width', type=int, default=512, help='Weight of image')
parser.add_argument('--image_height', type=int, default=512, help='Height of image')
parser.add_argument('--data_path', type=str, default="/cache/data/", help='path of data')
parser.add_argument('--enable_modelarts', type=ast.literal_eval, default=False, help='choose modelarts')
parser.add_argument('--output_path', type=str, default="/cache/train/", help='path of output')
parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU'],
                    help='device target, support Ascend and GPU.')
parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='run distribute.')
parser.add_argument('--batch_size', type=int, default=16, help='batch size.')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate.')
parser.add_argument('--save_checkpoint', type=ast.literal_eval, default=True, help='save checkpoint file')
parser.add_argument('--keep_checkpoint_max', type=int, default=20, help='keep checkpoint max')
parser.add_argument('--epoch_size', type=int, default=20, help='epoch size.')
parser.add_argument('--filter_weight', type=ast.literal_eval, default=False, help='filter weight.')
parser.add_argument('--pre_trained_path', type=str, default='', help='pretrained path.')
args, unknown = parser.parse_known_args()

def save_ckpt_to_air(save_ckpt_path, path):
    srcnn = SRCNN()
    # load the parameter into net
    load_checkpoint(path, net=srcnn)
    input_size = np.random.uniform(0.0, 1.0, size=[1, 1, args.image_width, args.image_height]).astype(np.float32)
    export(srcnn, Tensor(input_size), file_name=save_ckpt_path +'srcnn', file_format="AIR")

def filter_checkpoint_parameter_by_list(origin_dict, param_filter):
    """remove useless parameters according to filter_list"""
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del origin_dict[key]
                break

def run_train():
    pretrained_ckpt_path = args.pre_trained_path
    if args.enable_modelarts == "True":
        sync_data(args.data_url, args.data_path)
        if args.pre_trained_path:
            sync_data(args.pre_trained_path, pretrained_ckpt_path)
    data_path1 = args.data_url
    data_path1 += "/srcnn.mindrecord00"
    if args.device_target == "GPU":
        if args.run_distribute:
            init()
        context.set_context(mode=context.GRAPH_MODE,
                            device_target=args.device_target,
                            save_graphs=False)
    elif args.device_target == "Ascend":
        context.set_context(mode=context.GRAPH_MODE,
                            device_target=args.device_target,
                            device_id=int(os.environ["DEVICE_ID"]),
                            save_graphs=False)
        if args.run_distribute:
            init()
    else:
        raise ValueError("Unsupported device target.")

    rank = 0
    device_num = 1
    if args.run_distribute:
        rank = get_rank()
        device_num = get_group_size()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL)
        train_dataset = create_train_dataset(data_path1, args.batch_size,
                                             shard_id=rank, num_shard=device_num)
    else:
        train_dataset = create_train_dataset(data_path1, args.batch_size,
                                             shard_id=rank, num_shard=device_num)
    step_size = train_dataset.get_dataset_size()

    # define net
    net = SRCNN()

    # init weight
    if args.pre_trained_path:
        param_dict = load_checkpoint(pretrained_ckpt_path)
        if args.filter_weight:
            filter_list = [x.name for x in net.end_point.get_parameters()]
            filter_checkpoint_parameter_by_list(param_dict, filter_list)
        load_param_into_net(net, param_dict)

    lr = Tensor(float(args.lr), ms.float32)

    opt = nn.Adam(params=net.trainable_params(), learning_rate=lr, eps=1e-07)
    loss = nn.MSELoss(reduction='mean')
    model = Model(net, loss_fn=loss, optimizer=opt)

    # define callbacks
    callbacks = [LossMonitor(), TimeMonitor(data_size=step_size)]
    epoch = 1
    if args.save_checkpoint and rank == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=step_size,
                                     keep_checkpoint_max=args.keep_checkpoint_max)
        save_ckpt_path = os.path.join(args.output_path, 'ckpt_' + str(rank) + '/')
        ckpt_cb = ModelCheckpoint(prefix="srcnn", directory=save_ckpt_path, config=config_ck)
        callbacks.append(ckpt_cb)
    print("============== Starting Training ==============")
    model.train(epoch, train_dataset, callbacks=callbacks)
    print("============== End Training ==============")
    if args.enable_modelarts == "True":
        sync_data(args.output_path, args.train_url)
    path = os.path.join(save_ckpt_path, 'srcnn'+'-'+str(epoch)+'_'+str(step_size)+'.ckpt')
    save_ckpt_to_air(save_ckpt_path, path)
    moxing.file.copy_parallel(save_ckpt_path, args.train_url)
if __name__ == '__main__':
    run_train()
