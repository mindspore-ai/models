'''
train protonet model on modelarts
'''
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

import os
import argparse
import time
import datetime
import numpy as np
import moxing as mox



from src.model_utils.device_adapter import get_device_id, get_device_num, get_rank_id, get_job_id
from src.protonet import ProtoNet
from src.PrototypicalLoss import PrototypicalLoss
from src.protonet import WithLossCell
from src.EvalCallBack import EvalCallBack
from model_init import init_dataloader

import mindspore.nn as nn
from mindspore.communication.management import init
from mindspore import context
from mindspore import export
from mindspore import Tensor
from mindspore import dataset as ds
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed


parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument("--enable_modelarts", type=bool, default=True, help="")
parser.add_argument('--device_target', type=str, default='Ascend', choices=("Ascend", "GPU", "CPU"),
                    help="Device target, support Ascend, GPU and CPU.")
parser.add_argument("--data_path", type=str, default="/cache/data", help="path to dataset on modelarts")
parser.add_argument("--data_url", type=str, default="", help="path to dataset on obs")
parser.add_argument("--train_url", type=str, default="", help="path to training output on obs")
parser.add_argument("--output_path", type=str, default="/cache/out", help="path to training output on modelarts")

parser.add_argument("--learning_rate", type=float, default=0.001, help="")
parser.add_argument("--epoch_size", type=int, default=1, help="")
parser.add_argument("--save_checkpoint_steps", type=int, default=10, help="")
parser.add_argument("--keep_checkpoint_max", type=int, default=5, help="")

parser.add_argument("--batch_size", type=int, default=100, help="")
parser.add_argument("--image_height", type=int, default=28, help="")
parser.add_argument("--image_width", type=int, default=28, help="")
parser.add_argument("--file_name", type=str, default="protonet", help="the name of air file ")

parser.add_argument('-cTr', '--classes_per_it_tr',
                    type=int,
                    help='number of random classes per episode for training, default=60',
                    default=20)
parser.add_argument('-nsTr', '--num_support_tr',
                    type=int,
                    help='number of samples per class to use as support for training, default=5',
                    default=5)
parser.add_argument('-nqTr', '--num_query_tr',
                    type=int,
                    help='number of samples per class to use as query for training, default=5',
                    default=5)
parser.add_argument('-cVa', '--classes_per_it_val',
                    type=int,
                    help='number of random classes per episode for validation, default=5',
                    default=20)
parser.add_argument('-nsVa', '--num_support_val',
                    type=int,
                    help='number of samples per class to use as support for validation, default=5',
                    default=5)
parser.add_argument('-nqVa', '--num_query_val',
                    type=int,
                    help='number of samples per class to use as query for validation, default=15',
                    default=15)
parser.add_argument('-its', '--iterations',
                    type=int,
                    help='number of episodes per epoch, default=100',
                    default=100)


config = parser.parse_args()

set_seed(1)
_global_sync_count = 0

def frozen_to_air(net, args):
    param_dict = load_checkpoint(args.get("ckpt_file"))
    load_param_into_net(net, param_dict)
    input_arr = Tensor(np.zeros([args.get("batch_size"), 1,
                                 args.get("image_height"), args.get("image_width")], np.float32))
    export(net, input_arr, file_name=args.get("file_name"), file_format=args.get("file_format"))


def sync_data(from_path, to_path):
    """
    Download data from remote obs to local directory if the first url is remote url and the second one is local path
    Upload data from local directory to remote obs in contrast.
    """
    global _global_sync_count
    sync_lock = "/tmp/copy_sync.lock" + str(_global_sync_count)
    _global_sync_count += 1

    # Each server contains 8 devices as most.
    if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
        print("from path: ", from_path)
        print("to path: ", to_path)
        mox.file.copy_parallel(from_path, to_path)
        print("===finish data synchronization===")
        try:
            os.mknod(sync_lock)
        except IOError:
            print("Failed to create directory")
        print("===save flag===")

    while True:
        if os.path.exists(sync_lock):
            break
        time.sleep(1)

    print("Finish sync data from {} to {}.".format(from_path, to_path))


def wrapped_func(config_name):
    """
    Transfer data and file from obs to modelarts
    """
    if config_name.enable_modelarts:
        if config_name.data_url:
            if not os.path.isdir(config_name.data_path):
                os.makedirs(config_name.data_path)
                sync_data(config_name.data_url, config_name.data_path)
                print("Dataset downloaded: ", os.listdir(config.data_path))
            if config_name.train_url:
                if not os.path.isdir(config_name.output_path):
                    os.makedirs(config_name.output_path)
                sync_data(config_name.train_url, config_name.output_path)
                print("Workspace downloaded: ", os.listdir(config_name.output_path))

def train_protonet_model():
    '''
    train protonet model
    '''
    print(config)
    print('device id:', get_device_id())
    print('device num:', get_device_num())
    print('rank id:', get_rank_id())
    print('job id:', get_job_id())

    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    context.set_context(save_graphs=False)

    device_target = config.device_target
    if device_target == "GPU":
        context.set_context(enable_graph_kernel=True)
        context.set_context(graph_kernel_flags="--enable_cluster_ops=MatMul")

    device_num = get_device_num()
    if device_num > 1:
        if device_target == "Ascend":
            init()
        elif device_target == "GPU":
            init()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          radients_mean=True)
    context.set_context(device_id=get_device_id())

    tr_dataloader = init_dataloader(config, 'train', config.data_path)
    val_dataloader = init_dataloader(config, 'val', config.data_path)

    Net = ProtoNet()
    loss_fn = PrototypicalLoss(config.num_support_tr, config.num_query_tr, config.classes_per_it_tr)
    eval_loss_fn = PrototypicalLoss(config.num_support_tr, config.num_query_tr,
                                    config.classes_per_it_val, is_train=False)
    my_loss_cell = WithLossCell(Net, loss_fn)
    my_acc_cell = WithLossCell(Net, eval_loss_fn)
    optim = nn.Adam(params=Net.trainable_params(), learning_rate=config.learning_rate)
    model = Model(my_loss_cell, optimizer=optim)

    train_data = ds.GeneratorDataset(tr_dataloader, column_names=['data', 'label', 'classes'])
    eval_data = ds.GeneratorDataset(val_dataloader, column_names=['data', 'label', 'classes'])
    eval_cb = EvalCallBack(config, my_acc_cell, eval_data, config.output_path)

    config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_steps,
                                 keep_checkpoint_max=config.keep_checkpoint_max,
                                 saved_network=Net)
    ckpoint_cb = ModelCheckpoint(prefix='protonet_ckpt', directory=config.output_path, config=config_ck)

    print("============== Starting Training ==============")
    starttime = datetime.datetime.now()
    model.train(config.epoch_size, train_data, callbacks=[ckpoint_cb, eval_cb, TimeMonitor()],)
    endtime = datetime.datetime.now()
    print('epoch time: ', (endtime - starttime).seconds / 10, 'per step time:', (endtime - starttime).seconds / 1000)

    frozen_to_air_args = {"ckpt_file": config.output_path + "/" + "best_ck.ckpt",
                          "batch_size": config.batch_size,
                          "image_height": config.image_height,
                          "image_width": config.image_width,
                          "file_name": config.output_path + "/" + config.file_name,
                          "file_format": "AIR"}

    frozen_to_air(Net, frozen_to_air_args)
    mox.file.copy_parallel(config.output_path, config.train_url)

if __name__ == "__main__":
    wrapped_func(config)
    train_protonet_model()
