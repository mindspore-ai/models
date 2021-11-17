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

"""
######################## train lenet example ########################
train lenet and get network model files(.ckpt) :
python train.py --data_path /YourDataPath
"""

import os
import argparse
import glob
import sys
import time
import numpy as np
import moxing as mox

from src.model_utils.moxing_adapter import get_device_id, get_device_num, get_rank_id, get_job_id
from src.dataset import create_dataset
from src.lenet import LeNet5

import mindspore.nn as nn
from mindspore.context import ParallelMode
from mindspore.communication.management import init
from mindspore import context
from mindspore import export
from mindspore import Tensor
from mindspore.train import Model

from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn.metrics import Accuracy
from mindspore.common import set_seed

root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))  # src root dir
cwd = os.getcwd()
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))

parser = argparse.ArgumentParser(description='mindspore lenet training')
parser.add_argument("--enable_modelarts", default='True', type=str, help="")
parser.add_argument("--data_url", type=str, default="", help="dataset path for obs")
parser.add_argument("--train_url", type=str, default="", help="train path for obs")
parser.add_argument('--data_path', type=str, default='/cache/data', help='Dataset url for local')
parser.add_argument("--output_path", type=str, default="/cache/train", help="dir of training output for local")
# parser.add_argument("--checkpoint_path", type=str, default="./checkpoint/", help="setting dir of checkpoint output")
parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU', 'CPU'],
                    help='device where the code will be implemented. (Default: Ascend)')
parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--epoch_size', type=int, default=1, help='epoch sizse')
parser.add_argument("--learning_rate", type=float, default=0.002, help="")
parser.add_argument("--sink_size", type=int, default=-1, help="")
parser.add_argument("--momentum", type=float, default=0.9, help="")
parser.add_argument("--save_checkpoint_steps", type=int, default=125, help="")
parser.add_argument('--lr', type=float, default=0.01, help='base learning rate')
parser.add_argument("--image_height", type=int, default=32, help="")
parser.add_argument("--image_width", type=int, default=32, help="")
parser.add_argument("--buffer_size", type=int, default=1000, help="")
parser.add_argument("--keep_checkpoint_max", type=int, default=10, help="")
parser.add_argument('--z', type=str, default='AIR', choices=['AIR', 'ONNX', 'MINDIR'],
                    help='Format of output model(Default: AIR)')
parser.add_argument('--file_name', type=str, default='lenet', help='output file name')

parser.add_argument("--ckpt_path", type=str, default="/cache/train", help="")
parser.add_argument("--ckpt_file", type=str, default="/cache/train/checkpoint_lenet-10_1875.ckpt", help="")

cfg = parser.parse_args()
set_seed(1)
_global_sync_count = 0


def frozen_to_air(net, args):
    param_dict = load_checkpoint(args.get("ckpt_file"))
    load_param_into_net(net, param_dict)
    input_arr = Tensor(np.zeros([args.get("batch_size"),
                                 1, args.get("image_height"), args.get("image_width")], np.float32))
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
        Download data from remote obs to local directory if the first url is remote url and the second one is local path
        Upload data from local directory to remote obs in contrast.
    """
    if config_name.enable_modelarts:
        if config_name.data_url:
            if not os.path.isdir(config_name.data_path):
                os.makedirs(config_name.data_path)
                sync_data(config_name.data_url, config_name.data_path)
                print("Dataset downloaded: ", os.listdir(cfg.data_path))
            if config_name.train_url:
                if not os.path.isdir(config_name.output_path):
                    os.makedirs(config_name.output_path)
                sync_data(config_name.train_url, config_name.output_path)
                print("Workspace downloaded: ", os.listdir(config_name.output_path))


def train_lenet_model():
    """
        main function to train model in modelArts
    """
    print(cfg)
    print('device id:', get_device_id())
    print('device num:', get_device_num())
    print('rank id:', get_rank_id())
    print('job id:', get_job_id())
    device_target = cfg.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target)
    context.set_context(save_graphs=False)
    if device_target == "GPU":
        context.set_context(enable_graph_kernel=True)
        context.set_context(graph_kernel_flags="--enable_cluster_ops=MatMul")

    device_num = get_device_num()

    if device_num > 1:
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
        if device_target == "Ascend":
            context.set_context(device_id=get_device_id())
            init()
        elif device_target == "GPU":
            init()
    else:
        context.set_context(device_id=get_device_id())

    # create dataset
    ds_train = create_dataset(os.path.join(cfg.data_path, "train"), cfg.batch_size)

    if ds_train.get_dataset_size() == 0:
        raise ValueError("Please check dataset size > 0 and batch_size <= dataset size")
    print("dataset size is : " + str(ds_train.get_dataset_size()))

    network = LeNet5(cfg.num_classes)

    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), cfg.lr, cfg.momentum)
    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
    config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                                 keep_checkpoint_max=cfg.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet", directory=cfg.ckpt_path, config=config_ck)

    if cfg.device_target != "Ascend":
        if cfg.device_target == "GPU":
            context.set_context(enable_graph_kernel=True)
        model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
    else:
        model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()}, amp_level="O2")

    print("============== Starting Training ==============")
    model.train(cfg.epoch_size, ds_train, callbacks=[time_cb, ckpoint_cb, LossMonitor()])

    print("============== Training finish ==============")

    ckpt_list = glob.glob(str(cfg.output_path) + "/*lenet*.ckpt")
    print(ckpt_list)
    if not ckpt_list:
        print("ckpt file not generated")
        ckpt_list.sort(key=os.path.getmtime)
    ckpt_model = ckpt_list[-1]
    print(ckpt_model)

    frozen_to_air_args = {"ckpt_file": ckpt_model,
                          "batch_size": cfg.batch_size,
                          "image_height": cfg.image_height,
                          "image_width": cfg.image_width,
                          "file_name": "/cache/train/lenet",
                          "file_format": "AIR"}
    frozen_to_air(network, frozen_to_air_args)

    mox.file.copy_parallel(cfg.output_path, cfg.train_url)


if __name__ == "__main__":
    wrapped_func(cfg)
    train_lenet_model()
