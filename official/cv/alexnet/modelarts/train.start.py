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
######################## train alexnet example ########################
train alexnet and get network model files(.ckpt) :
python train.py --data_path /YourDataPath
"""

import os
import argparse
import glob
import sys
import time
import numpy as np
import moxing as mox


from src.model_utils.device_adapter import get_device_id, get_device_num, get_rank_id, get_job_id
from src.dataset import create_dataset_cifar10, create_dataset_imagenet
from src.generator_lr import get_lr_cifar10, get_lr_imagenet
from src.alexnet import AlexNet
from src.get_param_groups import get_param_groups

import mindspore.nn as nn
from mindspore.communication.management import init, get_rank
from mindspore import context
from mindspore import export
from mindspore import Tensor
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.nn.metrics import Accuracy
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument("--enable_modelarts", type=bool, default=True, help="")
parser.add_argument("--output_path", type=str, default="/cache/train", help="setting dir of training output")
parser.add_argument("--checkpoint_path", type=str, default="./checkpoint/", help="setting dir of checkpoint output")
parser.add_argument('--device_target', type=str, default='Ascend', choices=("Ascend", "GPU", "CPU"),
                    help="Device target, support Ascend, GPU and CPU.")
parser.add_argument("--dataset_name", type=str, default="cifar10", choices=("cifar10", "imagenet"),
                    help="Dataset Name, support cifar10 and imagenet")
parser.add_argument("--learning_rate", type=float, default=0.002, help="")
parser.add_argument("--epoch_size", type=int, default=30, help="")
parser.add_argument("--data_path", type=str, default="/cache/data", help="path to dataset")
parser.add_argument("--batch_size", type=int, default=1, help="")
parser.add_argument("--num_classes", type=int, default=10, help="")
parser.add_argument("--sink_size", type=int, default=-1, help="")
parser.add_argument("--momentum", type=float, default=0.9, help="")
parser.add_argument("--save_checkpoint_steps", type=int, default=1562, help="")
parser.add_argument("--data_url", type=str, default="", help="")
parser.add_argument("--train_url", type=str, default="", help="")
parser.add_argument("--image_height", type=int, default=227, help="")
parser.add_argument("--image_width", type=int, default=227, help="")
parser.add_argument("--buffer_size", type=int, default=1000, help="")
parser.add_argument("--dataset_sink_mode", type=bool, default=True, help="")
parser.add_argument("--weight_decay", type=float, default=0.0001, help="")
parser.add_argument("--loss_scale", type=int, default=1024, help="")
parser.add_argument("--is_dynamic_loss_scale", type=int, default=0, help="")
parser.add_argument("--keep_checkpoint_max", type=int, default=10, help="")
parser.add_argument("--ckpt_path", type=str, default="/cache/train", help="")
config = parser.parse_args()

set_seed(1)
_global_sync_count = 0

def frozen_to_air(net, args):
    param_dict = load_checkpoint(args.get("ckpt_file"))
    load_param_into_net(net, param_dict)
    input_arr = Tensor(np.zeros([args.get("batch_size"), \
    3, args.get("image_height"), args.get("image_width")], np.float32))
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

def train_alexnet_model():
    print(config)
    print('device id:', get_device_id())
    print('device num:', get_device_num())
    print('rank id:', get_rank_id())
    print('job id:', get_job_id())
    device_target = config.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    context.set_context(save_graphs=False)
    if device_target == "GPU":
        context.set_context(enable_graph_kernel=True)
        context.set_context(graph_kernel_flags="--enable_cluster_ops=MatMul")

    device_num = get_device_num()
    if config.dataset_name == "cifar10":
        if device_num > 1:
            config.learning_rate = config.learning_rate * device_num
            config.epoch_size = config.epoch_size * 2
    elif config.dataset_name == "imagenet":
        pass
    elif config.dataset_name != "imagenet":
        raise ValueError("Unsupported dataset.")

    if device_num > 1:
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num, \
            parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
        if device_target == "Ascend":
            context.set_context(device_id=get_device_id())
            init()
        elif device_target == "GPU":
            init()
    else:
        context.set_context(device_id=get_device_id())

    if config.dataset_name == "cifar10":
        ds_train = create_dataset_cifar10(config, config.data_path, config.batch_size, target=config.device_target)
    elif config.dataset_name == "imagenet":
        ds_train = create_dataset_imagenet(config, config.data_path, config.batch_size)
    else:
        raise ValueError("Unsupported dataset.")

    if ds_train.get_dataset_size() == 0:
        raise ValueError("Please check dataset size > 0 and batch_size <= dataset size")

    network = AlexNet(config.num_classes, phase='train')

    loss_scale_manager = None
    metrics = None
    step_per_epoch = ds_train.get_dataset_size() if config.sink_size == -1 else config.sink_size

    if config.dataset_name == 'cifar10':
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        lr = Tensor(get_lr_cifar10(0, config.learning_rate, config.epoch_size, step_per_epoch))
        opt = nn.Momentum(network.trainable_params(), lr, config.momentum)
        metrics = {"Accuracy": Accuracy()}

    elif config.dataset_name == 'imagenet':
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        lr = Tensor(get_lr_imagenet(config.learning_rate, config.epoch_size, step_per_epoch))
        opt = nn.Momentum(params=get_param_groups(network),
                          learning_rate=lr,
                          momentum=config.momentum,
                          weight_decay=config.weight_decay,
                          loss_scale=config.loss_scale)

        from mindspore.train.loss_scale_manager import DynamicLossScaleManager, FixedLossScaleManager
        if config.is_dynamic_loss_scale == 1:
            loss_scale_manager = DynamicLossScaleManager(init_loss_scale=65536, scale_factor=2, scale_window=2000)
        else:
            loss_scale_manager = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    else:
        raise ValueError("Unsupported dataset.")

    if device_target == "Ascend":
        model = Model(network, loss_fn=loss, optimizer=opt, metrics=metrics, amp_level="O2", keep_batchnorm_fp32=False,
                      loss_scale_manager=loss_scale_manager)
    elif device_target == "GPU":
        model = Model(network, loss_fn=loss, optimizer=opt, metrics=metrics, amp_level="O2",
                      loss_scale_manager=loss_scale_manager)
    else:
        raise ValueError("Unsupported platform.")

    if device_num > 1:
        ckpt_save_dir = os.path.join(config.ckpt_path + "_" + str(get_rank()))
    else:
        ckpt_save_dir = config.ckpt_path

    time_cb = TimeMonitor(data_size=step_per_epoch)
    config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_steps,
                                 keep_checkpoint_max=config.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_alexnet", directory=ckpt_save_dir, config=config_ck)

    print("============== Starting Training ==============")
    model.train(config.epoch_size, ds_train, callbacks=[time_cb, ckpoint_cb, LossMonitor()],
                dataset_sink_mode=config.dataset_sink_mode, sink_size=config.sink_size)
    ckpt_list = glob.glob(str(ckpt_save_dir) + "/*alexnet*.ckpt")
    if not ckpt_list:
        print("ckpt file not generated")
        ckpt_list.sort(key=os.path.getmtime)
    ckpt_model = ckpt_list[-1]

    network = AlexNet(config.num_classes, phase='train')

    frozen_to_air_args = {"ckpt_file": ckpt_model,
                          "batch_size": config.batch_size,
                          "image_height": config.image_height,
                          "image_width": config.image_width,
                          "file_name": "/cache/train/alexnet",
                          "file_format": "AIR"}
    frozen_to_air(network, frozen_to_air_args)
    mox.file.copy_parallel(config.output_path, config.train_url)

if __name__ == "__main__":
    wrapped_func(config)
    train_alexnet_model()
