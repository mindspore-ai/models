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
"""train ICNet and get checkpoint files."""
import os
import sys
import logging
import argparse
import yaml
import numpy as np
from mindspore import Tensor, load_checkpoint, load_param_into_net, export
import mindspore.common.dtype as dtype
import mindspore.nn as nn
from mindspore import Model
from mindspore import context
from mindspore import set_seed
from mindspore.context import ParallelMode
from mindspore.communication import init
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import LossMonitor
from mindspore.train.callback import TimeMonitor

import moxing as mox

rank_id = int(os.getenv('RANK_ID'))
device_id = int(os.getenv('DEVICE_ID'))
device_num = int(os.getenv('RANK_SIZE'))
local_data_url = '/cache/data'
local_train_url = '/cache/train'

context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')

parser = argparse.ArgumentParser(description="ICNet Train")

parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="AIR", help="file format")
parser.add_argument('--device_target', type=str, default="Ascend",
                    choices=['Ascend', 'GPU', 'CPU'], help='device target (default: Ascend)')
parser.add_argument("--data_url", type=str, default=None, help="data_url")
parser.add_argument("--train_url", type=str, default=None, help="train_url")
parser.add_argument("--epoch_size", type=int, default=160, help="epoch_size")

args = parser.parse_args()


def get_last_ckpt(ckpt_dir):
    """get_last_ckpt"""
    ckpt_files = [ckpt_file for ckpt_file in os.listdir(ckpt_dir)
                  if ckpt_file.endswith('.ckpt')]
    if not ckpt_files:
        print("No ckpt file found.")
        return None

    return os.path.join(ckpt_dir, sorted(ckpt_files)[-1])


def export_air(ckpt_dir):
    """export_air"""
    ckpt_file = get_last_ckpt(ckpt_dir)
    air_name = os.path.join(ckpt_dir, 'icnet')
    net = ICNet(pretrained_path=os.path.join(project_path, 'ResNet50V1B-150_625.ckpt'))

    param_dict = load_checkpoint(ckpt_file)

    load_param_into_net(net, param_dict)
    net.set_train(False)

    img = Tensor(np.ones([1, 3, cfg['model']["base_size"], cfg['model']["base_size"] * 2]), dtype.float32)

    export(net, img, file_name=air_name, file_format=args.file_format)

    return 0


def train_net():
    """train"""
    global local_data_url, local_train_url
    set_seed(1234)
    if device_num >= 1:
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          parameter_broadcast=True,
                                          gradients_mean=True)
        init()
        device_data_url = os.path.join(local_data_url, "device{0}".format(device_id))
        device_train_url = os.path.join(local_train_url, "device{0}".format(device_id))
        local_train_file = os.path.join(device_data_url, 'cityscapes-2975.mindrecord')
    mox.file.make_dirs(local_data_url)
    mox.file.make_dirs(local_train_url)
    mox.file.make_dirs(device_data_url)
    mox.file.make_dirs(device_train_url)
    mox.file.copy_parallel(src_url=args.data_url, dst_url=device_data_url)
    mox.file.copy_parallel(args.data_url, local_data_url)
    dataset = create_icnet_dataset(local_train_file, batch_size=cfg['train']["train_batch_size_percard"],
                                   device_num=device_num, rank_id=device_id)

    train_data_size = dataset.get_dataset_size()
    print("data_size", train_data_size)
    epoch = args.epoch_size
    if device_num > 1:
        network = ICNetdc(pretrained_path=os.path.join(project_path, 'ResNet50V1B-150_625.ckpt'))
    else:
        network = ICNetdc(pretrained_path=os.path.join(project_path, 'ResNet50V1B-150_625.ckpt'),
                          norm_layer=nn.BatchNorm2d)

    iters_per_epoch = train_data_size
    total_train_steps = iters_per_epoch * epoch
    base_lr = cfg["optimizer"]["init_lr"]
    iter_lr = poly_lr(base_lr, total_train_steps, total_train_steps, end_lr=0.0, power=0.9)
    optim = nn.SGD(params=network.trainable_params(), learning_rate=iter_lr, momentum=cfg["optimizer"]["momentum"],
                   weight_decay=cfg["optimizer"]["weight_decay"])

    model = Model(network, optimizer=optim, metrics=None)

    config_ck_train = CheckpointConfig(save_checkpoint_steps=iters_per_epoch * cfg["train"]["save_checkpoint_epochs"],
                                       keep_checkpoint_max=cfg["train"]["keep_checkpoint_max"])

    ckpoint_cb_train = ModelCheckpoint(prefix='ICNet', directory=device_train_url, config=config_ck_train)
    time_cb_train = TimeMonitor(data_size=dataset.get_dataset_size())
    loss_cb_train = LossMonitor()
    print("train begins------------------------------")
    model.train(epoch=epoch, train_dataset=dataset, callbacks=[ckpoint_cb_train, loss_cb_train, time_cb_train],
                dataset_sink_mode=True)

    export_air(device_train_url)

if __name__ == '__main__':
    project_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(project_path)
    from src.cityscapes_mindrecord import create_icnet_dataset
    from src.models.icnet_dc import ICNetdc
    from src.models import ICNet
    from src.lr_scheduler import poly_lr
    config_file = "src/model_utils/icnet.yaml"
    config_path = os.path.join(project_path, config_file)
    with open(config_path, "r") as yaml_file:
        cfg = yaml.load(yaml_file.read())
    logging.basicConfig(level=logging.INFO)
    train_net()
    mox.file.copy("/cache/train/device0/ICNet-" + str(args.epoch_size) + "_744.ckpt", os.path.join(args.train_url,
                                                                                                   "ICNet.ckpt"))
    mox.file.copy("/cache/train/device0/icnet.air", os.path.join(args.train_url, "icnet.air"))
