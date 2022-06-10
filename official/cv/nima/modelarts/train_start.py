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
import os
import time
import argparse
import glob
import ast
import moxing as mox
import numpy as np

import mindspore.nn as nn
from mindspore import Model
import mindspore.context as context
from mindspore.common import set_seed
from mindspore.common import dtype as mstype
from mindspore.context import ParallelMode
from mindspore.communication.management import init
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore import Tensor, export

from src.callback import EvalCallBack
from src.dataset import create_dataset
from src.resnet import resnet50 as resnet
from src.metric import EmdLoss, PrintFps, spearman
from src.device_adapter import get_device_id, get_device_num, _get_rank_info


def model_export(arguments):
    """export air"""
    output_dir = arguments.local_output_dir
    epoch_size = str(arguments.epoch_size)
    ckpt_file = glob.glob(output_dir + '/' + '*' + epoch_size + '*' + '.ckpt')[0]
    print("ckpt_file: ", ckpt_file)
    network = resnet(10)
    param_dict_ckpt = load_checkpoint(ckpt_file)
    load_param_into_net(network, param_dict_ckpt)
    img = np.random.randint(0, 255, size=(1, 3, config.image_size, config.image_size))
    img = Tensor(np.array(img), mstype.float32)
    export_file = os.path.join(output_dir, arguments.file_name)
    export(network, img, file_name=export_file, file_format=arguments.file_format)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image classification')
    parser.add_argument("--enable_modelarts", type=ast.literal_eval, default=True, help="")
    parser.add_argument("--is_distributed", type=ast.literal_eval, default=False, help="")
    parser.add_argument('--device_target', type=str, default='Ascend', choices=("Ascend", "GPU", "CPU"),
                        help="Device target, support Ascend, GPU and CPU.")
    parser.add_argument('--device_id', type=int, default=0, help='device id of Ascend. (Default: None)')

    parser.add_argument('--train_url', required=True, default=None, help='obs browser path')
    parser.add_argument('--data_url', required=True, default=None, help='Location of data')
    parser.add_argument('--data1_url', required=True, default=None, help='Location of data')
    parser.add_argument('--local_data_dir', type=str, default="/cache")
    parser.add_argument('--local_output_dir', type=str, default="/cache/train_output")
    parser.add_argument('--train_label_path', type=str, default="~/NIMA/AVA_train.txt")
    parser.add_argument('--val_label_path', type=str, default="~/NIMA/AVA_test.txt")
    parser.add_argument('--data_path', type=str, default="~/NIMA/data/")
    parser.add_argument('--ckpt_save_dir', type=str, default="./output")
    parser.add_argument('--ckpt_filename', type=str, default="NIMA")
    parser.add_argument('--checkpoint_path', type=str, default="")

    parser.add_argument("--learning_rate", type=float, default=0.001, help="")
    parser.add_argument("--epoch_size", type=int, default=50, help="")
    parser.add_argument("--steps_per_epoch_train", type=int, default=10, help="")
    parser.add_argument("--keep_checkpoint_max", type=int, default=10, help="")
    parser.add_argument("--num_parallel_workers", type=int, default=16, help="")
    parser.add_argument("--momentum", type=float, default=0.95, help="")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="")
    parser.add_argument("--bf_crop_size", type=int, default=256, help="")

    parser.add_argument("--batch_size", type=int, default=256, help="")
    parser.add_argument("--image_size", type=int, default=224, help="")
    parser.add_argument("--file_name", type=str, default="NIMA", help="the name of air file ")
    parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="AIR",
                        help="file format")
    parser.add_argument("--ckpt_file", type=str, default="~/NIMA/model/NIMA-2_898.ckpt")

    config = parser.parse_args()
    local_data_path = config.local_data_dir
    train_output_path = config.local_output_dir
    mox.file.copy_parallel(config.data_url, local_data_path)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~file copy success~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    if not os.path.exists(config.local_output_dir):
        os.mkdir(config.local_output_dir)
    if config.enable_modelarts:
        mox.file.shift('os', 'mox')
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    if config.is_distributed:
        device_num = get_device_num()
        config.batch_size = int(config.batch_size/device_num)
        device_id = get_device_id()
        context.set_context(device_id=device_id)
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        init()
    else:
        device_num = 1
        device_id = config.device_id
        context.set_context(device_id=device_id)
    print('batch_size:', config.batch_size, 'workers:', config.num_parallel_workers)
    print('device_id', device_id, 'device_num', device_num)
    set_seed(10)

    net = resnet(10)
    # load ckpt
    param_dict = load_checkpoint(config.checkpoint_path)
    param_dict_new = {}
    for key, values in param_dict.items():
        if key.startswith('moments.'):
            continue
        elif key.startswith('end_point'):
            continue
        else:
            param_dict_new[key] = values
    load_param_into_net(net, param_dict_new, strict_load=False)

    # loss
    criterion = EmdLoss()
    # opt
    learning_rate = config.learning_rate
    momentum = config.momentum
    weight_decay = config.weight_decay
    opt = nn.SGD(params=net.trainable_params(), learning_rate=learning_rate,
                 momentum=momentum, weight_decay=weight_decay)
    # Construct model
    metrics = {'spearman': spearman()}
    net = Model(net, criterion, opt, metrics=metrics)
    # Train
    if config.is_distributed:
        config.device_num, config.rank = _get_rank_info()
    else:
        config.device_num = 1
        config.rank = config.device_id
    ds_train, steps_per_epoch_train = create_dataset(config, data_mode='train')
    ds_val, _ = create_dataset(config, data_mode='val')
    print('steps_per_epoch_train', steps_per_epoch_train, 'epoch_size', config.epoch_size)
    config_ck = CheckpointConfig(save_checkpoint_steps=config.steps_per_epoch_train,
                                 keep_checkpoint_max=config.keep_checkpoint_max)
    ckpt_save_dir = "/cache/train_output"
    ckpoint_cb = ModelCheckpoint(prefix=config.ckpt_filename, directory=ckpt_save_dir, config=config_ck)
    eval_per_epoch = 1
    print("============== Starting Training ==============")
    epoch_per_eval = {"epoch": [], "spearman": []}
    eval_cb = EvalCallBack(net, ds_val, eval_per_epoch, epoch_per_eval)
    train_data_num = steps_per_epoch_train*config.batch_size
    init_time = time.time()
    fps = PrintFps(train_data_num, init_time, init_time)
    time_cb = TimeMonitor(train_data_num)
    dataset_sink_mode = not config.device_target == "CPU"
    net.train(config.epoch_size, ds_train, callbacks=[ckpoint_cb, time_cb, fps, eval_cb],
              dataset_sink_mode=dataset_sink_mode, sink_size=steps_per_epoch_train)
    print(os.listdir(train_output_path))
    model_export(config)

    if config.enable_modelarts:
        for file in os.listdir(config.local_output_dir):
            mox.file.copy(os.path.join(config.local_output_dir, file),
                          os.path.join(config.train_url, 'Ascend_{}P_'.format(device_num) + file))
