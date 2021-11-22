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
import time

import mindspore.nn as nn
from mindspore import Model
import mindspore.context as context
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.communication.management import init
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor

from src.config import config
from src.callback import EvalCallBack
from src.dataset import create_dataset
from src.resnet import resnet50 as resnet
from src.metric import EmdLoss, PrintFps, spearman
from src.device_adapter import get_device_id, get_device_num, _get_rank_info

def train_net(model, args):
    if args.is_distributed:
        args.device_num, args.rank = _get_rank_info()
    else:
        args.device_num = 1
        args.rank = args.device_id
    ds_train, steps_per_epoch_train = create_dataset(args, data_mode='train')
    ds_val, _ = create_dataset(args, data_mode='val')
    print('steps_per_epoch_train', steps_per_epoch_train, 'epoch_size', args.epoch_size)
    config_ck = CheckpointConfig(save_checkpoint_steps=steps_per_epoch_train,
                                 keep_checkpoint_max=args.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix=args.ckpt_filename, directory=args.ckpt_save_dir, config=config_ck)
    eval_per_epoch = 1
    print("============== Starting Training ==============")
    epoch_per_eval = {"epoch": [], "spearman": []}
    eval_cb = EvalCallBack(model, ds_val, eval_per_epoch, epoch_per_eval)
    train_data_num = steps_per_epoch_train*args.batch_size
    init_time = time.time()
    fps = PrintFps(train_data_num, init_time, init_time)
    time_cb = TimeMonitor(train_data_num)
    dataset_sink_mode = not args.device_target == "CPU"
    model.train(args.epoch_size, ds_train, callbacks=[ckpoint_cb, time_cb, fps, eval_cb],
                dataset_sink_mode=dataset_sink_mode, sink_size=steps_per_epoch_train)

if __name__ == "__main__":
    if not os.path.exists(config.ckpt_save_dir):
        os.mkdir(config.ckpt_save_dir)
    if config.enable_modelarts:
        import moxing as mox
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
    train_net(net, config)
    if config.enable_modelarts:
        for file in os.listdir(config.ckpt_save_dir):
            mox.file.copy(os.path.join(config.ckpt_save_dir, file),
                          os.path.join(config.output_path, 'Ascend_{}P_'.format(device_num) + file))
