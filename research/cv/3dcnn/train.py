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
python train.py
"""
import os
import random

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common import set_seed
import mindspore.common.dtype as mstype
from mindspore.train.model import Model, ParallelMode
from mindspore.communication.management import init
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor

from src.models import Dense24
from src.lr_schedule import dynamic_lr
from src.dataset import create_dataset
from src.loss import NetWithLoss
from src.config import config

if config.isModelArts:
    import moxing as mox

random.seed(1)
set_seed(1)

if __name__ == '__main__':
    target = config.device_target

    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
    if config.run_distribute:
        device_id = int(os.getenv('DEVICE_ID'))
        context.set_context(device_id=device_id)
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=False)
        init()
    else:
        device_id = int(os.getenv('DEVICE_ID', '0'))
        context.set_context(device_id=device_id)

    if config.isModelArts:
        mox.file.copy_parallel(src_url=config.data_url, dst_url='/cache/dataset/device_{}'.format(device_id))
        train_dataset_path = '/cache/dataset/device_{}'.format(device_id)

        order = 'cd ' + train_dataset_path + ';'
        order = order + 'tar -xzf MICCAI_BraTS17_Data_Training.tar.gz' + ';'
        order = order + 'cd ../../../'
        os.system(order)
        train_dataset_path = os.path.join(train_dataset_path, "MICCAI_BraTS17_Data_Training/HGG")
    else:
        train_dataset_path = config.data_path

    # create dataset
    train_dataset = create_dataset(train_dataset_path, config.train_path, config.height_size, config.width_size,
                                   config.channel_size, config.pred_size, config.batch_size, config.correction,
                                   target="Ascend")
    train_data_size = train_dataset.get_dataset_size()

    # create network
    network = Dense24(config.num_classes)
    net_with_loss = NetWithLoss(network, config.num_classes)
    network.set_train(True)

    # lr = config.lr
    rank_size = int(os.getenv("RANK_SIZE", "1"))
    lr = Tensor(dynamic_lr(config, train_data_size, rank_size), mstype.float32)

    # optimizer
    optimizer = nn.SGD(params=network.trainable_params(), learning_rate=lr, momentum=config.momentum,
                       weight_decay=config.weight_decay, nesterov=True)
    model = Model(net_with_loss, optimizer=optimizer)

    # save checkpoint
    time_cb = TimeMonitor(data_size=train_data_size)
    loss_cb = LossMonitor()
    ckpt_config = CheckpointConfig(save_checkpoint_steps=train_data_size,
                                   keep_checkpoint_max=config.keep_checkpoint_max)

    if config.isModelArts:
        save_checkpoint_path = '/cache/train_output/device_{}/'.format(device_id)
    else:
        save_checkpoint_path = './ckpt_{}/'.format(device_id)

    ckpoint_cb = ModelCheckpoint(prefix='{}'.format(config.model),
                                 directory=save_checkpoint_path,
                                 config=ckpt_config)
    callbacks_list = [loss_cb, time_cb, ckpoint_cb]
    print("============== Starting Training ==============")
    model.train(config.epoch_size, train_dataset, callbacks=callbacks_list, dataset_sink_mode=True)

    if config.isModelArts:
        mox.file.copy_parallel(src_url='/cache/train_output', dst_url=config.train_url)
    print("============== End Training ==============")
