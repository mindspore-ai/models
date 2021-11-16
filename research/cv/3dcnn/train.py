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
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
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
        if target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(device_id=device_id)
            group_size = 8
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=False)
            init()
        else:
            # target == "GPU"
            init()
            device_id = get_rank()
            group_size = get_group_size()
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                              device_num=group_size,
                                              gradients_mean=False)
    else:
        if target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID', '0'))
            context.set_context(device_id=device_id)
        else:
            # target == "GPU"
            device_id = int(config.device_id)
            context.set_context(device_id=device_id)
            group_size = 1


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
                                   target=target, mindrecord_path=config.mindrecord_path,
                                   use_mindrecord=config.use_mindrecord, group_size=group_size,
                                   device_id=device_id)

    train_data_size = train_dataset.get_dataset_size()

    # create network
    network = Dense24(config.num_classes)
    net_with_loss = NetWithLoss(network, config.num_classes)
    network.set_train(True)

    rank_size = int(os.getenv("RANK_SIZE", "1"))
    if config.use_dynamic_lr:
        lr = Tensor(dynamic_lr(config, train_data_size, rank_size), mstype.float32)
    else:
        lr = Tensor(float(config.lr), mstype.float32)

    if config.use_loss_scale:
        loss_scale = config.loss_scale
        scale_manager = FixedLossScaleManager(loss_scale=loss_scale, drop_overflow_update=True)
    else:
        scale_manager = None
        loss_scale = 1.0

    if config.use_optimizer == "SGD":
        optimizer = nn.SGD(params=network.trainable_params(), learning_rate=lr, momentum=config.momentum,
                           weight_decay=config.weight_decay, nesterov=True)
    elif config.use_optimizer == "Adam":
        optimizer = nn.Adam(params=network.trainable_params(),
                            learning_rate=lr,
                            loss_scale=loss_scale)

    model = Model(net_with_loss, optimizer=optimizer, loss_scale_manager=scale_manager)

    # save checkpoint
    time_cb = TimeMonitor(data_size=train_data_size)
    loss_cb = LossMonitor()
    ckpt_config = CheckpointConfig(save_checkpoint_steps=train_data_size,
                                   keep_checkpoint_max=config.keep_checkpoint_max)

    if config.isModelArts:
        save_checkpoint_path = '/cache/train_output/device_{}/'.format(device_id)
    else:
        save_checkpoint_path = './result/ckpt_{}/'.format(device_id)

    ckpoint_cb = ModelCheckpoint(prefix='{}'.format(config.model),
                                 directory=save_checkpoint_path,
                                 config=ckpt_config)
    callbacks_list = [loss_cb, time_cb, ckpoint_cb]
    print("============== Starting Training ==============")
    model.train(config.epoch_size, train_dataset, callbacks=callbacks_list, dataset_sink_mode=config.dataset_sink_mode)

    if config.isModelArts:
        mox.file.copy_parallel(src_url='/cache/train_output', dst_url=config.train_url)
    print("============== End Training ==============")
