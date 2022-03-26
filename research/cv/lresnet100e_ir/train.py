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
"""
python train.py
"""
import numpy as np

import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.train.model import Model, ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor, LossMonitor
from mindspore.communication.management import init, get_rank
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.ops import functional as F
from mindspore.common import set_seed, dtype
from mindspore.train.loss_scale_manager import FixedLossScaleManager

from src.dataset import create_dataset
from src.iresnet import iresnet100
from src.config import config


set_seed(1)

def lr_generator(lr_init, total_epochs, steps_per_epoch):
    """lr_generator
    """
    lr_each_step = []
    for i in range(total_epochs):
        if i in config.schedule:
            lr_init *= config.gamma
        for _ in range(steps_per_epoch):
            lr_each_step.append(lr_init)
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    return Tensor(lr_each_step)

class LResNetWithLoss(nn.Cell):
    """
    WithLossCell
    """
    def __init__(self, net, num_classes, num_features=512):
        super(LResNetWithLoss, self).__init__(auto_prefix=False)
        self._net = net.to_float(dtype.float16)
        self._fc = nn.Dense(num_features, num_classes).to_float(dtype.float16)
        self._loss_fn = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    def construct(self, data, label):
        features = self._net(data)
        output = self._fc(features)
        out = F.cast(output, dtype.float32)
        loss = self._loss_fn(out, label)
        return loss

if __name__ == "__main__":

    # set context and device init
    train_epoch = config.epoch_size
    if config.use_pynative_mode:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=config.device_target,
                            device_id=config.device_id, save_graphs=False, enable_graph_kernel=True)
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target,
                            device_id=config.device_id, save_graphs=False, enable_graph_kernel=True)

    if config.run_distribute:
        init()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                          device_num=config.device_num,
                                          all_reduce_fusion_config=config.all_reduce_fusion_config,
                                          gradients_mean=True)
        config.rank = get_rank()
    else:
        config.rank = 0

    # define dataset
    train_dataset = create_dataset(dataset_path=config.data_url, do_train=True,
                                   img_shape=config.img_shape, repeat_num=config.repeat_num,
                                   batch_size=config.batch_size, run_distribute=config.run_distribute)

    step = train_dataset.get_dataset_size()

    # define net
    network = iresnet100()
    train_net = LResNetWithLoss(network, config.num_classes)

    # define lr
    lr = lr_generator(config.lr_init, train_epoch, steps_per_epoch=step)

    # define optimizer
    optimizer = nn.SGD(params=train_net.trainable_params(),
                       learning_rate=lr / 512 * config.batch_size * config.device_num,
                       momentum=config.momentum, weight_decay=config.weight_decay, loss_scale=config.loss_scale)
    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)

    # define model
    model = Model(train_net, optimizer=optimizer, loss_scale_manager=loss_scale)

    time_cb = TimeMonitor(data_size=train_dataset.get_dataset_size())
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    if config.rank == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_steps,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix="lresnet100e_ir", config=config_ck,
                                  directory=config.checkpoint_path)
        cb.append(ckpt_cb)

    # begin train
    model.train(train_epoch, train_dataset,
                callbacks=cb, dataset_sink_mode=config.dataset_sink_mode)
