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
######################## train lenet example ########################
train lenet and get network model files(.ckpt) :
"""

import os
from src.model_utils.config import config
from src.dataset import create_dataset
from src.lenet import LeNet5

import mindspore.nn as nn
from mindspore import context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy
from mindspore.common import set_seed
from mindspore import load_checkpoint, load_param_into_net
from algorithm import create_simqat

set_seed(1)


def train_lenet():
    if config.device_target != "GPU":
        raise NotImplementedError("SimQAT only support running on GPU now!")
    if config.mode_name == "GRAPH":
        context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=config.device_target)
    ds_train = create_dataset(os.path.join(config.data_path), config.batch_size)
    if ds_train.get_dataset_size() == 0:
        raise ValueError("Please check dataset size > 0 and batch_size <= dataset size")

    network = LeNet5(config.num_classes)
    if config.fp32_ckpt:
        fp32_ckpt = load_checkpoint(config.fp32_ckpt)
        load_param_into_net(network, fp32_ckpt)
    # apply golden stick algorithm on LeNet5 model
    algo = create_simqat()
    network = algo.apply(network)

    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), config.lr, config.momentum)
    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
    config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_steps,
                                 keep_checkpoint_max=config.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet", directory="./ckpt", config=config_ck)

    context.set_context(enable_graph_kernel=True)
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    print("============== Starting Training ==============")
    model.train(config.epoch_size, ds_train, callbacks=[time_cb, ckpoint_cb, LossMonitor()])


if __name__ == "__main__":
    train_lenet()
