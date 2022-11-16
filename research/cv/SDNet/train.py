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
"""python train.py"""

import os
import random
import numpy as np
import mindspore.nn as nn
import mindspore as ms
from mindspore.context import ParallelMode
from mindspore import context, Model, Tensor
from mindspore.communication.management import init, get_rank, get_group_size
from src.model.SDNet import SDNet, EvalCallBack, CustomWithLossCell, TimeLossMonitor, Focalnet, CustomTrainOneStepCell
from src.model.Decoder import Lower, Decoder
from src.dataset import create_loaders
from src.EvalMetrics import ErrorRateAt95Recall, Accuracy
from src.Losses import Losses
from model_utils.device_adapter import get_device_id
from model_utils.config import config

ms.common.set_seed(config.seed)
random.seed(config.seed)
np.random.seed(config.seed)


def get_lr(base_lr, total_epochs, steps_per_epoch, device_num, global_epoch=0):
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    global_steps = steps_per_epoch * global_epoch
    for i in range(total_steps):
        lr_step = base_lr * (1.0 - float(i) * float(config.batch_size) * device_num
                             / (config.num_train * float(config.epochs)))
        lr_each_step.append(lr_step)
    lr_each_step = lr_each_step[global_steps:]
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    return lr_each_step


if __name__ == '__main__':
    ckpt_save_dir = config.ckpt_save_dir

    if config.modelArts_mode:
        import moxing as mox

        local_data_url = '/cache/data'
        mox.file.copy_parallel(src_url=config.data_url, dst_url=local_data_url)
        local_train_url = '/cache/ckpt/'
        # download dataset from obs to cache
        if "obs://" in config.checkpoint_path:
            local_checkpoint_url = "/cache/" + config.checkpoint_path.split("/")[-1]
            mox.file.copy_parallel(config.checkpoint_path, local_checkpoint_url)
            config.checkpoint_path = local_checkpoint_url
        config.dataroot = local_data_url
        ckpt_save_dir = local_train_url

    device_id = get_device_id()

    if not os.path.exists(ckpt_save_dir):
        os.makedirs(ckpt_save_dir)

    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)

    if config.device_target == "GPU":
        context.set_context(enable_graph_kernel=True)
    elif config.device_target == "Ascend":
        context.set_context(device_id=device_id)

    if config.is_distributed:
        init()
        config.rank = get_rank()
        config.group_size = get_group_size()
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True,
                                          device_num=config.group_size)
    model_Feature = Focalnet()
    model_Lower = Lower()
    model_Decoder = Decoder()
    network = SDNet(model_Feature, model_Lower, model_Decoder)
    network.set_train(True)
    train_data, eval_data = create_loaders(config=config)
    step_size = train_data.get_dataset_size()
    net_loss = Losses(config)
    withloss_net = CustomWithLossCell(network, net_loss)
    lr = Tensor(get_lr(base_lr=config.lr, total_epochs=config.epochs, steps_per_epoch=step_size,
                       device_num=config.group_size, global_epoch=0))
    model_Feature_opt = nn.Adam(model_Feature.trainable_params(),
                                learning_rate=lr,
                                beta1=config.beta1,
                                beta2=config.beta2,
                                weight_decay=config.wd)
    model_Lower_opt = nn.Adam(model_Lower.trainable_params(),
                              learning_rate=lr,
                              beta1=config.beta1,
                              beta2=config.beta2,
                              weight_decay=config.wd)
    model_Decoder_opt = nn.Adam(model_Decoder.trainable_params(),
                                learning_rate=lr,
                                beta1=config.beta1,
                                beta2=config.beta2,
                                weight_decay=config.wd)
    train_net = CustomTrainOneStepCell(withloss_net, model_Feature_opt, model_Lower_opt, model_Decoder_opt)
    model = Model(network=train_net)

    fpr95 = ErrorRateAt95Recall()
    acc_fn = Accuracy()
    eval_cb = EvalCallBack(network, eval_data, ckpt_save_dir, acc_fn, fpr95)

    time_loss_cb = TimeLossMonitor(lr_base=lr.asnumpy())
    cb = [time_loss_cb, eval_cb]
    if config.is_distributed and device_id != 0:
        cb = [time_loss_cb]

    model.train(config.epochs, train_data, callbacks=cb, dataset_sink_mode=False)

    if config.modelArts_mode:
        # copy train result from cache to obs
        if config.rank == 0:
            mox.file.copy_parallel(src_url=local_train_url, dst_url=config.train_url)
