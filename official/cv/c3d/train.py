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
#################train C3D  ########################
"""

import datetime
import os
import mindspore
from mindspore import Tensor
from mindspore import context, FixedLossScaleManager
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.nn.optim import SGD
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_param_into_net, load_checkpoint
from mindspore.common import set_seed
from mindspore.nn.metrics import Accuracy

from src.dataset import classification_dataset
from src.utils import get_adaptive_lr_param_groups
from src.c3d_model import C3D
from src.model_utils.device_adapter import get_device_id
from src.loss import Max_Entropy
from src.lr_schedule import linear_warmup_learning_rate
from src.model_utils.config import config
from src.evalcallback import EvalCallBack


def train():
    """run train"""
    config.load_type = 'train'
    set_seed(config.seed)
    print(config)
    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=config.enable_graph_kernel_diff,
                        device_target=config.device_target, save_graphs=False, reserve_class_name_in_scope=False)
    config.device_id = get_device_id()

    if config.is_distributed:
        if config.device_target == "Ascend":
            init()
            config.rank = get_rank()
            config.group_size = get_group_size()
            config.batch_size = config.batch_size // (config.group_size)
            context.set_context(device_id=config.device_id)
        elif config.device_target == "GPU":
            init()
            config.rank = get_rank()
            config.group_size = get_group_size()

        context.set_auto_parallel_context(device_num=config.group_size, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True, parameter_broadcast=True)
    else:
        config.rank = 0
        config.group_size = 1
        context.set_context(device_id=config.device_id)

    # select for master rank save ckpt or all rank save, compatible for model parallel

    config.outputs_dir = os.path.join(config.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))

    # dataset
    config.load_type = 'train'
    train_dataset, _ = classification_dataset(config.batch_size, config.group_size, shuffle=True,
                                              repeat_num=1, drop_remainder=True)
    batch_num = train_dataset.get_dataset_size()

    # get network and init
    print('*' * 20, 'start create network', '*' * 20)
    network = C3D(config.num_classes)
    # load pre_trained model
    if config.pre_trained:
        param_dict = load_checkpoint(config.pre_trained_ckpt_path)
        param_not_load = load_param_into_net(network, param_dict)
        print('load pre_trained model, but parameters {} not load'.format(param_not_load))

    # lr scheduler
    if config.device_target == "GPU":
        lr = linear_warmup_learning_rate(lr_max=config.lr*config.group_size, epoch_step=config.milestones,
                                         global_step=0, lr_init=1e-6*config.group_size, warmup_epochs=1,
                                         total_epochs=config.epoch, steps_per_epoch=batch_num)
    elif config.device_target == "Ascend":
        lr = linear_warmup_learning_rate(lr_max=config.lr, epoch_step=config.milestones,
                                         global_step=0, lr_init=1e-6, warmup_epochs=1,
                                         total_epochs=config.epoch, steps_per_epoch=batch_num)

    # optimizer
    lr_1x_params, lr_10x_params = get_adaptive_lr_param_groups(network)
    if config.is_distributed:
        params = [{'params': lr_1x_params, 'lr': Tensor(lr, mindspore.float32) / config.group_size},
                  {'params': lr_10x_params, 'lr': Tensor(lr, mindspore.float32) / config.group_size * 10}]
    else:
        params = [{'params': lr_1x_params, 'lr': Tensor(lr, mindspore.float32)},
                  {'params': lr_10x_params, 'lr': Tensor(lr, mindspore.float32) * 10}]

    opt = SGD(params=params, learning_rate=Tensor(lr, mindspore.float32), momentum=config.momentum, dampening=0.0,
              weight_decay=config.weight_decay, nesterov=False, loss_scale=config.loss_scale)

    loss = Max_Entropy()
    loss_scale_manager = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)

    model = Model(network, loss_fn=loss, optimizer=opt, metrics={'accuracy': Accuracy()},
                  amp_level=config.amp_level_diff, loss_scale_manager=loss_scale_manager)

    # define callbacks
    time_cb = TimeMonitor(data_size=batch_num)
    loss_cb = LossMonitor(per_print_times=1)
    callbacks = [time_cb, loss_cb]
    ckpt_args = CheckpointConfig(save_checkpoint_steps=config.ckpt_interval * batch_num,
                                 keep_checkpoint_max=config.keep_checkpoint_max)
    save_ckpt_path = os.path.join(config.outputs_dir, 'ckpt_' + str(config.rank) + '/')
    ckpt_cb = ModelCheckpoint(config=ckpt_args, directory=save_ckpt_path, prefix='{}'.format(config.rank))
    callbacks.append(ckpt_cb)
    if config.is_evalcallback:
        eval_per_epoch = 5
        epoch_per_eval = {"epoch": [], "acc": []}
        eval_cb = EvalCallBack(model, eval_per_epoch, epoch_per_eval, save_ckpt_path, batch_num)
        callbacks.append(eval_cb)

    model.train(config.epoch, train_dataset, callbacks=callbacks, dataset_sink_mode=True, sink_size=-1)


if __name__ == '__main__':
    train()
