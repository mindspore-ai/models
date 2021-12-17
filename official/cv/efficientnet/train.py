# Copyright 2020 Huawei Technologies Co., Ltd
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
import math
import random
import os

import numpy as np
import mindspore
from mindspore import nn
from mindspore import Tensor, context
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore.nn import SGD, RMSProp
from mindspore.context import ParallelMode
from mindspore.train.callback import (CheckpointConfig, LossMonitor,
                                      ModelCheckpoint, TimeMonitor)
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import SummaryCollector

from src.config import config
from src.dataset import create_dataset, create_dataset_val
from src.efficientnet import efficientnet_b0, efficientnet_b1
from src.loss import LabelSmoothingCrossEntropy
from src.callbacks import EvalCallBack


mindspore.common.set_seed(config.random_seed)
random.seed(config.random_seed)
np.random.seed(config.random_seed)


def get_lr(base_lr, total_epochs, steps_per_epoch, decay_steps=1,
           decay_rate=0.9, warmup_steps=0., warmup_lr_init=0., global_epoch=0):
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    global_steps = steps_per_epoch * global_epoch
    self_warmup_delta = ((base_lr - warmup_lr_init) /
                         warmup_steps) if warmup_steps > 0 else 0
    self_decay_rate = decay_rate if decay_rate < 1 else 1 / decay_rate
    for i in range(total_steps):
        steps = math.floor(i / steps_per_epoch)
        cond = 1 if (steps < warmup_steps) else 0
        warmup_lr = warmup_lr_init + steps * self_warmup_delta
        decay_nums = math.floor(steps / decay_steps)
        decay_rate = math.pow(self_decay_rate, decay_nums)
        decay_lr = base_lr * decay_rate
        step_lr = cond * warmup_lr + (1 - cond) * decay_lr
        lr_each_step.append(step_lr)
    lr_each_step = lr_each_step[global_steps:]
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    return lr_each_step


def _create_net():
    if config.model == 'efficientnet_b0':
        return efficientnet_b0(num_classes=config.num_classes,
                               cfg=config,
                               drop_rate=config.drop,
                               drop_connect_rate=config.drop_connect,
                               global_pool=config.gp,
                               bn_tf=config.bn_tf,
                              )
    if config.model == 'efficientnet_b1':
        return efficientnet_b1(num_classes=config.num_classes,
                               cfg=config,
                               drop_rate=config.drop,
                               drop_connect_rate=config.drop_connect,
                               global_pool=config.gp,
                               bn_tf=config.bn_tf,
                              )
    raise NotImplementedError("This model currently not supported")


def _create_optim(net_, lr_):
    if config.opt == 'sgd':
        return SGD(net_.trainable_params(), learning_rate=lr_, momentum=config.momentum,
                   weight_decay=config.weight_decay,
                   loss_scale=config.loss_scale
                  )
    if config.opt == 'rmsprop':
        return RMSProp(net_.trainable_params(), learning_rate=lr_, decay=0.9,
                       weight_decay=config.weight_decay, momentum=config.momentum,
                       epsilon=config.opt_eps, loss_scale=config.loss_scale
                      )
    raise NotImplementedError("This optimizer currently not supported")


if __name__ == '__main__':
    print("\n=====> Loading...")
    print("\n===> Configuration:")
    print(config)

    local_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
    summary_dir = local_path + "/train/summary/"

    rank_id, rank_size = 0, 1
    context.set_context(mode=context.GRAPH_MODE)

    if config.platform == "GPU":
        dataset_sink_mode = True
        context.set_context(device_target='GPU', enable_graph_kernel=True)
    elif config.platform == "CPU":
        dataset_sink_mode = False
        context.set_context(device_target='CPU')
    else:
        raise NotImplementedError("Training only supported for CPU and GPU.")

    if config.distributed:
        if config.platform == "GPU":
            init("nccl")
        else:
            raise NotImplementedError("Distributed Training only supported for GPU.")
        context.reset_auto_parallel_context()
        rank_id = get_rank()
        rank_size = get_group_size()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True, device_num=rank_size)
        summary_dir += "thread_num_" + str(rank_id) + "/"

    print("\n===> Creating summary collector callback with path:")
    summary_cb = SummaryCollector(summary_dir)

    if config.model == 'efficientnet_b0':
        model_name = 'efficientnet_b0'
    elif config.model == 'efficientnet_b1':
        model_name = 'efficientnet_b1'
    else:
        raise NotImplementedError("This model currently not supported")

    dataset_type = config.dataset.lower()

    print(f"\n===> Creating {model_name} network...")

    net = _create_net()

    if dataset_type == 'imagenet':
        data_url = config.data_path
        train_data_url = data_url + '/train'
        val_data_url = data_url + '/val'
    elif dataset_type == 'cifar10':
        data_url = config.data_path
        train_data_url = data_url
        val_data_url = data_url

    print(f"\n===> Creating train dataset from path: {train_data_url} ...")

    train_dataset = create_dataset(
        dataset_type, model_name, train_data_url, config.batch_size,
        workers=config.workers, distributed=config.distributed)
    batches_per_epoch = train_dataset.get_dataset_size()
    print("Batches_per_epoch: ", batches_per_epoch)

    print(f"\n===> Creating validation dataset from path: {val_data_url} ...")

    val_dataset = create_dataset_val(
        dataset_type, model_name, val_data_url, config.batch_size,
        workers=config.workers, distributed=config.distributed)

    print("\n===> Creating Loss...")

    loss_cb = LossMonitor(per_print_times=1 if config.platform == "CPU" else batches_per_epoch)
    loss = LabelSmoothingCrossEntropy(smooth_factor=config.smoothing, num_classes=config.num_classes)
    time_cb = TimeMonitor(data_size=batches_per_epoch)
    loss_scale_manager = FixedLossScaleManager(
        config.loss_scale, drop_overflow_update=False)

    callbacks = [time_cb, loss_cb, summary_cb]

    print("\n===> Creating metrics for validation...")

    eval_metrics = {'Loss': nn.Loss(),
                    'Top1-Acc': nn.Top1CategoricalAccuracy(),
                    'Top5-Acc': nn.Top5CategoricalAccuracy()}

    if config.save_checkpoint:
        config_ck = CheckpointConfig(
            save_checkpoint_steps=batches_per_epoch, keep_checkpoint_max=config.keep_checkpoint_max)
        ckpoint_cb = ModelCheckpoint(
            prefix=config.model, directory='./ckpt_' + str(rank_id) + '/', config=config_ck)
        callbacks += [ckpoint_cb]

    lr = Tensor(get_lr(base_lr=config.lr, total_epochs=config.epochs, steps_per_epoch=batches_per_epoch,
                       decay_steps=config.decay_epochs, decay_rate=config.decay_rate,
                       warmup_steps=config.warmup_epochs, warmup_lr_init=config.warmup_lr_init,
                       global_epoch=config.resume_start_epoch))

    optimizer = _create_optim(net, lr)

    loss.add_flags_recursive(fp32=True, fp16=False)

    if config.resume:
        print("\n===> Resuming from checkpoint...")
        ckpt = load_checkpoint(config.resume)
        load_param_into_net(net, ckpt)

    print("\n===> Creating Model object...")

    model = Model(net, loss, optimizer,
                  loss_scale_manager=loss_scale_manager,
                  amp_level=config.amp_level,
                  metrics=eval_metrics
                  )

    eval_log = {"Epoch": [], "Val_Loss": [], "Val_Top1-Acc": [], "Val_Top5-Acc": []}
    eval_cb = EvalCallBack(model, val_dataset, eval_per_epoch=10, eval_log=eval_log)
    callbacks += [eval_cb]

    print("\n=====> Finished loading, starting training...\n\n")

    if config.resume:
        real_epoch = config.epochs - config.resume_start_epoch
        model.train(real_epoch, train_dataset,
                    callbacks=callbacks, dataset_sink_mode=dataset_sink_mode)
    else:
        model.train(config.epochs, train_dataset,
                    callbacks=callbacks, dataset_sink_mode=dataset_sink_mode)
