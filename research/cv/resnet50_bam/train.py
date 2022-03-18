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
Train.
"""
import argparse
import math
import os

import numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore import nn
from mindspore.common import dtype as mstype
from mindspore.common import set_seed
from mindspore.communication.management import get_group_size
from mindspore.communication.management import get_rank
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.nn.loss.loss import LossBase
from mindspore.nn.optim.momentum import Momentum
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import TimeMonitor
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint
from mindspore.train.serialization import load_param_into_net

import src.ResNet50_BAM as ResNet_BAM
from src.config import imagenet_cfg
from src.dataset import create_dataset_imagenet
from src.my_lossmonitor import MyLossMonitor

set_seed(1)


def lr_steps_imagenet(_cfg, steps_per_epoch):
    """lr step for imagenet"""
    if _cfg.lr_scheduler == 'cosine_annealing':
        _lr = warmup_cosine_annealing_lr(_cfg.lr_init,
                                         steps_per_epoch,
                                         _cfg.warmup_epochs,
                                         _cfg.epoch_size,
                                         _cfg.T_max,
                                         _cfg.eta_min)
    else:
        raise NotImplementedError(_cfg.lr_scheduler)

    return _lr


def linear_warmup_lr(current_step, warmup_steps, base_lr, init_lr):
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    lr1 = float(init_lr) + lr_inc * current_step
    return lr1


def warmup_cosine_annealing_lr(lr5, steps_per_epoch, warmup_epochs, max_epoch, T_max, eta_min=0):
    """ warmup cosine annealing lr"""
    base_lr = lr5
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    lr_each_step = []
    for i in range(total_steps):
        last_epoch = i // steps_per_epoch
        if i < warmup_steps:
            lr5 = linear_warmup_lr(i + 1, warmup_steps, base_lr, warmup_init_lr)
        else:
            lr5 = eta_min + (base_lr - eta_min) * (1. + math.cos(math.pi * last_epoch / T_max)) / 2
        lr_each_step.append(lr5)

    return np.array(lr_each_step).astype(np.float32)


class CrossEntropySmooth(LossBase):
    """CrossEntropy"""
    def __init__(self, sparse=True, reduction='mean', smooth_factor=0., num_classes=1000):
        super(CrossEntropySmooth, self).__init__()
        self.onehot = P.OneHot()
        self.sparse = sparse
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(1.0 * smooth_factor / (num_classes - 1), mstype.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)

    def construct(self, logit, label):
        if self.sparse:
            label = self.onehot(label, F.shape(logit)[1], self.on_value, self.off_value)
        loss2 = self.ce(logit, label)
        return loss2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification')
    parser.add_argument('--device_target', type=str, default=None, choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--device_id', type=int, default=None, help='device id of Ascend or GPU. (Default: None)')
    parser.add_argument('--device_num', type=int, default=1, help='number of devices. (Default: 1)')
    parser.add_argument('--lr_init', type=float, default=None, help='Learning rate.')

    parser.add_argument('--is_distributed', type=int, default=0,
                        help='Whether to use distributed GPU training. (Default: 0)')

    args_opt = parser.parse_args()
    cfg = imagenet_cfg

    if args_opt.lr_init is not None:
        cfg.lr_init = args_opt.lr_init

    # set context
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
    if args_opt.device_target is not None:
        cfg.device_target = args_opt.device_target
    if args_opt.device_target == 'Ascend':
        context.set_context(enable_graph_kernel=True)

        device_num = int(os.getenv('DEVICE_NUM', '1'))
        device_id = int(os.getenv('DEVICE_ID', '0'))

        if args_opt.device_id is not None:
            context.set_context(device_id=args_opt.device_id)
        else:
            context.set_context(device_id=cfg.device_id)

        if device_num > 1:
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=args_opt.device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()
    else:
        device_num = 1
        device_id = 0
        if args_opt.is_distributed:
            init()
            device_num = get_group_size()
            device_id = get_rank()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)

    dataset = create_dataset_imagenet(cfg.data_path, 1)

    batch_num = dataset.get_dataset_size()

    net = ResNet_BAM.ResidualNet("ImageNet", 50, cfg.num_classes, "BAM")
    # Continue training if set pre_trained to be True
    if cfg.pre_trained:
        param_dict = load_checkpoint(cfg.checkpoint_path)
        load_param_into_net(net, param_dict)

    loss_scale_manager = None

    lr = lr_steps_imagenet(cfg, batch_num)

    def get_param_groups(network):
        """ get param groups """
        decay_params = []
        no_decay_params = []
        for x in network.trainable_params():
            parameter_name = x.name
            if parameter_name.endswith('.bias'):
                # all bias not using weight decay
                no_decay_params.append(x)
            elif parameter_name.endswith('.gamma'):
                # bn weight bias not using weight decay, be carefully for now x not include BN
                no_decay_params.append(x)
            elif parameter_name.endswith('.beta'):
                # bn weight bias not using weight decay, be carefully for now x not include BN
                no_decay_params.append(x)
            else:
                decay_params.append(x)

        return [{'params': no_decay_params, 'weight_decay': 0.0}, {'params': decay_params}]

    if cfg.is_dynamic_loss_scale:
        cfg.loss_scale = 1

    opt = Momentum(params=get_param_groups(net),
                   learning_rate=Tensor(lr),
                   momentum=cfg.momentum,
                   weight_decay=cfg.weight_decay,
                   loss_scale=cfg.loss_scale)
    if not cfg.use_label_smooth:
        cfg.label_smooth_factor = 0.0
    loss = CrossEntropySmooth(sparse=True, reduction="mean",
                              smooth_factor=cfg.label_smooth_factor, num_classes=cfg.num_classes)

    if args_opt.device_target == 'Ascend':
        if cfg.is_dynamic_loss_scale == 1:
            loss_scale_manager = DynamicLossScaleManager(init_loss_scale=65536, scale_factor=2, scale_window=2000)
        else:
            loss_scale_manager = FixedLossScaleManager(cfg.loss_scale, drop_overflow_update=False)
    else:
        loss_scale_manager = FixedLossScaleManager(cfg.loss_scale, drop_overflow_update=False)

    model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'},
                  amp_level="O3", keep_batchnorm_fp32=False,
                  loss_scale_manager=loss_scale_manager,
                  )

    config_ck = CheckpointConfig(save_checkpoint_steps=batch_num * 2, keep_checkpoint_max=cfg.keep_checkpoint_max)
    time_cb = TimeMonitor(data_size=batch_num)
    ckpt_save_dir = "./ckpt/"
    ckpoint_cb = ModelCheckpoint(prefix="train_resnet50_bam_imagenet", directory=ckpt_save_dir,
                                 config=config_ck)
    loss_cb = MyLossMonitor(per_print_times=1 if cfg.use_dataset_sink else 100)
    cbs = [time_cb, ckpoint_cb, loss_cb]
    if device_num > 1 and device_id != 0:
        cbs = [time_cb, loss_cb]
    model.train(cfg.epoch_size, dataset, callbacks=cbs, dataset_sink_mode=cfg.use_dataset_sink)
    print("train success")
