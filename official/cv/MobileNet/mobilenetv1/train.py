# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""train mobilenet_v1."""
import os

import mindspore as ms
import mindspore.nn as nn
import mindspore.communication as comm
import mindspore.common.initializer as weight_init
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor

from src.lr_generator import get_lr
from src.CrossEntropySmooth import CrossEntropySmooth
from src.mobilenet_v1 import mobilenet_v1 as mobilenet
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper, modelarts_process
from src.model_utils.device_adapter import get_device_num


ms.set_seed(1)

if config.dataset == 'cifar10':
    from src.dataset import create_dataset1 as create_dataset
else:
    from src.dataset import create_dataset2 as create_dataset


def init_weigth(net):
    # init weight
    if config.pre_trained:
        param_dict = ms.load_checkpoint(config.pre_trained)
        ms.load_param_into_net(net, param_dict)
    else:
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(weight_init.initializer(weight_init.XavierUniform(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))


@moxing_wrapper(pre_process=modelarts_process)
def train_mobilenetv1():
    """ train_mobilenetv1 """
    if config.dataset == 'imagenet2012':
        config.dataset_path = os.path.join(config.dataset_path, 'train')
    target = config.device_target
    ckpt_save_dir = config.save_checkpoint_path

    # init context
    ms.set_context(mode=ms.GRAPH_MODE, device_target=target, save_graphs=False)

    # Set mempool block size in PYNATIVE_MODE for improving memory utilization, which will not take effect in GRAPH_MODE
    if ms.get_context("mode") == ms.PYNATIVE_MODE:
        ms.set_context(mempool_block_size="31GB")

    if config.parameter_server:
        ms.set_ps_context(enable_ps=True)
    device_id = int(os.getenv('DEVICE_ID', '0'))
    if config.run_distribute:
        if target == "Ascend":
            ms.set_context(device_id=device_id)
            ms.set_auto_parallel_context(device_num=get_device_num(), parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                                         gradients_mean=True)
            comm.init()
            ms.set_auto_parallel_context(all_reduce_fusion_config=[75])
        # GPU target
        else:
            comm.init()
            ms.set_auto_parallel_context(device_num=comm.get_group_size(), parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                                         gradients_mean=True)
        ckpt_save_dir = config.save_checkpoint_path + "ckpt_" + str(comm.get_rank()) + "/"

    # create dataset
    dataset = create_dataset(dataset_path=config.dataset_path, do_train=True, device_num=config.device_num,
                             batch_size=config.batch_size, target=target)
    step_size = dataset.get_dataset_size()

    # define net
    net = mobilenet(class_num=config.class_num)
    if config.parameter_server:
        net.set_param_ps()

    init_weigth(net)

    # init lr
    lr = get_lr(lr_init=config.lr_init, lr_end=config.lr_end, lr_max=config.lr_max,
                warmup_epochs=config.warmup_epochs, total_epochs=config.epoch_size, steps_per_epoch=step_size,
                lr_decay_mode=config.lr_decay_mode)
    lr = ms.Tensor(lr)

    # define opt
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    if target == "Ascend":
        group_params = [{'params': decayed_params, 'weight_decay': config.weight_decay},
                        {'params': no_decayed_params},
                        {'order_params': net.trainable_params()}]
        opt = nn.Momentum(group_params, lr, config.momentum, loss_scale=config.loss_scale)
    else:
        opt = nn.Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, config.momentum,
                          config.weight_decay)
    # define loss, model
    if config.dataset == "imagenet2012":
        if not config.use_label_smooth:
            config.label_smooth_factor = 0.0
        loss = CrossEntropySmooth(sparse=True, reduction="mean",
                                  smooth_factor=config.label_smooth_factor, num_classes=config.class_num)
    else:
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    loss_scale = ms.FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    if target == "Ascend":
        model = ms.Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics={'acc'},
                         amp_level="O2", keep_batchnorm_fp32=False)
    else:
        model = ms.Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})

    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    if config.save_checkpoint and device_id % min(8, get_device_num()) == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix="mobilenetv1", directory=ckpt_save_dir, config=config_ck)
        cb += [ckpt_cb]

    # train model
    model.train(config.epoch_size - config.pretrain_epoch_size, dataset, callbacks=cb,
                sink_size=dataset.get_dataset_size(), dataset_sink_mode=(not config.parameter_server))


if __name__ == '__main__':
    train_mobilenetv1()
