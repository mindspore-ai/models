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
"""train net."""
import os

from mindspore import context
from mindspore import Tensor
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.common import set_seed
from mindspore.parallel import set_algo_parameters
import mindspore.nn as nn
import mindspore.common.initializer as weight_init

from src.lr_generator import get_lr
from src.CrossEntropySmooth import CrossEntropySmooth
from src.model_utils.config import config
from src.dataset import create_dataset2 as create_dataset


set_seed(1)

if config.dataset != "imagenet2012":
    raise ValueError("Currently only support of imagenet2012 dataset format")

if config.net_name == "se-resnet50":
    from src.resnet import se_resnet50 as se_resnet
elif config.net_name == "se-resnet101":
    from src.resnet import se_resnet101 as se_resnet

if __name__ == '__main__':
    target = config.device_target
    ckpt_save_dir = config.checkpoint_path

    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
    if config.parameter_server:
        context.set_ps_context(enable_ps=True)
    if config.run_distribute:
        if target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(device_id=device_id)
            context.set_auto_parallel_context(device_num=config.device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            set_algo_parameters(elementwise_op_strategy_follow=True)
            init()
        elif target == "GPU":
            init('nccl')
            context.reset_auto_parallel_context()
            rank = get_rank()
            device_num = get_group_size()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
        if config.net_name in ("se-resnet50", "se-resnet101"):
            context.set_auto_parallel_context(all_reduce_fusion_config=[85, 160])
        else:
            context.set_auto_parallel_context(all_reduce_fusion_config=[180, 313])

    # create dataset
    dataset = create_dataset(dataset_path=config.data_path, do_train=True, repeat_num=1,
                             batch_size=config.batch_size, target=target, distribute=config.run_distribute)
    step_size = dataset.get_dataset_size()

    # define net
    net = se_resnet(class_num=config.class_num)
    if config.parameter_server:
        net.set_param_ps()

    # init weight
    if config.pre_trained:
        param_dict = load_checkpoint(config.pre_trained)
        load_param_into_net(net, param_dict)
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

    # init lr
    if config.net_name in ("se-resnet50", "se-resnet101"):
        lr = get_lr(lr_init=config.lr_init, lr_end=config.lr_end, lr_max=config.lr_max,
                    warmup_epochs=config.warmup_epochs, total_epochs=config.epoch_size, steps_per_epoch=step_size,
                    lr_decay_mode=config.lr_decay_mode)
    lr = Tensor(lr)

    # define opt
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': config.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]
    opt = Momentum(group_params, lr, config.momentum, loss_scale=config.loss_scale)
    # define loss, model
    if target in ["Ascend", "GPU"]:
        if config.dataset == "imagenet2012":
            if not config.use_label_smooth:
                config.label_smooth_factor = 0.0
            loss = CrossEntropySmooth(sparse=True, reduction="mean",
                                      smooth_factor=config.label_smooth_factor, num_classes=config.class_num)
        loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
        model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics={'acc'},
                      amp_level="O2", keep_batchnorm_fp32=False)

    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    if config.save_checkpoint:
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        if target == "GPU" and config.run_distribute:
            ckpt_save_dir = os.path.join(config.save_checkpoint_path, "ckpt_" + str(rank) + "/")
        ckpt_cb = ModelCheckpoint(prefix="resnet", directory=ckpt_save_dir, config=config_ck)
        cb += [ckpt_cb]

    # train model
    model.train(config.epoch_size - config.pretrain_epoch_size, dataset, callbacks=cb,
                sink_size=dataset.get_dataset_size(), dataset_sink_mode=True)
