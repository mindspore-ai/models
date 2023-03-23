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
"""train resnet."""
import os
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.communication.management import init, get_rank
from mindspore.parallel import set_algo_parameters
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore_gs import PrunerKfCompressAlgo, PrunerFtCompressAlgo
from src.lr_generator import get_lr
from src.CrossEntropySmooth import CrossEntropySmooth
from src.resnet import conv_variance_scaling_initializer
from src.resnet import resnet50 as resnet
from src.model_utils.config import config

if config.dataset == "cifar10":
    from src.dataset import create_dataset1 as create_dataset
else:
    from src.dataset import create_dataset2 as create_dataset

ms.set_seed(1)


class NetWithLossCell(nn.WithLossCell):
    """Calculate NetWithLossCell."""

    def __init__(self, backbone, loss_fn, ngpu):
        super(NetWithLossCell, self).__init__(backbone, loss_fn)
        self.ngpu = ngpu

    def construct(self, data, label):
        num_pgpu = data.shape[0] // 2 * self.ngpu
        out = self._backbone(data)
        output_list = []
        for igpu in range(self.ngpu):
            output_list.append(out[igpu * num_pgpu * 2:igpu * num_pgpu * 2 + num_pgpu])
        out = ops.Concat(axis=0)(output_list)
        return self._loss_fn(out, label)


def filter_checkpoint_parameter_by_list(origin_dict, param_filter):
    """remove useless parameters according to filter_list"""
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del origin_dict[key]
                break


def set_parameter():
    """set_parameter"""
    target = config.device_target
    if target == "CPU":
        config.run_distribute = False

    # init context
    if config.mode_name == "GRAPH":
        ms.set_context(mode=ms.GRAPH_MODE, device_target=target, save_graphs=False)
    else:
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target=target, save_graphs=False)

    if config.run_distribute:
        if target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            ms.set_context(device_id=device_id)
            ms.set_auto_parallel_context(device_num=config.device_num, parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                                         gradients_mean=True)
            set_algo_parameters(elementwise_op_strategy_follow=True)
            if config.boost_mode not in ["O1", "O2"]:
                ms.set_auto_parallel_context(all_reduce_fusion_config=config.all_reduce_fusion_config)
            init()
        else:
            # GPU target
            init()
            ms.set_auto_parallel_context(device_num=config.device_num,
                                         parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                                         gradients_mean=True)
            ms.set_auto_parallel_context(all_reduce_fusion_config=config.all_reduce_fusion_config)


def init_weight(net, param_dict):
    """init_weight"""
    if config.pre_trained and param_dict:
        if param_dict.get("epoch_num") and param_dict.get("step_num"):
            config.has_trained_epoch = int(param_dict["epoch_num"].data.asnumpy())
            config.has_trained_step = int(param_dict["step_num"].data.asnumpy())
        else:
            config.has_trained_epoch = 0
            config.has_trained_step = 0

        if config.filter_weight:
            filter_list = [x.name for x in net.end_point.get_parameters()]
            filter_checkpoint_parameter_by_list(param_dict, filter_list)
        ms.load_param_into_net(net, param_dict)
    else:
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                if config.conv_init == "XavierUniform":
                    cell.weight.set_data(ms.common.initializer.initializer(ms.common.initializer.XavierUniform(),
                                                                           cell.weight.shape,
                                                                           cell.weight.dtype))
                elif config.conv_init == "TruncatedNormal":
                    weight = conv_variance_scaling_initializer(cell.in_channels,
                                                               cell.out_channels,
                                                               cell.kernel_size[0])
                    cell.weight.set_data(weight)
            if isinstance(cell, nn.Dense):
                if config.dense_init == "TruncatedNormal":
                    cell.weight.set_data(ms.common.initializer.initializer(ms.common.initializer.TruncatedNormal(),
                                                                           cell.weight.shape,
                                                                           cell.weight.dtype))
                elif config.dense_init == "RandomNormal":
                    in_channel = cell.in_channels
                    out_channel = cell.out_channels
                    weight = np.random.normal(loc=0, scale=0.01, size=out_channel * in_channel)
                    weight = ms.Tensor(np.reshape(weight, (out_channel, in_channel)), dtype=cell.weight.dtype)
                    cell.weight.set_data(weight)


def load_fp32_ckpt(net):
    if config.fp32_ckpt:
        if os.path.isfile(config.fp32_ckpt):
            ckpt = ms.load_checkpoint(config.fp32_ckpt)
            if config.filter_weight:
                filter_list = [x.name for x in net.end_point.get_parameters()]
                filter_checkpoint_parameter_by_list(ckpt, filter_list)
            ms.load_param_into_net(net, ckpt)
        else:
            print(f"Invalid fp32_ckpt {config.fp32_ckpt} parameter.")


def load_pretrained_ckpt(net):
    if config.pre_trained:
        if os.path.isfile(config.pre_trained):
            ckpt = ms.load_checkpoint(config.pre_trained)
            if ckpt.get("epoch_num") and ckpt.get("step_num"):
                config.has_trained_epoch = int(ckpt["epoch_num"].data.asnumpy())
                config.has_trained_step = int(ckpt["step_num"].data.asnumpy())
            else:
                config.has_trained_epoch = 0
                config.has_trained_step = 0

            if config.has_trained_epoch > config.epoch_size:
                raise RuntimeError("If retrain, epoch_size should be bigger than has_trained_epoch after "
                                   "loading pretrained weight, but got epoch_size {}, has_trained_epoch {}"
                                   "".format(config.epoch_size, config.has_trained_epoch))

            if config.filter_weight:
                filter_list = [x.name for x in net.end_point.get_parameters()]
                filter_checkpoint_parameter_by_list(ckpt, filter_list)
            not_load_param, _ = ms.load_param_into_net(net, ckpt)
            if not_load_param:
                raise RuntimeError("Load param into net fail.")
        else:
            raise RuntimeError("Pretrained ckpt file {} does not exist.".format(config.pre_trained))


def init_group_params(net):
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
    return group_params


def set_save_ckpt_dir():
    """set save ckpt dir"""
    ckpt_save_dir = os.path.join(config.output_path, config.checkpoint_path)
    if config.run_distribute:
        ckpt_save_dir = ckpt_save_dir + "ckpt_" + str(get_rank()) + "/"
    return ckpt_save_dir


def init_loss_scale():
    if config.dataset == "imagenet2012":
        if not config.use_label_smooth:
            config.label_smooth_factor = 0.0
        loss = CrossEntropySmooth(sparse=True, reduction="mean",
                                  smooth_factor=config.label_smooth_factor, num_classes=config.class_num)
    else:
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    return loss


def train_net():
    """train net"""
    print("Train configure: {}".format(config))
    target = config.device_target
    set_parameter()
    dataset = create_dataset(dataset_path=config.data_path, do_train=True,
                             batch_size=config.batch_size, train_image_size=config.train_image_size,
                             eval_image_size=config.eval_image_size, target=target,
                             distribute=config.run_distribute)
    step_size = dataset.get_dataset_size()
    net = resnet(class_num=config.class_num)

    # apply golden-stick algo
    algo_kf = PrunerKfCompressAlgo({})
    load_fp32_ckpt(net)
    net = algo_kf.apply(net)
    lr = get_lr(lr_init=config.lr_init,
                lr_end=0.0,
                lr_max=config.lr_max_kf,
                warmup_epochs=config.warmup_epochs,
                start_epoch=config.start_epoch,
                total_epochs=config.epoch_kf,
                steps_per_epoch=step_size,
                lr_decay_mode='cosine')

    optimizer = nn.Momentum(filter(lambda p: p.requires_grad, net.get_parameters()),
                            learning_rate=lr,
                            momentum=config.momentum
                            )

    kf_loss_fn = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    algo_cb_list = algo_kf.callbacks()
    cb = [loss_cb, time_cb]
    cb += algo_cb_list
    if config.pre_trained:
        train_ft(net)
    else:
        model = ms.Model(net, loss_fn=kf_loss_fn, optimizer=optimizer)
        model.train(config.epoch_kf, dataset, callbacks=cb, dataset_sink_mode=False)
        train_ft(net)


def train_ft(net):
    """train finetune."""
    dataset = create_dataset(dataset_path=config.data_path, do_train=True,
                             batch_size=config.batch_size, train_image_size=config.train_image_size,
                             eval_image_size=config.eval_image_size, target=config.device_target,
                             distribute=config.run_distribute)
    algo_ft = PrunerFtCompressAlgo({'prune_rate': config.prune_rate})
    net = algo_ft.apply(net)
    load_pretrained_ckpt(net)
    lr_ft_new = ms.Tensor(get_lr(lr_init=config.lr_init,
                                 lr_end=config.lr_end_ft,
                                 lr_max=config.lr_max_ft,
                                 warmup_epochs=config.warmup_epochs,
                                 total_epochs=config.epoch_ft,
                                 start_epoch=config.start_epoch,
                                 steps_per_epoch=dataset.get_dataset_size(),
                                 lr_decay_mode='poly'))

    optimizer_ft = nn.Momentum(filter(lambda p: p.requires_grad, net.get_parameters()),
                               learning_rate=lr_ft_new,
                               momentum=config.momentum,
                               loss_scale=config.loss_scale
                               )
    net.set_train()
    metrics = {"acc"}
    loss_scale = FixedLossScaleManager(1024, drop_overflow_update=False)
    ft_loss_fn = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    model_ft = ms.Model(net, loss_fn=ft_loss_fn, optimizer=optimizer_ft, loss_scale_manager=loss_scale,
                        metrics=metrics,
                        amp_level="O2", boost_level="O0", keep_batchnorm_fp32=False)

    step_size = dataset.get_dataset_size()

    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    ckpt_save_dir = set_save_ckpt_dir()
    config_ck = CheckpointConfig(save_checkpoint_steps=5 * step_size,
                                 keep_checkpoint_max=config.keep_checkpoint_max)
    ckpt_cb = ModelCheckpoint(prefix="resnet", directory=ckpt_save_dir,
                              config=config_ck)
    ft_cb = [time_cb, loss_cb, ckpt_cb]

    model_ft.train(config.epoch_ft, dataset, callbacks=ft_cb,
                   sink_size=dataset.get_dataset_size(), dataset_sink_mode=True)


if __name__ == '__main__':
    train_net()
