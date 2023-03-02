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

import mindspore as ms
import mindspore.nn as nn
import mindspore.common.initializer as weight_init
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.communication.management import init, get_rank
from mindspore.parallel import set_algo_parameters
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore_gs import GhostAlgo
from src.lr_generator import get_lr
from src.CrossEntropySmooth import CrossEntropySmooth
from src.resnet import resnet50 as resnet
from src.model_utils.config import config

if config.dataset == "cifar10":
    from src.dataset import create_dataset1 as create_dataset
else:
    from src.dataset import create_dataset2 as create_dataset

ms.set_seed(1)


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
                cell.weight.set_data(weight_init.initializer(weight_init.HeUniform(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(weight_init.initializer(weight_init.HeNormal(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))


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
    init_weight(net=net, param_dict=None)
    # apply golden-stick algo
    algo = GhostAlgo({})
    net = algo.apply(net)
    lr = get_lr(lr_init=config.lr_init,
                lr_end=config.lr_end,
                lr_max=config.lr_max,
                warmup_epochs=config.warmup_epochs,
                total_epochs=config.epoch_size,
                start_epoch=config.start_epoch,
                steps_per_epoch=step_size,
                lr_decay_mode=config.lr_decay_mode)

    group_params = init_group_params(net)

    optimizer = nn.Momentum(group_params,
                            learning_rate=lr,
                            momentum=config.momentum,
                            loss_scale=config.loss_scale
                            )
    kf_loss_fn = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    loss_scale = ms.FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    ckpt_save_dir = set_save_ckpt_dir()
    config_ck = CheckpointConfig(save_checkpoint_steps=5 * step_size,
                                 keep_checkpoint_max=config.keep_checkpoint_max)
    ckpt_cb = ModelCheckpoint(prefix="resnet", directory=ckpt_save_dir,
                              config=config_ck)
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    metrics = {"acc"}
    cb = [loss_cb, time_cb, ckpt_cb]
    model = ms.Model(net, loss_fn=kf_loss_fn, optimizer=optimizer, loss_scale_manager=loss_scale, metrics=metrics,
                     boost_level=config.boost_mode,
                     boost_config_dict={"grad_freeze": {"total_steps": config.epoch_size * step_size}})
    model.train(config.epoch_size, dataset, callbacks=cb, dataset_sink_mode=True)


if __name__ == '__main__':
    train_net()
