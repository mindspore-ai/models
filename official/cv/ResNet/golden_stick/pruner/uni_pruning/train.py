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
from mindspore import context, nn
from mindspore.train.model import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.communication.management import init, get_rank
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore_gs.pruner.uni_pruning import UniPruner

from src.lr_generator import get_lr
from src.CrossEntropySmooth import CrossEntropySmooth
from src.metric import DistAccuracy
from src.resnet import conv_variance_scaling_initializer
from src.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from src.model_utils.config import config

if config.dataset == "cifar10":
    from src.dataset import create_dataset1 as create_dataset
else:
    if config.mode_name == "GRAPH":
        from src.dataset import create_dataset2 as create_dataset
    else:
        from src.dataset import create_dataset_pynative as create_dataset

ms.set_seed(1)


def filter_checkpoint_parameter_by_list(origin_dict, param_filter):
    """remove useless parameters according to filter_list"""
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del origin_dict[key]
                break


def init_env(args):
    """set context and training mode"""
    rank = 0
    device_num = 1
    if args.mode_name == 'GRAPH' and args.device_target == "GPU":
        print('GPU GRAPH MODE')
        context.set_context(mode=context.GRAPH_MODE,
                            device_target=args.device_target, device_id=args.device_id)
        if args.device_num > 1:
            context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL,
                                              gradients_mean=True)
            init("nccl")
            rank = get_rank()
            device_num = args.device_num

    elif args.mode_name == 'GRAPH' and args.device_target == 'Ascend':
        print('Ascend GRAPH MODE')
        device_num = int(os.getenv('RANK_SIZE'))
        device_id = int(os.getenv('DEVICE_ID'))
        rank = int(os.getenv('RANK_ID'))
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
        context.set_context(max_call_depth=2000)
        if device_num > 1:
            os.environ['MINDSPORE_HCCL_CONFIG_PATH'] = os.getenv('RANK_TABLE_FILE')
        context.set_context(device_id=device_id)
        if device_num > 1:
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()
    else:
        print(f'Single node pynative mode on {args.device_target}')
        context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target,
                            device_id=args.device_id)

    return rank


def init_weight(net):
    """init_weight"""
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


def init_group_params(net):
    """split decayed and not decayed params"""
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
    if config.dataset in ["imagenet2012", "cifar10"]:
        if not config.use_label_smooth:
            config.label_smooth_factor = 0.0
        loss = CrossEntropySmooth(sparse=True, reduction="mean",
                                  smooth_factor=config.label_smooth_factor, num_classes=config.class_num)
    else:
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    return loss


def load_pretrained_ckpt(net):
    """load checkpoint"""
    if config.pre_trained:
        if os.path.isfile(config.pre_trained):
            ckpt = ms.load_checkpoint(config.pre_trained)
            if ckpt.get("epoch_num") and ckpt.get("step_num"):
                config.has_trained_epoch = int(ckpt["epoch_num"].data.asnumpy())
                config.has_trained_step = int(ckpt["step_num"].data.asnumpy())
            else:
                config.has_trained_epoch = 0
                config.has_trained_step = 0

            if config.filter_weight:
                filter_list = [x.name for x in net.end_point.get_parameters()]
                filter_checkpoint_parameter_by_list(ckpt, filter_list)
            ms.load_param_into_net(net, ckpt)
        else:
            print(f"Invalid pre_trained {config.pre_trained} parameter.")


def train_net():
    """train net"""
    print(f"Train configure: {config}")
    target = config.device_target
    rank = init_env(config)
    dataset = create_dataset(dataset_path=config.data_path, do_train=True,
                             batch_size=config.batch_size, train_image_size=config.train_image_size,
                             eval_image_size=config.eval_image_size, target=target,
                             distribute=config.run_distribute)
    step_size = dataset.get_dataset_size()
    if config.net_name == 'resnet18':
        net = resnet18(class_num=config.class_num)
    elif config.net_name == 'resnet34':
        net = resnet34(class_num=config.class_num)
    elif config.net_name == 'resnet50':
        net = resnet50(class_num=config.class_num)
    elif config.net_name == 'resnet101':
        net = resnet101(class_num=config.class_num)
    elif config.net_name == 'resnet152':
        net = resnet152(class_num=config.class_num)
    init_weight(net)
    input_size = [config.export_batch_size, 3, config.height, config.width]
    algo = UniPruner({"exp_name": config.exp_name,
                      "frequency": config.retrain_epochs,
                      "target_sparsity": 1 - config.prune_rate,
                      "pruning_step": config.pruning_step,
                      "filter_lower_threshold": config.filter_lower_threshold,
                      "input_size": input_size,
                      "output_path": config.output_path,
                      "prune_flag": config.prune_flag,
                      "rank": rank,
                      "device_target": config.device_target})
    algo.apply(net)
    load_pretrained_ckpt(net)

    lr = get_lr(lr_init=config.lr_init,
                lr_end=config.lr_end,
                lr_max=config.lr_max,
                warmup_epochs=config.warmup_epochs,
                total_epochs=config.epoch_size,
                start_epoch=config.start_epoch,
                steps_per_epoch=step_size,
                lr_decay_mode='cosine')

    if config.pre_trained:
        lr = lr[config.has_trained_epoch * step_size:]

    lr = ms.Tensor(lr)
    # define optimizer
    group_params = init_group_params(net)
    if config.optimizer == 'Momentum':
        opt = nn.Momentum(group_params, lr, config.momentum, loss_scale=config.loss_scale)

    loss = init_loss_scale()
    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)

    metrics = {"acc"}
    if config.run_distribute:
        metrics = {'acc': DistAccuracy(batch_size=config.batch_size, device_num=config.device_num)}
    model = ms.Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics=metrics,
                     amp_level="O2", boost_level="O0", keep_batchnorm_fp32=False)

    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()

    cb = [time_cb, loss_cb]
    print('UniPruning enabled')
    algo_cb = algo.callbacks()[0]
    cb.append(algo_cb)

    ckpt_save_dir = set_save_ckpt_dir()
    if config.save_checkpoint:
        ckpt_append_info = [{"epoch_num": config.has_trained_epoch, "step_num": config.has_trained_step}]
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max,
                                     append_info=ckpt_append_info)
        ckpt_cb = ModelCheckpoint(prefix="resnet", directory=ckpt_save_dir, config=config_ck)
        cb += [ckpt_cb]
    # train model
    dataset_sink_mode = target != "CPU"
    print(dataset.get_dataset_size())
    model.train(config.epoch_size - config.has_trained_epoch, dataset, callbacks=cb,
                sink_size=dataset.get_dataset_size(), dataset_sink_mode=dataset_sink_mode)


if __name__ == '__main__':
    train_net()
