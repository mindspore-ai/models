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
import time

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, RunContext
from mindspore.communication.management import init, get_rank

from src.lr_generator import get_lr
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_rank_id
from src.dataset import create_dataset_cifar10

ms.set_seed(1)


def set_parameter():
    """set_parameter"""
    if config.mode_name == 'GRAPH':
        ms.set_context(mode=ms.GRAPH_MODE, device_target=config.device_target, save_graphs=config.save_graphs)
    else:
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target=config.device_target, save_graphs=config.save_graphs)
    if config.run_distribute:
        device_id = int(os.getenv('DEVICE_ID'))
        device_num = int(os.getenv('RANK_SIZE'))
        ms.set_context(device_id=device_id)
        ms.set_auto_parallel_context(device_num=device_num, parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                                     gradients_mean=True)
        if config.all_reduce_fusion_config:
            ms.set_auto_parallel_context(all_reduce_fusion_config=config.all_reduce_fusion_config)

        init()
    else:
        ms.set_context(device_id=config.device_id)


def init_weight(net):
    """init_weight"""
    for _, cell in net.cells_and_names():
        if isinstance(cell, nn.Conv2d):
            if config.conv_init == "XavierUniform":
                cell.weight.set_data(ms.common.initializer.initializer(ms.common.initializer.XavierUniform(),
                                                                       cell.weight.shape,
                                                                       cell.weight.dtype))
        if isinstance(cell, nn.Dense):
            if config.dense_init == "TruncatedNormal":
                cell.weight.set_data(ms.common.initializer.initializer(ms.common.initializer.TruncatedNormal(),
                                                                       cell.weight.shape,
                                                                       cell.weight.dtype))


def init_loss_scale():
    """init loss scale"""
    if config.loss_function == "SoftmaxCrossEntropy":
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    elif config.loss_function == "MSE":
        loss = nn.MSELoss(reduction='mean')
    return loss


def init_group_params(net):
    """init group params"""
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
    if config.enable_modelarts and config.run_distribute:
        ckpt_save_dir = ckpt_save_dir + "ckpt_" + str(get_rank_id()) + "/"
    else:
        if config.run_distribute:
            ckpt_save_dir = ckpt_save_dir + "ckpt_" + str(get_rank()) + "/"
    return ckpt_save_dir


class InternalCallbackParam(dict):
    """Internal callback object's parameters."""

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', tb_writer=None):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.tb_writer = tb_writer
        self.cur_step = 1
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(self.name, self.val, self.cur_step)
        self.cur_step += 1

    def __str__(self):
        fmtstr = '{name}:{avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


def snn_model_build():
    """build snn model for resnet50 and lenet"""
    if config.net_name == "resnet50":
        from src.snn_resnet import snn_resnet50
        net = snn_resnet50(class_num=config.class_num)
        init_weight(net=net)
    elif config.net_name == "lenet":
        from src.snn_lenet import snn_lenet
        net = snn_lenet(num_class=config.class_num)
    else:
        raise ValueError(f'config.model: {config.model_name} is not supported')
    return net


@moxing_wrapper()
def train_net():
    """train net: resnet50_snn or lenet_snn"""
    set_parameter()
    dataset = create_dataset_cifar10(data_path=config.data_path, do_train=True, batch_size=config.batch_size)
    step_size = dataset.get_dataset_size()
    net = snn_model_build()

    if config.pre_trained:
        ckpt_param_dict = ms.load_checkpoint(config.pre_trained)
        ms.load_param_into_net(net, ckpt_param_dict)

    # define opt
    if config.optimizer == "Momentum":
        lr = get_lr(lr_init=config.lr_init, lr_end=config.lr_end, lr_max=config.lr_max,
                    warmup_epochs=config.warmup_epochs, total_epochs=config.epoch_size, steps_per_epoch=step_size,
                    lr_decay_mode=config.lr_decay_mode)
        lr = ms.Tensor(lr)
        group_params = init_group_params(net)
        opt = nn.Momentum(group_params, lr, config.momentum, loss_scale=config.loss_scale)
    elif config.optimizer == "Adam":
        opt = nn.Adam(net.trainable_params(), config.lr_init, loss_scale=config.loss_scale)

    loss = init_loss_scale()
    network_with_loss = nn.WithLossCell(net, loss)
    network_train = nn.TrainOneStepCell(network_with_loss, opt, sens=config.loss_scale)
    network_train.set_train(True)
    loss_meter = AverageMeter('loss')

    # define callbacks
    ckpt_save_dir = set_save_ckpt_dir()

    if config.save_checkpoint:
        cb_params = InternalCallbackParam()
        cb_params.train_network = network_train
        cb_params.epoch_num = config.epoch_size
        cb_params.cur_epoch_num = 1
        run_context = RunContext(cb_params)
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix=config.net_name, directory=ckpt_save_dir, config=config_ck)
        ckpt_cb.begin(run_context)

    dataset_size = dataset.get_dataset_size()
    first_step = True
    print("Start train resnet, the first epoch will be slower because of the graph compilation.", flush=True)
    t_end = time.time()
    for epoch_idx in range(config.epoch_size):
        for step_idx, data in enumerate(dataset.create_dict_iterator()):
            images = data["image"]
            label = data["label"]
            if config.loss_function == "MSE":
                onehot = ops.OneHot()
                label = onehot(label, config.class_num, Tensor(1.0, ms.float32), Tensor(0.0, ms.float32))
            loss = network_train(images, label)
            loss_meter.update(loss.asnumpy())

            if config.save_checkpoint:
                cb_params.cur_epoch_num = epoch_idx + 1
                cb_params.cur_step_num = step_idx + 1 + epoch_idx * dataset_size
                cb_params.batch_num = dataset_size
                ckpt_cb.step_end(run_context)

        time_used = (time.time() - t_end) * 1000
        if first_step:
            per_step_time = time_used
            first_step = False
        else:
            per_step_time = time_used / dataset_size
        print('epoch: {}, step: {}, loss is {}, epoch time: {:.3f}ms, per step time: {:.3f}ms'.format(
            epoch_idx + 1, dataset_size, loss_meter, time_used, per_step_time), flush=True)
        t_end = time.time()
        loss_meter.reset()


if __name__ == '__main__':
    train_net()
