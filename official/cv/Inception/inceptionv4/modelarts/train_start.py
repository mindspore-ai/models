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
"""train imagenet"""
import math
import os
import argparse
import glob
import moxing as mox
import numpy as np

from mindspore import Model
from mindspore import Tensor
from mindspore import context
from mindspore.common import set_seed
from mindspore.common.initializer import XavierUniform, initializer
from mindspore.communication import init, get_rank, get_group_size
from mindspore.nn import RMSProp
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor, LossMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.model import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import export

from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_id, get_device_num
from src.dataset import create_dataset_imagenet, create_dataset_cifar10
from src.inceptionv4 import Inceptionv4


os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

DS_DICT = {
    "imagenet": create_dataset_imagenet,
    "cifar10": create_dataset_cifar10,
}

config.device_id = get_device_id()
config.device_num = get_device_num()
device_num = config.device_num
create_dataset = DS_DICT[config.ds_type]

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument("--filter_weight", type=str, default=True,
                    help="Filter head weight parameters, default is False.")
parser.add_argument('--data_url',
                    metavar='DIR',
                    default='/cache/data_url',
                    help='path to dataset')
parser.add_argument('--train_url',
                    default="/cache/output/",
                    type=str,
                    help="setting dir of training output")
parser.add_argument('--resume',
                    default="",
                    type=str,
                    help="resume training with existed checkpoint")
parser.add_argument('--ds_type',
                    default="imagenet",
                    type=str,
                    help="dataset type, imagenet or cifar10")
parser.add_argument('--num_classes',
                    default="1000",
                    type=int,
                    help="classes")
parser.add_argument('--epoch_size',
                    default="250",
                    type=int,
                    help="epoch_size")
parser.add_argument('--batch_size',
                    default="128",
                    type=int,
                    help="batch_size")

args_opt = parser.parse_args()

set_seed(1)


def generate_cosine_lr(steps_per_epoch, total_epochs,
                       lr_init=config.lr_init,
                       lr_end=config.lr_end,
                       lr_max=config.lr_max,
                       warmup_epochs=config.warmup_epochs):
    """
    Applies cosine decay to generate learning rate array.

    Args:
       steps_per_epoch(int): steps number per epoch
       total_epochs(int): all epoch in training.
       lr_init(float): init learning rate.
       lr_end(float): end learning rate
       lr_max(float): max learning rate.
       warmup_steps(int): all steps in warmup epochs.

    Returns:
       np.array, learning rate array.
    """
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    decay_steps = total_steps - warmup_steps
    lr_each_step = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr_inc = (float(lr_max) - float(lr_init)) / float(warmup_steps)
            lr = float(lr_init) + lr_inc * (i + 1)
        else:
            cosine_decay = 0.5 * (1 + math.cos(math.pi * (i - warmup_steps) / decay_steps))
            lr = (lr_max - lr_end) * cosine_decay + lr_end
        lr_each_step.append(lr)
    learning_rate = np.array(lr_each_step).astype(np.float32)
    current_step = steps_per_epoch * (config.start_epoch - 1)
    learning_rate = learning_rate[current_step:]
    return learning_rate


def filter_checkpoint_parameter_by_list(origin_dict, param_filter):
    """remove useless parameters according to filter_list"""
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del origin_dict[key]
                break


def frozen_to_air(network, args):
    param_dict_t = load_checkpoint(args.get("ckpt_file"))
    load_param_into_net(network, param_dict_t)
    input_arr = Tensor(np.zeros([args.get("batch_size"), 3, args.get("height"), args.get("width")], np.float32))
    export(network, input_arr, file_name=args.get("file_name"), file_format=args.get("file_format"))


if __name__ == '__main__':
    if not os.path.exists(config.data_path):
        os.makedirs(config.data_path, exist_ok=True)
    if not os.path.exists(config.load_path):
        os.makedirs(config.load_path, exist_ok=True)
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path, exist_ok=True)
    mox.file.copy_parallel(args_opt.data_url, config.data_path)

    config.resume = args_opt.resume
    config.ds_type = args_opt.ds_type
    config.class_num = args_opt.num_classes
    config.epoch_size = args_opt.epoch_size
    config.batch_size = args_opt.batch_size
    config.resume = os.path.join(config.dataset_path, config.resume)
    config.dataset_path = os.path.join(config.dataset_path, "train")

    print('epoch_size: {} batch_size: {} class_num {}'.format(config.epoch_size, config.batch_size, config.num_classes))

    context.set_context(mode=context.GRAPH_MODE, device_target=config.platform)
    if config.platform == "Ascend":
        context.set_context(device_id=get_device_id())
        context.set_context(enable_graph_kernel=False)

    if device_num > 1:
        init()
        config.rank = get_rank()
        config.group_size = get_group_size()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True,
                                          all_reduce_fusion_config=[200, 400])
    else:
        config.rank = 0
        config.group_size = 1

    # create dataset
    train_dataset = create_dataset(dataset_path=config.dataset_path, do_train=True, cfg=config)
    train_step_size = train_dataset.get_dataset_size()
    print("print(train_step_size):", train_step_size)
    # create model
    net = Inceptionv4(classes=config.num_classes)
    # loss
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    # learning rate
    lr_t = Tensor(generate_cosine_lr(steps_per_epoch=train_step_size, total_epochs=config.epoch_size))

    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            param.set_data(initializer(XavierUniform(), param.data.shape, param.data.dtype))
    group_params = [{'params': decayed_params, 'weight_decay': config.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]

    opt = RMSProp(group_params, lr_t, decay=config.decay, epsilon=config.epsilon, weight_decay=config.weight_decay,
                  momentum=config.momentum, loss_scale=config.loss_scale)

    if get_device_id() == 0:
        print(lr_t)
        print(train_step_size)

    if args_opt.resume:
        param_dict = load_checkpoint(config.resume)
        if args_opt.filter_weight:
            filter_list = [x.name for x in net.softmax.get_parameters()]
            filter_checkpoint_parameter_by_list(param_dict, filter_list)
        load_param_into_net(net, param_dict)

    loss_scale_manager = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)

    model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc', 'top_1_accuracy', 'top_5_accuracy'},
                  loss_scale_manager=loss_scale_manager, amp_level=config.amp_level)

    # define callbacks
    performance_cb = TimeMonitor(data_size=train_step_size)
    loss_cb = LossMonitor(per_print_times=train_step_size)
    ckp_save_step = config.save_checkpoint_epochs * train_step_size
    config_ck = CheckpointConfig(save_checkpoint_steps=ckp_save_step, keep_checkpoint_max=config.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix=f"inceptionV4-train-rank{config.rank}",
                                 directory=config.output_path, config=config_ck)
    callbacks = [performance_cb, loss_cb]
    if device_num > 1 and config.is_save_on_master and get_device_id() == 0:
        callbacks.append(ckpoint_cb)
    else:
        callbacks.append(ckpoint_cb)

    # train model
    model.train(config.epoch_size, train_dataset, callbacks=callbacks, dataset_sink_mode=config.ds_sink_mode)

    ckpt_list = glob.glob(config.output_path + "/inceptionV4*.ckpt")
    if not ckpt_list:
        print("ckpt file not generated.")

    ckpt_list.sort(key=os.path.getmtime)
    ckpt_model = ckpt_list[-1]
    print("checkpoint path", ckpt_model)

    net = Inceptionv4(classes=config.num_classes)

    frozen_to_air_args = {'ckpt_file': ckpt_model,
                          'batch_size': 1,
                          'height': 299,
                          'width': 299,
                          'file_name': config.output_path + '/inceptionV4',
                          'file_format': 'AIR'}
    frozen_to_air(net, frozen_to_air_args)

    mox.file.copy_parallel(config.output_path, args_opt.train_url)
    print('Inceptionv4 training success!')
