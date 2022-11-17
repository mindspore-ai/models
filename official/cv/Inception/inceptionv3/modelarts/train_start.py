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
"""inceptionv3 train"""
import os
import argparse
import glob
import moxing as mox
import numpy as np
import mindspore as ms
from mindspore import Model
from mindspore import Tensor
from mindspore import context
from mindspore.common import set_seed
from mindspore.common.initializer import XavierUniform, initializer
from mindspore.communication import init, get_rank, get_group_size
from mindspore.nn import RMSProp
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor, LossMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.model import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import export
from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_id, get_device_num
from src.dataset import create_dataset_imagenet, create_dataset_cifar10
from src.inception_v3 import InceptionV3
from src.loss import CrossEntropy
from src.lr_generator import get_lr


set_seed(1)
DS_DICT = {
    "imagenet": create_dataset_imagenet,
    "cifar10": create_dataset_cifar10,
}

config.device_id = get_device_id()
config.device_num = get_device_num()
device_num = config.device_num
create_dataset = DS_DICT[config.ds_type]

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument("--filter_weight", type=str, default=True, help="Filter head weight parameters, default is True.")
parser.add_argument('--data_url', metavar='DIR', default='/cache/data_url', help='path to dataset')
parser.add_argument('--train_url', default="/cache/output/", type=str, help="setting dir of training output")
parser.add_argument('--resume', default="", type=str, help="resume training with existed checkpoint")
parser.add_argument('--ds_type', default="imagenet", type=str, help="dataset type, imagenet or cifar10")
parser.add_argument('--num_classes', default="1000", type=int, help="classes")
parser.add_argument('--epoch_size', default="250", type=int, help="epoch_size")
parser.add_argument('--batch_size', default="128", type=int, help="batch_size")
args_opt = parser.parse_args()

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
    input_arr = Tensor(np.random.uniform(0.0, 1.0, size=[args.get("batch_size"), 3, args.get("width"), \
        args.get("height")]), ms.float32)
    export(network, input_arr, file_name=args.get("file_name"), file_format=args.get("file_format"))


if __name__ == '__main__':
    if not os.path.exists(config.data_path):
        os.makedirs(config.data_path, exist_ok=True)
    if not os.path.exists(config.load_path):
        os.makedirs(config.load_path, exist_ok=True)
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path, exist_ok=True)
    mox.file.copy_parallel(args_opt.data_url, config.data_path)

    config.class_num = args_opt.num_classes
    config.epoch_size = args_opt.epoch_size
    config.batch_size = args_opt.batch_size
    config.resume = os.path.join(config.dataset_path, args_opt.resume)
    config.dataset_path = os.path.join(config.dataset_path, "train")

    if config.platform == "GPU":
        context.set_context(enable_graph_kernel=True)

    context.set_context(mode=context.GRAPH_MODE, device_target=config.platform, save_graphs=False)
    if os.getenv('DEVICE_ID', "not_set").isdigit():
        context.set_context(device_id=int(os.getenv('DEVICE_ID')))

    # init distributed
    if config.is_distributed:
        init()
        config.rank = get_rank()
        config.group_size = get_group_size()
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode, device_num=config.group_size,
                                          gradients_mean=True)
    else:
        config.rank = 0
        config.group_size = 1

    # dataloader
    dataset = create_dataset(config.dataset_path, True, config)
    batches_per_epoch = dataset.get_dataset_size()

    # network
    net = InceptionV3(num_classes=config.num_classes, dropout_keep_prob=config.dropout_keep_prob, \
        has_bias=config.has_bias)

    # loss
    loss = CrossEntropy(smooth_factor=config.smooth_factor, num_classes=config.num_classes, factor=config.aux_factor)

    # learning rate schedule
    lr = get_lr(lr_init=config.lr_init, lr_end=config.lr_end, lr_max=config.lr_max, warmup_epochs=config.warmup_epochs,
                total_epochs=config.epoch_size, steps_per_epoch=batches_per_epoch, lr_decay_mode=config.decay_method)
    lr = Tensor(lr)

    # optimizer
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    if config.platform == "Ascend":
        for param in net.trainable_params():
            if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
                param.set_data(initializer(XavierUniform(), param.data.shape, param.data.dtype))
    group_params = [{'params': decayed_params, 'weight_decay': config.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]
    optimizer = RMSProp(group_params, lr, decay=0.9, weight_decay=config.weight_decay,
                        momentum=config.momentum, epsilon=config.opt_eps, loss_scale=config.loss_scale)

    if args_opt.resume:
        ckpt = load_checkpoint(config.resume)
        if args_opt.filter_weight:
            filter_list = [x.name for x in net.logits.get_parameters()]
            filter_checkpoint_parameter_by_list(ckpt, filter_list)
        load_param_into_net(net, ckpt)
    if config.platform == "Ascend":
        loss_scale_manager = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
        model = Model(net, loss_fn=loss, optimizer=optimizer, metrics={'acc'}, amp_level=config.amp_level,
                      loss_scale_manager=loss_scale_manager)
    else:
        model = Model(net, loss_fn=loss, optimizer=optimizer, metrics={'acc'}, amp_level=config.amp_level)

    print("============== Starting Training ==============")
    loss_cb = LossMonitor(per_print_times=batches_per_epoch)
    time_cb = TimeMonitor(data_size=batches_per_epoch)
    callbacks = [loss_cb, time_cb]
    config_ck = CheckpointConfig(save_checkpoint_steps=batches_per_epoch, \
        keep_checkpoint_max=config.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix=f"inceptionv3-rank{config.rank}", \
        directory=config.output_path, config=config_ck)
    if config.is_distributed & config.is_save_on_master:
        if config.rank == 0:
            callbacks.append(ckpoint_cb)
        model.train(config.epoch_size, dataset, callbacks=callbacks, dataset_sink_mode=config.ds_sink_mode)
    else:
        callbacks.append(ckpoint_cb)
        model.train(config.epoch_size, dataset, callbacks=callbacks, dataset_sink_mode=config.ds_sink_mode)
    ckpt_list = glob.glob(config.output_path + "/inceptionv3*.ckpt")
    if not ckpt_list:
        print("ckpt file not generated.")

    ckpt_list.sort(key=os.path.getmtime)
    ckpt_model = ckpt_list[-1]
    print("checkpoint path", ckpt_model)

    net = InceptionV3(num_classes=config.num_classes, dropout_keep_prob=config.dropout_keep_prob, \
        has_bias=config.has_bias)

    frozen_to_air_args = {'ckpt_file': ckpt_model, 'batch_size': 1, 'height': 299, 'width': 299,
                          'file_name': config.output_path + '/inceptionv3', 'file_format': 'AIR'}
    frozen_to_air(net, frozen_to_air_args)
    mox.file.copy_parallel(config.output_path, args_opt.train_url)
    print("train success")
