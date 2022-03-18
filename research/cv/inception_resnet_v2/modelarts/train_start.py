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
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import ops as P
from mindspore.common import set_seed
from mindspore.common.initializer import XavierUniform, initializer
from mindspore.communication import init
from mindspore.context import ParallelMode
from mindspore.nn import RMSProp, Momentum
from mindspore.nn.loss.loss import LossBase
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor, LossMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import export

from src.config import config_gpu, config_ascend
from src.dataset import create_dataset
from src.inception_resnet_v2 import Inception_resnet_v2


os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

arg_parser = argparse.ArgumentParser(description='inception resnet v2 image classification training')
arg_parser.add_argument("--filter_weight", type=str, default=True,
                        help="Filter head weight parameters, default is False.")
arg_parser.add_argument('--train_url', type=str, default='/cache/output/', help='the path model saved')
arg_parser.add_argument('--data_url', type=str, default='/cache/data_url', help='the training data')
arg_parser.add_argument('--dataset_path', type=str, default='/cache/data', help='Dataset path')
arg_parser.add_argument('--output_path', type=str, default='/cache/train', help='output path')
arg_parser.add_argument('--device_id', type=int, default=0, help='device id')
arg_parser.add_argument('--platform', type=str, default='Ascend', choices=('Ascend', 'GPU'), help='run platform')
arg_parser.add_argument('--epoch_size', default="10", type=int, help="epoch_size")
arg_parser.add_argument('--batch_size', default="128", type=int, help="batch_size")
arg_parser.add_argument('--num_classes', default="1000", type=int, help="classes")
arg_parser.add_argument('--resume', type=str, default='', help='resume training with existed checkpoint')

args = arg_parser.parse_args()

set_seed(1)


class CrossEntropySmooth(LossBase):
    """CrossEntropy"""

    def __init__(self, sparse=True, reduction='mean', smooth_factor=0., num_classes=1000):
        super(CrossEntropySmooth, self).__init__()
        self.onehot = P.OneHot()
        self.sparse = sparse
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(1.0 * smooth_factor / (num_classes - 1), mstype.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)
        self.cast = P.Cast()

    def construct(self, logit, label):
        if self.sparse:
            label = self.onehot(label, logit.shape[1], self.on_value, self.off_value)
        loss2 = self.ce(logit, label)
        return loss2


def generate_cosine_lr(steps_per_epoch, total_epochs,
                       lr_init, lr_end,
                       lr_max, warmup_epochs,
                       start_epoch):
    """
    Applies cosine decay to generate learning rate array.

    Args:
       steps_per_epoch(int): steps number per epoch
       total_epochs(int): all epoch in training.
       lr_init(float): init learning rate.
       lr_end(float): end learning rate
       lr_max(float): max learning rate.
       warmup_steps(int): all steps in warmup epochs.
       start_epoch(int): start epoch.

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
            l_r = float(lr_init) + lr_inc * (i + 1)
        else:
            cosine_decay = 0.5 * (1 + math.cos(math.pi * (i - warmup_steps) / decay_steps))
            l_r = (lr_max - lr_end) * cosine_decay + lr_end
        lr_each_step.append(l_r)
    learning_rate = np.array(lr_each_step).astype(np.float32)
    current_step = steps_per_epoch * (start_epoch - 1)
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

def frozen_to_air(network, arguments):
    param_dict_t = load_checkpoint(arguments.get("ckpt_file"))
    load_param_into_net(network, param_dict_t)
    input_arr = Tensor(np.zeros([arguments.get("batch_size"), 3,
                                 arguments.get("height"), arguments.get("width")], np.float32))
    export(network, input_arr, file_name=arguments.get("file_name"), file_format=arguments.get("file_format"))

if __name__ == '__main__':

    config = config_gpu if args.platform == "GPU" else config_ascend
    print('config:', config)

    config.resume = args.resume
    config.dataset_path = args.dataset_path
    config.output_path = args.output_path
    config.epoch_size = args.epoch_size
    config.batch_size = args.batch_size
    config.num_classes = args.num_classes
    config.filter_weight = args.filter_weight

    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path, exist_ok=True)
    mox.file.copy_parallel(args.data_url, config.dataset_path)

    print('----------------------------------------------------------')
    dir_path = os.path.dirname(os.path.abspath(__file__))
    print("dir_path", dir_path)
    config.resume = os.path.join(dir_path, config.resume)
    print(config.resume)
    print('----------------------------------------------------------')
    config.dataset_path = os.path.join(config.dataset_path, "train")
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    context.set_context(enable_graph_kernel=False)

    device_num = int(os.environ.get("RANK_SIZE", 1))

    if device_num == 1:
        context.set_context(device_id=args.device_id)
    elif device_num > 1:
        init()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True, all_reduce_fusion_config=[200, 400])

    train_dataset = create_dataset(dataset_path=config.dataset_path, do_train=True,
                                   repeat_num=1, batch_size=config.batch_size, config=config)

    train_step_size = train_dataset.get_dataset_size()

    net = Inception_resnet_v2(classes=config.num_classes)

    loss = CrossEntropySmooth(sparse=True, reduction="mean",
                              smooth_factor=config.smooth_factor, num_classes=config.num_classes)
    lr = Tensor(generate_cosine_lr(steps_per_epoch=train_step_size, total_epochs=config.epoch_size,
                                   lr_init=config.lr_init, lr_end=config.lr_end, lr_max=config.lr_max,
                                   warmup_epochs=config.warmup_epochs, start_epoch=config.start_epoch))

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

    if config.optim.lower() == 'rmsprop':
        opt = RMSProp(group_params, lr, decay=config.decay, epsilon=config.epsilon, weight_decay=config.weight_decay,
                      momentum=config.momentum, loss_scale=config.loss_scale)
    elif config.optim.lower() == 'momentum':
        opt = Momentum(params=group_params, learning_rate=lr,
                       momentum=config.momentum, weight_decay=config.weight_decay,
                       loss_scale=config.loss_scale)
    else:
        raise ValueError("Unsupported optimizer.")

    if args.device_id == 0:
        print(lr)
        print(train_step_size)

    if args.resume:
        ckpt = load_checkpoint(config.resume)
        if config.filter_weight:
            filter_list = [x.name for x in net.softmax.get_parameters()]
            filter_checkpoint_parameter_by_list(ckpt, filter_list)

        load_param_into_net(net, ckpt)

    loss_scale_manager = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)

    model = Model(net, loss_fn=loss, optimizer=opt, metrics={
        'acc', 'top_1_accuracy', 'top_5_accuracy'}, loss_scale_manager=loss_scale_manager, amp_level=config.amp_level)

    # define callbacks
    performance_cb = TimeMonitor(data_size=train_step_size)
    loss_cb = LossMonitor(per_print_times=train_step_size)
    ckp_save_step = config.save_checkpoint_epochs * train_step_size
    config_ck = CheckpointConfig(save_checkpoint_steps=ckp_save_step, keep_checkpoint_max=config.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="inception_resnet_v2", directory=config.output_path, config=config_ck)

    callbacks = [performance_cb, loss_cb]

    if device_num > 1 and config.is_save_on_master:
        if args.device_id == 0:
            callbacks.append(ckpoint_cb)
    else:
        callbacks.append(ckpoint_cb)

    print("===============start training===========")
    model.train(config.epoch_size, train_dataset, callbacks=callbacks, dataset_sink_mode=True)
    print("config.output_path:", config.output_path)
    print("===============training finish==========")

    # After the training is complete, copy the generated model to the guidance output directory
    ckpt_list = glob.glob(config.output_path + "/inception_resnet_v2*.ckpt")
    print("===========ckpt===============:", ckpt_list)
    if not ckpt_list:
        print("ckpt file not generated.")
        ckpt_list.sort(key=os.path.getmtime)

    ckpt_model = ckpt_list[-1]
    print("checkpoint path", ckpt_model)

    net = Inception_resnet_v2(classes=config.num_classes)
    frozen_to_air_args = {'ckpt_file': ckpt_model,
                          'batch_size': 1,
                          'height': 299,
                          'width': 299,
                          'file_name': config.output_path + '/inception_resnet_v2',
                          'file_format': 'AIR'}
    frozen_to_air(net, frozen_to_air_args)
    mox.file.copy_parallel(config.output_path, args.train_url)
    print('Inceptionv_resnet_v2 training success!')
