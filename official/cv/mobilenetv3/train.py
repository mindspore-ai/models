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
"""train_imagenet."""

import time
import argparse
import ast
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, Callback
from mindspore.communication.management import init, get_group_size, get_rank

from src.dataset import create_dataset
from src.dataset import create_dataset_cifar
from src.lr_generator import get_lr
from src.config import config_gpu
from src.config import config_cpu
from src.mobilenetV3 import mobilenet_v3_large

ms.set_seed(1)

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
parser.add_argument('--pre_trained', type=str, default=None, help='Pretrained checkpoint path')
parser.add_argument('--device_target', type=str, default="GPU", help='run device_target')
parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='Run distribute')
args_opt = parser.parse_args()

if args_opt.device_target == "GPU":
    ms.set_context(mode=ms.GRAPH_MODE,
                   device_target="GPU",
                   save_graphs=False)
    if args_opt.run_distribute:
        init()
        ms.set_auto_parallel_context(device_num=get_group_size(),
                                     parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                                     gradients_mean=True)
elif args_opt.device_target == "CPU":
    ms.set_context(mode=ms.GRAPH_MODE,
                   device_target="CPU",
                   save_graphs=False)
else:
    raise ValueError("Unsupported device_target.")


class CrossEntropyWithLabelSmooth(nn.LossBase):
    """
    CrossEntropyWith LabelSmooth.

    Args:
        smooth_factor (float): smooth factor for label smooth. Default is 0.
        num_classes (int): number of classes. Default is 1000.

    Returns:
        None.

    Examples:
        >>> CrossEntropyWithLabelSmooth(smooth_factor=0., num_classes=1000)
    """

    def __init__(self, smooth_factor=0., num_classes=1000):
        super(CrossEntropyWithLabelSmooth, self).__init__()
        self.onehot = ops.OneHot()
        self.on_value = ms.Tensor(1.0 - smooth_factor, ms.float32)
        self.off_value = ms.Tensor(1.0 * smooth_factor / (num_classes - 1), ms.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.mean = ops.ReduceMean(False)
        self.cast = ops.Cast()

    def construct(self, logit, label):
        one_hot_label = self.onehot(self.cast(label, ms.int32), ops.shape(logit)[1],
                                    self.on_value, self.off_value)
        out_loss = self.ce(logit, one_hot_label)
        out_loss = self.mean(out_loss, 0)
        return out_loss


class Monitor(Callback):
    """
    Monitor loss and time.

    Args:
        lr_init (numpy array): train lr

    Returns:
        None

    Examples:
        >>> Monitor(100,lr_init=ms.Tensor([0.05]*100).asnumpy())
    """

    def __init__(self, lr_init=None):
        super(Monitor, self).__init__()
        self.lr_init = lr_init
        self.lr_init_len = len(lr_init)

    def epoch_begin(self, run_context):
        self.losses = []
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()

        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        per_step_mseconds = epoch_mseconds / cb_params.batch_num
        print("epoch time: {:5.3f}, per step time: {:5.3f}, avg loss: {:5.3f}".format(epoch_mseconds,
                                                                                      per_step_mseconds,
                                                                                      np.mean(self.losses)))

    def step_begin(self, run_context):
        self.step_time = time.time()

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        step_mseconds = (time.time() - self.step_time) * 1000
        step_loss = cb_params.net_outputs

        if isinstance(step_loss, (tuple, list)) and isinstance(step_loss[0], ms.Tensor):
            step_loss = step_loss[0]
        if isinstance(step_loss, ms.Tensor):
            step_loss = np.mean(step_loss.asnumpy())

        self.losses.append(step_loss)
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num

        print("epoch: [{:3d}/{:3d}], step:[{:5d}/{:5d}], loss:[{:5.3f}/{:5.3f}], time:[{:5.3f}], lr:[{:5.3f}]".format(
            cb_params.cur_epoch_num -
            1, cb_params.epoch_num, cur_step_in_epoch, cb_params.batch_num, step_loss,
            np.mean(self.losses), step_mseconds, self.lr_init[cb_params.cur_step_num - 1]))


if __name__ == '__main__':
    config_ = None
    if args_opt.device_target == "GPU":
        config_ = config_gpu
        ms.set_context(enable_graph_kernel=True)
    elif args_opt.device_target == "CPU":
        config_ = config_cpu
    else:
        raise ValueError("Unsupported device_target.")
    # train on device
    print("train args: ", args_opt)
    print("cfg: ", config_)

    # define net
    net = mobilenet_v3_large(num_classes=config_.num_classes)
    # define loss
    if config_.label_smooth > 0:
        loss = CrossEntropyWithLabelSmooth(
            smooth_factor=config_.label_smooth, num_classes=config_.num_classes)
    else:
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    # define dataset
    epoch_size = config_.epoch_size
    if args_opt.device_target == "GPU":
        dataset = create_dataset(dataset_path=args_opt.dataset_path,
                                 do_train=True,
                                 config=config_,
                                 device_target=args_opt.device_target,
                                 batch_size=config_.batch_size,
                                 run_distribute=args_opt.run_distribute)
    elif args_opt.device_target == "CPU":
        dataset = create_dataset_cifar(args_opt.dataset_path,
                                       do_train=True,
                                       batch_size=config_.batch_size)
    else:
        raise ValueError("Unsupported device_target.")
    step_size = dataset.get_dataset_size()
    # resume
    if args_opt.pre_trained:
        param_dict = ms.load_checkpoint(args_opt.pre_trained)
        ms.load_param_into_net(net, param_dict)
    # define optimizer
    loss_scale = ms.FixedLossScaleManager(
        config_.loss_scale, drop_overflow_update=False)
    lr = ms.Tensor(get_lr(global_step=0,
                          lr_init=0,
                          lr_end=0,
                          lr_max=config_.lr,
                          warmup_epochs=config_.warmup_epochs,
                          total_epochs=epoch_size,
                          steps_per_epoch=step_size))
    opt = nn.Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, config_.momentum,
                      config_.weight_decay, config_.loss_scale)
    # define model
    model = ms.Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale)

    cb = [Monitor(lr_init=lr.asnumpy())]
    if args_opt.run_distribute and args_opt.device_target != "CPU":
        ckpt_save_dir = config_gpu.save_checkpoint_path + "ckpt_" + str(get_rank()) + "/"
    else:
        ckpt_save_dir = config_gpu.save_checkpoint_path + "ckpt_" + "/"
    if config_.save_checkpoint:
        config_ck = CheckpointConfig(save_checkpoint_steps=config_.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config_.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix="mobilenetV3", directory=ckpt_save_dir, config=config_ck)
        cb += [ckpt_cb]
    # begine train
    model.train(epoch_size, dataset, callbacks=cb, dataset_sink_mode=True)
