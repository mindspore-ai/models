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
"""train_imagenet."""

import os
import time
import argparse
import ast
import numpy as np

from mindspore import context
from mindspore import Tensor
from mindspore import nn
from mindspore.nn.optim.momentum import Momentum
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn.loss.loss import LossBase
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, Callback
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from mindspore.communication.management import init
from mindspore import save_checkpoint
from src.dataset import create_dataset
from src.lr_generator import get_lr
from src.config import config_ascend
from src.mobilenetV3 import mobilenet_v3_large


set_seed(1)

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--data_url', type=str)
parser.add_argument('--train_url', type=str)
parser.add_argument('--train_dataset_path', type=str, default=None, help='Dataset path')
parser.add_argument('--eval_dataset_path', type=str, default=None, help='Dataset path')
parser.add_argument('--device_id', type=int, default=0, help='device id of Ascend. (Default: 0)')
parser.add_argument('--pre_trained', type=str, default=None, help='Pretrained checkpoint path')
parser.add_argument('--is_modelarts', type=ast.literal_eval, default=False, help='modelarts')
parser.add_argument('--run_distribute', type=ast.literal_eval, default=True, help='Run distribute')
args_opt = parser.parse_args()

class SaveCallback(Callback):
    """
    SaveCallback.

    Args:
        model_save (nn.Cell): the network.
        eval_dataset_save (dataset): dataset used to evaluation.
        save_file_path (string): the path to save checkpoint.

    Returns:
        None.

    Examples:
        >>> SaveCallback(model, dataset, './save_ckpt')
    """
    def __init__(self, model_save, eval_dataset_save, save_file_path):
        super(SaveCallback, self).__init__()
        self.model = model_save
        self.eval_dataset = eval_dataset_save
        self.acc = 0.75
        self.save_path = save_file_path

    def step_end(self, run_context):
        cb_params = run_context.original_args()

        result = self.model.eval(self.eval_dataset)
        print(result)
        if result['Top1-Acc'] > self.acc:
            self.acc = result['Top1-Acc']
            file_name = self.save_path + str(self.acc) + ".ckpt"
            save_checkpoint(save_obj=cb_params.train_network, ckpt_file_name=file_name)
            print("Save the maximum accuracy checkpoint,the accuracy is", self.acc)


class CrossEntropyWithLabelSmooth(LossBase):
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
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(1.0 * smooth_factor /
                                (num_classes - 1), mstype.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.mean = P.ReduceMean(False)
        self.cast = P.Cast()

    def construct(self, logit, label):
        one_hot_label = self.onehot(self.cast(label, mstype.int32), F.shape(logit)[1],
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
        >>> Monitor(100,lr_init=Tensor([0.05]*100).asnumpy())
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
        """step_end"""
        cb_params = run_context.original_args()
        step_mseconds = (time.time() - self.step_time) * 1000
        step_loss = cb_params.net_outputs

        if isinstance(step_loss, (tuple, list)) and isinstance(step_loss[0], Tensor):
            step_loss = step_loss[0]
        if isinstance(step_loss, Tensor):
            step_loss = np.mean(step_loss.asnumpy())

        self.losses.append(step_loss)
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num

        print("epoch: [{:3d}/{:3d}], step:[{:5d}/{:5d}], loss:[{:5.3f}/{:5.3f}], time:[{:5.3f}], lr:[{:5.3f}]".format(
            cb_params.cur_epoch_num -
            1, cb_params.epoch_num, cur_step_in_epoch, cb_params.batch_num, step_loss,
            np.mean(self.losses), step_mseconds, self.lr_init[cb_params.cur_step_num - 1]))


if __name__ == '__main__':

    config = config_ascend

    # print configuration
    print("train args: ", args_opt)
    print("cfg: ", config)

    device_id = args_opt.device_id

    # set context and device init
    if args_opt.run_distribute:
        context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', device_id=device_id)
        init()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
    else:
        context.set_context(device_id=device_id)

    # define net
    net = mobilenet_v3_large(num_classes=config.num_classes)

    # define loss
    if config.label_smooth > 0:
        loss = CrossEntropyWithLabelSmooth(
            smooth_factor=config.label_smooth, num_classes=config.num_classes)
    else:
        loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # define dataset
    epoch_size = config.epoch_size

    if args_opt.is_modelarts:
        import moxing as mox

        mox.file.copy_parallel(src_url=args_opt.data_url, dst_url='/cache/dataset/device_' + os.getenv('DEVICE_ID'))
        train_dataset_path = '/cache/dataset/device_' + str(device_id) + '/train'
        eval_dataset_path = '/cache/dataset/device_' + str(device_id) + '/val'
        dataset = create_dataset(dataset_path=train_dataset_path,
                                 do_train=True,
                                 config=config,
                                 repeat_num=1,
                                 batch_size=args_opt.batch_size)
        eval_dataset = create_dataset(dataset_path=eval_dataset_path,
                                      do_train=False,
                                      config=config,
                                      repeat_num=1,
                                      batch_size=args_opt.batch_size)

    else:
        dataset = create_dataset(dataset_path=args_opt.train_dataset_path,
                                 do_train=True,
                                 config=config,
                                 repeat_num=1,
                                 batch_size=config.batch_size,
                                 run_distribute=args_opt.run_distribute)

        eval_dataset = create_dataset(
            dataset_path=args_opt.eval_dataset_path,
            do_train=False,
            config=config,
            repeat_num=1,
            batch_size=config.batch_size)

    step_size = dataset.get_dataset_size()

    # resume
    if args_opt.pre_trained:
        param_dict = load_checkpoint(args_opt.pre_trained)
        load_param_into_net(net, param_dict)

    # define optimizer
    loss_scale = FixedLossScaleManager(
        config.loss_scale, drop_overflow_update=False)
    lr = Tensor(get_lr(global_step=0,
                       lr_init=config.lr_init,
                       lr_end=config.lr_end,
                       lr_max=config.lr,
                       warmup_epochs=config.warmup_epochs,
                       total_epochs=epoch_size,
                       steps_per_epoch=step_size))

    # define optimizer
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, config.momentum,
                   config.weight_decay,
                   config.loss_scale)

    # define evaluation metrics
    eval_metrics = {'Loss': nn.Loss(),
                    'Top1-Acc': nn.Top1CategoricalAccuracy(),
                    'Top5-Acc': nn.Top5CategoricalAccuracy()}
    # define model
    model = Model(net, loss_fn=loss, optimizer=opt,
                  loss_scale_manager=loss_scale, metrics=eval_metrics)

    cb = [Monitor(lr_init=lr.asnumpy())]

    if args_opt.is_modelarts:
        save_checkpoint_path = '/cache/train_output/device_' + str(device_id) + '/'
    else:
        rank = 0
        save_checkpoint_path = 'ckpts_rank_' + str(rank)
    ckp_save_step = config.save_checkpoint_epochs * step_size
    config_ck = CheckpointConfig(save_checkpoint_steps=ckp_save_step, keep_checkpoint_max=config.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix=f"mobilenetv3",
                                 directory=save_checkpoint_path, config=config_ck)
    save_cb = SaveCallback(model, eval_dataset, save_checkpoint_path)
    cb += [ckpoint_cb, save_cb]

    # begin train
    model.train(epoch_size, dataset, callbacks=cb, dataset_sink_mode=True)
    if args_opt.is_modelarts:
        mox.file.copy_parallel(src_url='/cache/train_output', dst_url=args_opt.train_url)
