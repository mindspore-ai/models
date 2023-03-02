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
"""train delf"""
import os
import time

import mindspore.nn as nn
from mindspore import context, Model
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.context import ParallelMode
from mindspore import load_checkpoint, load_param_into_net
import mindspore.ops as ops
from mindspore.train.callback import LossMonitor
from mindspore.train.callback import Callback
from mindspore.train.callback import SummaryCollector
from mindspore.communication.management import init
from mindspore.profiler import Profiler

import numpy as np

import src.convert_h5_to_weight as h5
import src.data_augmentation_parallel as daa
import src.delf_model as model_h5

from model_utils.config import config as args
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_device_num


class LossFunc(nn.Cell):
    """loss function"""

    def __init__(self, attention_loss_weight=1.0, state='tuning'):
        super(LossFunc, self).__init__()
        self._loss_fn = nn.SoftmaxCrossEntropyWithLogits(
            sparse=True, reduction="mean")
        self.state = state
        self.attention_loss_weight = attention_loss_weight

    def construct(self, base, label):
        """construct"""
        label = ops.clip_by_value(label, 0, 81313)
        if self.state == 'tuning':
            total_loss = self._loss_fn(base, label)
        else:
            (attn_logits, _) = base
            attn_loss = self._loss_fn(attn_logits, label)
            total_loss = self.attention_loss_weight * attn_loss

        return total_loss


class MySGD(nn.SGD):
    """my SGD"""

    def __init__(self, *args_in, **kwargs):
        super().__init__(*args_in, **kwargs)
        self._original_construct = super().construct
        self.gradient_names = [param.name +
                               ".gradient" for param in self.parameters]
        self.count = len(self.gradient_names)

    def construct(self, grads):
        grads = ops.clip_by_global_norm(grads, 10.0)
        return self._original_construct(grads)


class EvalCallBack(Callback):
    """my callback"""

    def __init__(self, cur_iters, fianel_iter, state,
                 eval_interval=1000, eval_start_step=1):
        super(EvalCallBack, self).__init__()

        self.eval_start_step = eval_start_step
        if eval_interval < 1:
            raise ValueError("interval should >= 1.")
        self.eval_interval = eval_interval
        self.fianel_iter = fianel_iter
        self.state = state
        self.cur_iters = cur_iters

    def begin(self, run_context):
        cb_params = run_context.original_args()
        cb_params.cur_step_num = self.cur_iters

    def record_accuracy(self, logits, label):
        """Record accuracy given predicted logits and ground-truth labels."""
        y_pred = logits.asnumpy()
        label = label.asnumpy()
        indices = np.argmax(y_pred, axis=1)
        result = (np.equal(indices, label) * 1).reshape(-1)
        correct_num = np.sum(result)
        total_num = result.shape[0]
        return correct_num / total_num

    # calculate accuracy
    def compute_acc(self, desc_logits, attn_logits, label):
        desc_acc = self.record_accuracy(desc_logits, label)
        attn_acc = self.record_accuracy(attn_logits, label)
        return desc_acc, attn_acc

    def step_end(self, run_context):
        """callback"""
        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num
        cur_network = cb_params.network
        cur_data, cur_label = cb_params.train_dataset_element

        # print accuracy
        if cur_step >= self.eval_start_step and (cur_step - self.eval_start_step) % self.eval_interval == 0:
            if self.state != 'tuning':
                attn_logits, desc_logits = cur_network(cur_data)
                desc_acc, attn_acc = self.compute_acc(
                    desc_logits, attn_logits, cur_label)
                print("step: %s, train desc Acc %s" %
                      (cur_step, desc_acc), flush=True)
                print("step: %s, train attn Acc %s" %
                      (cur_step, attn_acc), flush=True)

        # stop the training
        if cur_step > self.fianel_iter:
            run_context.request_stop()


class TimeMonitor_mine(Callback):
    """my timeMonitor"""

    def __init__(self, data_size=None):
        super(TimeMonitor_mine, self).__init__()
        self.step_size = data_size
        self.interval_time = time.time()

    def epoch_begin(self, run_context):
        self.interval_time = time.time()

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num
        if cur_step % self.step_size == 0:
            interval_seconds = (time.time() - self.interval_time) * 1000
            self.interval_time = time.time()
            step_seconds = interval_seconds / self.step_size
            print("interval time: {:5.3f} ms, per step time: {:5.3f} ms".format
                  (interval_seconds, step_seconds), flush=True)


def modelarts_pre_process():
    '''modelarts pre process function.'''
    args.save_ckpt = os.path.join(
        args.output_path, args.save_ckpt + str(get_device_id()))


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():
    """train"""
    # load data
    train_dataset = daa.create_dataset(
        args.traindata_path, args.image_size, args.batch_size,
        seed=args.seed, augmentation=True, repeat=True)

    # initial forward net
    delf_net = model_h5.Model(state=args.train_state)
    param_dict = h5.translate_h5(args.imagenet_checkpoint)
    load_param_into_net(delf_net, param_dict)

    # load ckpt
    if args.checkpoint_path != "":
        param_dict = load_checkpoint(args.checkpoint_path)
        not_load, _ = load_param_into_net(delf_net, param_dict)
        print('weights not load in ckpt: ', not_load)

    # freeze laysers
    print('freeze param:')
    if args.train_state == "attn":
        for param in delf_net.get_parameters():
            if ('attention.' not in param.name and
                    'attn.' not in param.name):
                print(param.name)
                param.requires_grad = False
    elif args.train_state == "tuning":
        for param in delf_net.get_parameters():
            if ('attention.' in param.name or
                    'attn.' in param.name):
                print(param.name)
                param.requires_grad = False

    # loss func
    loss_func = LossFunc(attention_loss_weight=args.attention_loss_weight,
                         state=args.train_state)

    # dynamic lr
    init_lr = args.initial_lr * (1 - args.start_iter / 250000)
    lr_schedule = nn.PolynomialDecayLR(
        learning_rate=init_lr, end_learning_rate=0.0001, decay_steps=500000, power=1.0)

    optim = MySGD(delf_net.trainable_params(), learning_rate=lr_schedule,
                  momentum=0.9, weight_decay=0.0)

    config_ck = CheckpointConfig(
        save_checkpoint_steps=args.save_ckpt_step, keep_checkpoint_max=args.keep_checkpoint_max)

    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_delf_" + args.train_state,
                                 directory=args.save_ckpt,
                                 config=config_ck)

    eval_cb = EvalCallBack(args.start_iter, args.max_iters, args.train_state)

    callback_size = 100
    callback_list = [TimeMonitor_mine(callback_size), eval_cb, LossMonitor(callback_size)]
    if device_id == 0:
        callback_list.append(ckpoint_cb)
        if args.need_summary:
            callback_list.append(summary_collector)

    print("Ready to train!")
    model = Model(network=delf_net, loss_fn=loss_func,
                  optimizer=optim, amp_level="O3")
    model.train(1, train_dataset, callbacks=callback_list,
                dataset_sink_mode=False)

    if args.need_profile:
        profiler.analyse()

    print("Train successfully!")


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    device_id = get_device_id()
    device_num = get_device_num()
    specified = {'histogram_regular': 'attention.*'}
    summary_collector = None
    profiler = None
    if args.enable_modelarts:
        args.save_summary = os.path.join(args.output_path, args.save_summary)
    if device_num > 1:
        context.set_context(device_id=device_id)
        if args.need_profile:
            profiler = Profiler(output_path=os.path.join(
                args.save_summary, 'summary_dir' + str(device_id)))
        if args.need_summary:
            summary_collector = SummaryCollector(
                summary_dir=os.path.join(
                    args.save_summary, 'summary_dir' + str(device_id)),
                collect_specified_data=specified, collect_freq=200)
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=False)
        init()
    else:
        context.set_context(device_id=device_id)
        if args.need_profile:
            profiler = Profiler(output_path=os.path.join(
                args.save_summary, 'summary_dir'))
        if args.need_summary:
            summary_collector = SummaryCollector(summary_dir=os.path.join(args.save_summary, 'summary_dir'),
                                                 collect_specified_data=specified, collect_freq=200)
    run_train()
