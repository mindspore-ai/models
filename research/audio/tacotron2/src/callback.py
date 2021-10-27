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
''' callback function '''
import time
import math

from mindspore.train.callback import Callback
from mindspore import Tensor
import numpy as np



def get_lr(init_lr, total_epoch, step_per_epoch,
           anneal_step=250):
    ''' warmup lr schedule'''
    total_step = total_epoch * step_per_epoch
    lr_step = []

    for step in range(total_step):
        lambda_lr = anneal_step**0.5 * \
            min((step + 1) * anneal_step**-1.5, (step + 1)**-0.5)
        lr_step.append(init_lr * lambda_lr)
    learning_rate = np.array(lr_step).astype(np.float32)
    return learning_rate


class TimeMonitor(Callback):
    """
    Time monitor for calculating cost of each epoch.

    Args:
        data_size (int): step size of an epoch.
    """

    def __init__(self, data_size):
        super(TimeMonitor, self).__init__()
        self.data_size = data_size

    def epoch_begin(self, run_context):
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        epoch_seconds = (time.time() - self.epoch_time)
        per_step_seconds = epoch_seconds / self.data_size
        print("epoch time: {}, per step time: {}".format(epoch_seconds, per_step_seconds), flush=True)

    def step_begin(self, run_context):
        self.step_time = time.time()

    def step_end(self, run_context):
        step_seconds = (time.time() - self.step_time)
        print("step time {}".format(step_seconds), flush=True)

class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss in NAN or INF terminating training.
    Note:
        if per_print_times is 0 do not print loss.
    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """
    def __init__(self, dataset_size=-1):
        super(LossCallBack, self).__init__()
        self._dataset_size = dataset_size
    def step_end(self, run_context):
        """
        Print loss after each step
        """
        cb_params = run_context.original_args()
        if self._dataset_size > 0:
            percent, epoch_num = math.modf(cb_params.cur_step_num / self._dataset_size)
            if percent == 0:
                percent = 1
                epoch_num -= 1
            print("epoch: {}, current epoch percent: {}, step: {}, outputs are {}"
                  .format(int(epoch_num), "%.3f" % percent, cb_params.cur_step_num, str(cb_params.net_outputs)),
                  flush=True)
        else:
            print("epoch: {}, step: {}, outputs are {}".format(cb_params.cur_epoch_num, cb_params.cur_step_num,
                                                               str(cb_params.net_outputs)), flush=True)

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

        epoch_mseconds = (time.time() - self.epoch_time)
        per_step_mseconds = epoch_mseconds / cb_params.batch_num
        print("epoch time: {:5.3f}, per step time: {:5.3f}, avg loss: {:5.6f}".format(\
            epoch_mseconds, per_step_mseconds, np.mean(self.losses)))

    def step_begin(self, run_context):
        self.step_time = time.time()

    def step_end(self, run_context):
        """step end"""
        cb_params = run_context.original_args()
        step_mseconds = (time.time() - self.step_time)
        step_loss = cb_params.net_outputs

        if isinstance(
                step_loss, (tuple, list)) and isinstance(
                    step_loss[0], Tensor):
            step_loss = step_loss[0]
        if isinstance(step_loss, Tensor):
            step_loss = np.mean(step_loss.asnumpy())

        self.losses.append(step_loss)
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num

        print("epoch: [{:3d}/{:3d}], step:[{:5d}/{:5d}], loss:[{:5.6f}/{:5.6f}], time:[{:5.3f}], lr:[{:.9f}]".format(\
            cb_params.cur_epoch_num -\
            1, cb_params.epoch_num, cur_step_in_epoch, cb_params.batch_num, step_loss,\
            np.mean(self.losses), step_mseconds, self.lr_init[cb_params.cur_step_num - 1].asnumpy()))
