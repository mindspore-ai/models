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
"""custom callbacks for ema and loss"""
import numpy as np
from mindspore.train.callback import Callback
from mindspore import Tensor


class LossMonitor(Callback):
    """
    Monitor the loss in training.

    If the loss is NAN or INF, it will terminate training.

    Note:
        If per_print_times is 0, do not print loss.

    Args:
        lr_array (numpy.array): scheduled learning rate.
        total_epochs (int): Total number of epochs for training.
        per_print_times (int): Print the loss every time. Default: 1.
        start_epoch (int): which epoch to start, used when resume from a
        certain epoch.

    Raises:
        ValueError: If print_step is not an integer or less than zero.
    """

    def __init__(self, lr_array, total_epochs, per_print_times=1, start_epoch=0):
        super(LossMonitor, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self._lr_array = lr_array
        self._total_epochs = total_epochs
        self._start_epoch = start_epoch

    def step_end(self, run_context):
        """log epoch, step, loss and learning rate"""
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        cur_epoch_num = cb_params.cur_epoch_num + self._start_epoch - 1
        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())
        global_step = cb_params.cur_step_num - 1
        cur_step_in_epoch = global_step % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                cur_epoch_num, cur_step_in_epoch))

        if self._per_print_times != 0 and cur_step_in_epoch % self._per_print_times == 0:
            print("epoch: {}/{}, step: {}/{}, loss is {}, learning rate: {}".format(cur_epoch_num,
                                                                                    self._total_epochs,
                                                                                    cur_step_in_epoch,
                                                                                    cb_params.batch_num,
                                                                                    loss,
                                                                                    self._lr_array[global_step]))
