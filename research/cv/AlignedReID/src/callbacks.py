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
""" Custom callback """

import os
import time

import mindspore
import numpy as np


class SavingLossMonitor(mindspore.train.callback.Callback):
    """ Monitor the loss in training with saving it to file.

    If the loss is NAN or INF, it will terminate training.

    Note:
        If per_print_times is 0, do not print loss.

    Args:
        per_print_times (int): Print the loss every seconds. Default: 1.
        logfile (str): path to save file
        timestamp (str/bool): add timestamp to file name (if str - add this string)
        init_info: (str): info to add file head

    Raises:
        ValueError: If per_print_times is not an integer or less than zero.
    """
    def __init__(self, per_print_times=1, logfile=None, timestamp=False, init_info=None):
        super(SavingLossMonitor, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("'Per_print_times' must be int and >= 0, "
                             "but got {}".format(per_print_times))
        self._per_print_times = per_print_times
        self._logfile = None
        if logfile is not None:
            if timestamp:
                if not os.path.isdir(logfile):
                    logfile = os.path.dirname(logfile)
                if not isinstance(timestamp, str):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                logfile = os.path.join(logfile, timestamp+".logs.txt")
            self._logfile = open(logfile, 'w', buffering=1)
            if init_info:
                print(init_info, file=self._logfile)

    def step_end(self, run_context):
        """
        Print training loss at the end of step.

        Args:
            run_context (RunContext): Context of the train running.

        """
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], mindspore.Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, mindspore.Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                cb_params.cur_epoch_num, cur_step_in_epoch))

        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            if isinstance(loss, float):
                logs = "epoch: %s step: %s, loss is %f" % (cb_params.cur_epoch_num, cur_step_in_epoch, loss)
            else:
                logs = "epoch: %s step: %s, loss is %s" % (cb_params.cur_epoch_num, cur_step_in_epoch, loss)
            print(logs, flush=True)
            if self._logfile is not None:
                print(logs, file=self._logfile)

    def end(self, run_context):
        """ Close file in the end """
        if self._logfile is not None:
            self._logfile.close()
