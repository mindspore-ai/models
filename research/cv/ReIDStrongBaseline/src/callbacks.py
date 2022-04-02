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
""" Custom callbacks """
import io
import os
import time

import numpy as np
from mindspore import Tensor
from mindspore.train.callback import Callback


class SavingCallback(Callback):
    """ File logging callback

    Args:
        logfile (str): path to save file
        timestamp (str/bool): add timestamp to file name (if str - add this string)
    """
    def __init__(self, logfile=None, timestamp=False):
        super(SavingCallback, self).__init__()
        self._logfile = None
        if logfile is not None:
            if isinstance(logfile, str):
                self._logfile = self.open_file(logfile, timestamp)
            elif isinstance(logfile, io.TextIOBase):
                self._logfile = logfile
            else:
                raise TypeError('Log file is not TextIOBase or path')

    @staticmethod
    def open_file(logfile=None, timestamp=False):
        """ Create and open file for logging

        Args:
            logfile (str): path to save file
            timestamp (str/bool): add timestamp to file name (if str - add this string)

        Returns:
            File IO
        """
        if logfile is None:
            return logfile
        if timestamp:
            if os.path.isfile(logfile):
                logfile = os.path.dirname(logfile)
            if not isinstance(timestamp, str):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
            logfile = os.path.join(logfile, timestamp + ".logs.txt")
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        logfile = open(logfile, 'w', buffering=1)
        return logfile

    def print_file(self, text):
        """ Print text to file """
        if (self._logfile is not None) and (not self._logfile.closed):
            print(text, file=self._logfile, flush=True)

    def end(self, run_context):
        """ Close file in the end """
        if (self._logfile is not None) and (not self._logfile.closed):
            self._logfile.close()


class SavingLossMonitor(SavingCallback):
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
    def __init__(self, per_print_times=1, init_info=None, logfile=None, timestamp=False):
        super(SavingLossMonitor, self).__init__(logfile, timestamp)
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("'Per_print_times' must be int and >= 0, "
                             "but got {}".format(per_print_times))
        self._per_print_times = per_print_times

        if init_info:
            self.print_file(init_info)

    def step_end(self, run_context):
        """
        Print training loss at the end of step.

        Args:
            run_context (RunContext): Context of the train running.

        """
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
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
            self.print_file(logs)


class SavingTimeMonitor(SavingCallback):
    """ Monitor the time in training with saving it to file.

    If the loss is NAN or INF, it will terminate training.

    Note:
        If per_print_times is 0, do not print loss.

    Args:
        data_size (int): How many steps are the intervals between print information each time.
            if the program get `batch_num` during training, `data_size` will be set to `batch_num`,
            otherwise `data_size` will be used. Default: None.
        logfile (str): path to save file
        timestamp (str/bool): add timestamp to file name (if str - add this string)

    Raises:
        ValueError: If data_size is not positive int.
    """
    def __init__(self, data_size=None, logfile=None, timestamp=False):
        super(SavingTimeMonitor, self).__init__(logfile, timestamp)
        self.data_size = data_size
        self.epoch_time = time.time()

    def epoch_begin(self, run_context):
        """
        Record time at the begin of epoch.

        Args:
            run_context (RunContext): Context of the process running.
        """
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        """
        Print process cost time at the end of epoch.

        Args:
           run_context (RunContext): Context of the process running.
        """
        epoch_seconds = (time.time() - self.epoch_time) * 1000
        step_size = self.data_size
        cb_params = run_context.original_args()
        if hasattr(cb_params, "batch_num"):
            batch_num = cb_params.batch_num
            if isinstance(batch_num, int) and batch_num > 0:
                step_size = cb_params.batch_num

        if not isinstance(step_size, int) or step_size < 1:
            raise ValueError("data_size must be positive int.")

        step_seconds = epoch_seconds / step_size
        logs = "epoch time: {:5.3f} ms, per step time: {:5.3f} ms".format(epoch_seconds, step_seconds)
        print(logs, flush=True)
        self.print_file(logs)
