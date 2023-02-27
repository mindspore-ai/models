# Copyright 2023 Huawei Technologies Co., Ltd
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
"""crnn callback."""
import time
import numpy as np

from mindspore import Tensor
from mindspore.train.callback import Callback

def handle_loss(loss):
    """Handle loss."""
    if isinstance(loss, (tuple, list)):
        if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
            loss = loss[0]
    elif isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
        loss = float(np.mean(loss.asnumpy()))
    return loss

class CRNNMonitor(Callback):
    def __init__(self, config, lr):
        self.config = config
        self.lr = lr
        self._last_print_time = 0
        self._per_print_times = config.per_print_time
        self.step_start_time = time.time()
        self.cur_steps = 0

    def on_train_step_begin(self, run_context):
        self.step_start_time = time.time()

    def on_train_step_end(self, run_context):
        """
        Called after each training step end.

        Args: run_context (RunContext): Context of the train running.
        """
        cb_params = run_context.original_args()
        loss = handle_loss(cb_params.net_outputs)
        cur_epoch = cb_params.cur_epoch_num
        cur_step_in_epoch = (cb_params.cur_epoch_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError(
                "epoch: {} step: {}. Invalid loss, terminating training.".format(cur_epoch, cur_step_in_epoch))

        if self._per_print_times != 0 and (cb_params.cur_epoch_num <= self._last_print_time):
            while cb_params.cur_epoch_num <= self._last_print_time:
                self._last_print_time -=\
                    max(self._per_print_times, cb_params.batch_num if cb_params.dataset_sink_mode else 1)

        if self._per_print_times != 0 and (cb_params.cur_epoch_num - self._last_print_time) >= self._per_print_times:
            self._last_print_time = cb_params.cur_epoch_num
            loss_log = "epoch: [%s/%s] step: [%s/%s], loss: %.6f, lr : %.6f, per step time: %.3f ms" % (
                cur_epoch, self.config.epoch_size, cur_step_in_epoch, self.config.steps_per_epoch,
                loss, self.lr[self.cur_steps], (time.time() - self.step_start_time) * 1000)
            self.config.logger.info(loss_log)
        self.cur_steps += 1

    def on_train_epoch_begin(self, run_context):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, run_context):
        """
        Called after each training epoch end.

        Args: run_context (RunContent): Content of the train running.
        """
        cb_params = run_context.original_args()
        loss = handle_loss(cb_params.net_outputs)
        cur_epoch = cb_params.cur_epoch_num
        epoch_time = (time.time() - self.epoch_start_time)
        loss_log = "epoch: [%s/%s], loss: %.6f, epoch time: %.3f ms, per step time: %.3f ms" % (
            cur_epoch, self.config.epoch_size, loss, epoch_time * 1000,
            epoch_time * 1000 / cb_params.batch_num)
        self.config.logger.info(loss_log)

        metrics = cb_params.get("metrics")
        if metrics:
            self.config.logger.info("Eval result: epoch %d, metrics: %s" % (cb_params.cur_epoch_num, metrics))

class ResumeCallback(Callback):
    def __init__(self, start_epoch=0):
        super(ResumeCallback, self).__init__()
        self.start_epoch = start_epoch

    def epoch_begin(self, run_context):
        run_context.original_args().cur_epoch_num += self.start_epoch
