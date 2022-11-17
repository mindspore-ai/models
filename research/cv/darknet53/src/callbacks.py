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
"""
custom callbacks for MindInsight
"""
import time
import math
import os
import stat
import numpy as np

from mindspore.train.callback import Callback, LossMonitor, TimeMonitor
from mindspore import save_checkpoint
from mindspore import Tensor

class CustomLossMonitor(LossMonitor):
    """Own Loss Monitor that uses specified Summary Record instance"""

    def __init__(self, summary_record, frequency):
        super(CustomLossMonitor, self).__init__(10)

        self._summary_record = summary_record
        self._freq = frequency

    def step_end(self, run_context):
        """Called after each step finished."""
        super(CustomLossMonitor, self).step_end(run_context)

        cb_params = run_context.original_args()
        if cb_params.cur_step_num % self._freq == 0:
            step_loss = cb_params.net_outputs

            if isinstance(step_loss, (tuple, list)) and isinstance(step_loss[0], Tensor):
                step_loss = step_loss[0]
            if isinstance(step_loss, Tensor):
                step_loss = np.mean(step_loss.asnumpy())

            self._summary_record.add_value('scalar', 'training_loss_', Tensor(step_loss))
            self._summary_record.record(cb_params.cur_step_num)


class CustomTimeMonitor(TimeMonitor):
    """Own Time Monitor that uses specified Summary Record instance"""

    def __init__(self, summary_record):
        super(CustomTimeMonitor, self).__init__()
        self._summary_record = summary_record

    def epoch_end(self, run_context):
        """Called after each epoch finished."""
        super(CustomTimeMonitor, self).epoch_end(run_context)

        epoch_seconds = (time.time() - self.epoch_time) * 1000
        cb_params = run_context.original_args()
        if hasattr(cb_params, "batch_num"):
            batch_num = cb_params.batch_num
            if isinstance(batch_num, int) and batch_num > 0:
                step_size = cb_params.batch_num

        if not isinstance(step_size, int) or step_size < 1:
            raise ValueError("data_size must be positive int.")

        step_seconds = epoch_seconds / step_size
        self._summary_record.add_value('scalar', 'step_time_', Tensor(step_seconds))
        self._summary_record.record(cb_params.cur_step_num)


class CustomCheckpointSaver(Callback):
    """Save the best model during training. Helpful to deal with training loss hesitation at the end"""

    def __init__(self, epoch_to_enable, save_dir):
        super(CustomCheckpointSaver, self).__init__()
        self.epoch_to_enable = epoch_to_enable
        self.save_dir = save_dir
        self.cur_best_loss = math.inf
        self.file_name = "best.ckpt"

    def step_end(self, run_context):
        """Called after each step finished."""

        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num

        if cur_epoch >= self.epoch_to_enable:
            step_loss = cb_params.net_outputs
            if isinstance(step_loss, (tuple, list)) and isinstance(step_loss[0], Tensor):
                step_loss = step_loss[0]
            if isinstance(step_loss, Tensor):
                step_loss = np.mean(step_loss.asnumpy())

            step_loss = round(step_loss, 4)
            if step_loss < self.cur_best_loss:
                self.remove_ckpoint_file(os.path.join(self.save_dir, self.file_name))
                self.file_name = "best_{}_{}.ckpt".format(cur_epoch, step_loss)
                save_checkpoint(cb_params.train_network, os.path.join(self.save_dir, self.file_name))
                self.cur_best_loss = step_loss

    def remove_ckpoint_file(self, file_name):
        """Remove the specified checkpoint file from this checkpoint manager and also from the directory."""
        try:
            os.chmod(file_name, stat.S_IWRITE)
            os.remove(file_name)
        except OSError:
            pass
