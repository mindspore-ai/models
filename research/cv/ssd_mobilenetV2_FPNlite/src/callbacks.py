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

"""Callbacks for loss monitoring and checkpoints saving"""

import os
import stat
from mindspore.train.callback import LossMonitor
from mindspore.train.summary import SummaryRecord
from mindspore import Tensor
from mindspore import save_checkpoint
from mindspore import log as logger
import numpy as np
from src.model_utils.config import config as cfg

class CustomLossMonitor(LossMonitor):
    """Own Loss Monitor that uses specified Summary Record instance"""

    def __init__(self, summary_record: SummaryRecord, mode: str, frequency: int = 1):
        super(CustomLossMonitor, self).__init__()
        self._summary_record = summary_record
        self._mode = mode
        self.best_loss = None
        self._freq = frequency
        self._best_ckpt_freq = frequency * 5
        self._best_ckpt_freq_upd = 0
        self.best_ckpt_path = os.path.join(cfg.ckpt_directory, cfg.best_ckpt_name)

    def remove_ckpoint_file(self, file_name):
        """Remove the specified checkpoint file from this checkpoint manager and also from the directory."""
        try:
            os.chmod(file_name, stat.S_IWRITE)
            os.remove(file_name)
        except OSError:
            logger.warning("OSError, failed to remove the older ckpt file %s.", file_name)
        except ValueError:
            logger.warning("ValueError, failed to remove the older ckpt file %s.", file_name)


    def epoch_begin(self, run_context):
        """Called before each epoch beginning."""
        if self._mode != 'eval':
            super(CustomLossMonitor, self).epoch_begin(run_context)

    def epoch_end(self, run_context):
        """Called after each epoch finished."""
        if self._mode != 'eval':
            cb_params = run_context.original_args()

            if cb_params.cur_step_num % self._freq == 0:
                step_loss = cb_params.net_outputs

                if isinstance(step_loss, (tuple, list)) and isinstance(step_loss[0], Tensor):
                    step_loss = step_loss[0]
                if isinstance(step_loss, Tensor):
                    step_loss = np.mean(step_loss.asnumpy())

                self._summary_record.add_value('scalar', 'loss_' + self._mode, Tensor(step_loss))
                self._summary_record.record(cb_params.cur_step_num)


            if cb_params.cur_epoch_num == self._best_ckpt_freq_upd + 100:
                self._best_ckpt_freq = max(int(self._best_ckpt_freq /2), 1)
                self._best_ckpt_freq_upd += 100

            if cb_params.cur_step_num % self._best_ckpt_freq == 0:
                step_loss = cb_params.net_outputs

                if isinstance(step_loss, (tuple, list)) and isinstance(step_loss[0], Tensor):
                    step_loss = step_loss[0]
                if isinstance(step_loss, Tensor):
                    step_loss = np.mean(step_loss.asnumpy())

                if not self.best_loss:
                    self.best_loss = step_loss + 1
                if step_loss < self.best_loss:
                    self.best_loss = step_loss
                    print(f"update best result: {step_loss}", flush=True)
                    if os.path.exists(self.best_ckpt_path):
                        self.remove_ckpoint_file(self.best_ckpt_path)
                    save_checkpoint(cb_params.train_network, self.best_ckpt_path)
                    print(f"update best checkpoint at: {self.best_ckpt_path}", flush=True)

            super(CustomLossMonitor, self).epoch_end(run_context)

    def step_begin(self, run_context):
        """Called before each step beginning."""
        if self._mode != 'eval':
            super(CustomLossMonitor, self).step_begin(run_context)

    def step_end(self, run_context):
        """Called after each step finished."""
        cb_params = run_context.original_args()

        if cb_params.cur_step_num % self._freq == 0:
            step_loss = cb_params.net_outputs

            if isinstance(step_loss, (tuple, list)) and isinstance(step_loss[0], Tensor):
                step_loss = step_loss[0]
            if isinstance(step_loss, Tensor):
                step_loss = np.mean(step_loss.asnumpy())

            self._summary_record.add_value('scalar', 'loss_' + self._mode, Tensor(step_loss))
            self._summary_record.record(cb_params.cur_step_num)

        if self._mode != 'eval':
            super(CustomLossMonitor, self).step_end(run_context)

    def end(self, run_context):
        """Called once after network training."""
        if self._mode != 'eval':
            super(CustomLossMonitor, self).end(run_context)
