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
"""Training callbacks."""

import time
import numpy as np

from mindspore.train.callback import Callback
from mindspore import Tensor


class TimeLossMonitor(Callback):
    """
    Monitor loss and time.

    Args:
        lr_init (numpy array): train lr

    Returns:
        None

    Examples:
        >>> TimeLossMonitor(100,lr_init=Tensor([0.05]*100).asnumpy())
    """

    def __init__(self, lr_init=None):
        super().__init__()
        self.lr_init = lr_init
        self.losses = []
        self.epoch_time = 0
        self.step_time = 0
        self.steps_made = 0

    def begin(self, run_context):
        print('Training start')

    def epoch_begin(self, run_context):
        """Epoch begin."""
        self.losses = []
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        """Epoch end."""
        cb_params = run_context.original_args()

        cur_epoch = cb_params.cur_epoch_num
        tot_epoch = cb_params.epoch_num
        epoch_seconds = (time.time() - self.epoch_time)
        batch_num = cb_params.batch_num
        per_step_mseconds = epoch_seconds / cb_params.batch_num * 1000
        mean_loss = np.mean(self.losses)
        cur_lr = self.lr_init[cb_params.cur_step_num - 1]
        print(f"epoch: [{cur_epoch:3d}/{tot_epoch:3d}], epoch time: {epoch_seconds:5.1f} s, "
              f"steps: {batch_num:5d}, per step time: {per_step_mseconds:5.3f} ms, "
              f"avg loss: {mean_loss:.5f}, lr: {cur_lr:8.6f}",
              flush=True)

    def step_begin(self, run_context):
        """Step begin."""
        self.step_time = time.time()
        self.steps_made += 1


    def step_end(self, run_context):
        """step end"""
        cb_params = run_context.original_args()
        step_loss = cb_params.net_outputs

        if isinstance(step_loss, (tuple, list)) and isinstance(step_loss[0], Tensor):
            step_loss = step_loss[0]
        if isinstance(step_loss, Tensor):
            step_loss = np.mean(step_loss.asnumpy())

        # Step time measurement, works only in dataset_sink_mode=False. Uncomment for debugging
        # step_time = (time.time() - self.step_time) * 1000
        # print(f'Step: {self.steps_made}, Loss: {step_loss}, step_time: {step_time} ms')

        self.losses.append(step_loss)
