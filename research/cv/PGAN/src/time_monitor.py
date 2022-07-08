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
"""TimeMonitor class."""

import time


class TimeMonitor:
    """
    Monitor the time in training.

    steps_log_interval (int): Number of steps between logging the intermediate performance values.
    """

    def __init__(self, steps_log_interval):
        self.num_steps = 0
        self.num_epochs = 0
        self.steps_log_interval = steps_log_interval
        self.epoch_time = None
        self.step_start_time = None
        self.data_iter_time_mark = None
        self.steps_accumulated_time = 0
        self.data_iter_accumulated_time = 0

    def epoch_begin(self):
        """
        Record time at the begin of epoch.
        """
        self.num_steps = 0
        self.steps_accumulated_time = 0
        self.data_iter_accumulated_time = 0
        self.epoch_time = time.time()
        self.data_iter_time_mark = self.epoch_time

    def step_start(self):
        self.step_start_time = time.time()
        self.data_iter_accumulated_time += self.step_start_time - self.data_iter_time_mark
        if self.num_steps == 0:
            print(f'Dataset first iteration time: {self.data_iter_accumulated_time * 1000:5.3f} ms', flush=True)

    def data_iter_end(self):
        """Record the time of the data iteration end

        (for computing the data loader time)
        """
        self.data_iter_time_mark = time.time()

    def step_end(self):
        """Step end callback"""
        self.num_steps += 1

        self.steps_accumulated_time += time.time() - self.step_start_time

        if self.num_steps % self.steps_log_interval == 0:
            print(
                f'Intermediate: epoch {self.num_epochs} step {self.num_steps}, '
                f'per_step_time {self.steps_accumulated_time / self.steps_log_interval * 1000:5.3f} ms, '
                f'(not including the data loader time per step '
                f'{self.data_iter_accumulated_time / self.steps_log_interval * 1000:5.3f} ms)',
                flush=True,
            )
            self.steps_accumulated_time = 0
            self.data_iter_accumulated_time = 0

    def epoch_end(self):
        """
        Print process cost time at the end of epoch.
        """
        if self.epoch_time is None:
            return

        epoch_seconds = (time.time() - self.epoch_time) * 1000

        if not isinstance(self.num_steps, int) or self.num_steps < 1:
            raise ValueError("data_size must be positive int.")

        step_seconds = epoch_seconds / self.num_steps
        print(f"epoch {self.num_epochs} time: {epoch_seconds:5.3f} ms, "
              f"per step time: {step_seconds:5.3f} ms")
        self.num_epochs += 1
        self.epoch_time = None
