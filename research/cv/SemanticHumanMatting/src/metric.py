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

"""Calculate metric"""
import numpy as np
from mindspore.nn import Metric


class Sad(Metric):
    """
    Sad Metric
    """

    def __init__(self):
        super(Sad, self).__init__()
        self.clear()

    def clear(self):
        """Clears the internal evaluation result."""
        self._sad_sum = 0
        self._samples_num = 0
        self._loss = 0

    def update(self, *inputs):
        """Update metric per step"""
        if len(inputs) != 3:
            raise ValueError("Sad need 3 inputs (alpha_pre, alpha_gt, loss), but got {}".format(len(inputs)))

        alpha_pre = self._convert_data(inputs[0])
        alpha_gt = self._convert_data(inputs[1])
        loss = self._convert_data(inputs[2])
        sad_ = np.abs(alpha_pre - alpha_gt).sum() / 1000

        self._sad_sum += sad_
        self._samples_num += 1
        self._loss += loss

    def eval(self):
        """Calculate metric"""
        if self._samples_num == 0:
            raise RuntimeError("Total samples num must not be 0.")

        return self._sad_sum / self._samples_num, self._loss / self._samples_num
