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

"""Metric calculation"""

import numpy as np
from mindspore.nn import Metric
from mindspore.nn import rearrange_inputs
from src.box_utils import default_boxes
from src.model_utils.config import config as cfg


class MymAP(Metric):
    """
    Calc mean average precision metric.

    Returns:
        mAP
    """

    def __init__(self):
        super(MymAP, self).__init__()
        self.default_boxes = default_boxes
        self.prior_scaling_xy = cfg.prior_scaling[0]
        self.prior_scaling_wh = cfg.prior_scaling[1]
        self.clear()

    def clear(self):
        """Clears the internal evaluation result."""
        self._abs_error_sum = 0
        self._samples_num = 0

    @rearrange_inputs
    def update(self, *inputs):
        if len(inputs) != 2:
            raise ValueError('Mean absolute error need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))
        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])
        abs_error_sum = np.abs(y.reshape(y_pred.shape) - y_pred)
        self._abs_error_sum += abs_error_sum.sum()
        self._samples_num += y.shape[0]

    def eval(self):
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return self._abs_error_sum / self._samples_num
