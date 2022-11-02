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

import logging
import numpy as np
from mindspore.train.callback import LossMonitor
from mindspore.train.summary import SummaryRecord
from mindspore import Tensor
logger = logging.getLogger(__name__)


class CustomLossMonitor(LossMonitor):
    """Own Loss Monitor that uses specified Summary Record instance"""

    def __init__(self, summary_record: SummaryRecord, mode: str):
        super(CustomLossMonitor, self).__init__()
        self._summary_record = summary_record
        self._mode = mode

    def step_end(self, run_context):
        """Called after each step finished."""
        cb_params = run_context.original_args()
        step_loss = cb_params.net_outputs
        if isinstance(step_loss, (tuple, list)) and isinstance(step_loss[0], Tensor):
            step_loss = step_loss[0]
        if isinstance(step_loss, Tensor):
            step_loss = np.mean(step_loss.asnumpy())
        self._summary_record.add_value('scalar', 'loss_' + self._mode, Tensor(step_loss))
        self._summary_record.record(cb_params.cur_step_num)

        if self._mode != 'eval':
            super(CustomLossMonitor, self).step_end(run_context)
