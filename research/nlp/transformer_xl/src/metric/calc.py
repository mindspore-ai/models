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

import math
from mindspore.nn import Metric


def bpc(loss):
    return loss / math.log(2)


def ppl(loss):
    return math.exp(loss)


class BPC(Metric):
    def __init__(self):
        super(BPC, self).__init__()

        self.loss = 0.0
        self.log_2 = math.log(2)

    def clear(self):
        """Clears the internal evaluation result."""
        self.loss = 0.0

    def update(self, loss):
        self.loss = loss

    def eval(self):
        return self.loss / self.log_2


class PPL(Metric):
    def __init__(self):
        super(PPL, self).__init__()
        self.loss = 0.0

    def clear(self):
        """Clears the internal evaluation result."""
        self.loss = 0.0

    def update(self, loss):
        self.loss = loss

    def eval(self):
        return math.exp(self.loss)
