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
'''optimizer'''


import mindspore.ops as P

from mindspore import dtype as mstype
from mindspore.nn.learning_rate_schedule import LearningRateSchedule


def exclude_from_weight_decay(p):
    """ exclude_from_weight_decay """
    name = p.name
    if name.find("layernorm") > -1:
        return True
    bias_suffix = ["bias", "_b", ".b_0"]
    for suffix in bias_suffix:
        if name.endswith(suffix):
            return True
    return False

class CustomWarmUpLR(LearningRateSchedule):
    """
    apply the functions to  the corresponding input fields.
    ·
    """
    def __init__(self, learning_rate, warmup_steps, max_train_steps):
        super(CustomWarmUpLR, self).__init__()
        assert isinstance(learning_rate, float) and learning_rate >= 0, "learning_rate should be equal or bigger than 0"
        assert isinstance(warmup_steps, int) and warmup_steps > 0, "warmup_steps should be bigger than 0"
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.max_train_steps = max_train_steps
        self.cast = P.Cast()

    def construct(self, current_step):
        if current_step < self.warmup_steps:
            warmup_percent = self.cast(current_step, mstype.float32) / self.warmup_steps
        else:
            warmup_percent = 1 - self.cast(current_step, mstype.float32) / self.max_train_steps

        return self.learning_rate * warmup_percent
