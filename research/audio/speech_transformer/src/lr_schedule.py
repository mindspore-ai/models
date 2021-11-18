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
"""Learning rate utilities."""


def rsqrt_hidden(hidden_size):
    """rsqrt of hidden size"""
    return float(hidden_size) ** -0.5


def create_dynamic_lr(training_steps, k, warmup_steps, hidden_size):
    """
    Generate dynamic learning rate.
    """
    lr = []
    for current_step in range(1, training_steps+1):
        cur_lr = rsqrt_hidden(hidden_size) * k * min(current_step ** (-0.5), current_step * (warmup_steps ** (-1.5)))
        lr.append(cur_lr)
    return lr
