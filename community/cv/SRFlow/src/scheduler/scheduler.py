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
"""
The scheduler of learning rate
"""


def warmup_lr(lr_init, lr_max, total_steps, warmup_steps, milestone, lr_decay):
    """

    Warmup LR

    Args:
        lr_init:
        lr_max:
        total_steps:
        warmup_steps:
        milestone:
        lr_decay:

    Returns: a list of learning rate

    """
    lr_each_step = []
    for i in range(int(warmup_steps)):
        lr_inc = (float(lr_max) - float(lr_init)) / float(warmup_steps)
        lr = float(lr_init) + lr_inc * (i + 1)
        lr_each_step.append(lr)
    milestone.insert(0, int(warmup_steps))
    milestone.append(total_steps)
    lr = lr_max
    for i in range(len(milestone) - 1):
        step = milestone[i + 1] - milestone[i]
        for _ in range(step):
            lr_each_step.append(lr)
        lr = lr / lr_decay
    return lr_each_step


def step_lr(lr_init, milestones, gamma, total_steps):
    """

    MutliStep LR

    Args:
        lr_init:
        milestones:
        gamma:
        total_steps:

    Returns: a list of learning rate

    """
    lr_each_step = []
    milestones.insert(0, 0)
    milestones.append(total_steps)
    lr = lr_init
    for i in range(len(milestones) - 1):
        step = milestones[i + 1] - milestones[i]
        for _ in range(step):
            lr_each_step.append(lr)
        lr = lr / gamma
    return lr_each_step
