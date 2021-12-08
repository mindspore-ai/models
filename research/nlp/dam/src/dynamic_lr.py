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
# ===========================================================================
"""Dynamic learning Rate"""


class ExponentialDecayLR:
    r"""
    Calculates learning rate base on exponential decay function.

    For the i-th step, the formula of computing decayed_learning_rate[i] is:

    .. math::
        decayed\_learning\_rate[i] = learning\_rate * decay\_rate^{p}

    Where :

    .. math::
        p = \frac{current\_step}{decay\_steps}

    If `is_stair` is True, the formula is :

    .. math::
        p = floor(\frac{current\_step}{decay\_steps})

    Args:
        learning_rate (float): The initial value of learning rate.
        decay_rate (float): The decay rate.
        decay_steps (int): A value used to calculate decayed learning rate.
        is_stair (bool): If true, learning rate is decayed once every `decay_steps` time. Default: False.

    """
    def __init__(self, learning_rate, decay_rate, decay_steps, is_stair=True):
        super(ExponentialDecayLR, self).__init__()
        print("Using Exponential Decay LR Scheduler")
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.is_stair = is_stair

    def __call__(self, iteration):
        p = iteration / self.decay_steps
        if self.is_stair:
            p = int(p)
        return self.learning_rate * (self.decay_rate ** p)


def exponential_decay_lr(learning_rate, decay_rate, decay_steps, max_iteration, is_stair=True):
    """
    Generate a list of learning rates.

    :param learning_rate: Initial learning rate
    :param decay_rate: The decay rate.
    :param decay_steps: A value used to calculate decayed learning rate.
    :param max_iteration: Maximum iteration
    :param is_stair: If true, learning rate is decayed once every `decay_steps` time. Default: False.
    :return: List
    """
    lr_scheduler = ExponentialDecayLR(learning_rate, decay_rate, decay_steps, is_stair=is_stair)
    lr = []
    for i in range(max_iteration):
        lr.append(lr_scheduler(i))
    return lr
