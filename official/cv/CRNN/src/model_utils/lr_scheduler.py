# Copyright 2023 Huawei Technologies Co., Ltd
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
"""crnn lr scheduler."""
import math

def cosine_decay_lr_with_start_step(min_lr, max_lr, total_step, step_per_epoch, decay_epoch, start_step):
    r"""
    Calculates learning rate base on cosine decay function. The learning rate for each step will be stored in a list.

    For the i-th step, the formula of computing decayed_learning_rate[i] is:

    .. math::
        decayed\_learning\_rate[i] = min\_lr + 0.5 * (max\_lr - min\_lr) *
        (1 + cos(\frac{current\_epoch}{decay\_epoch}\pi))

    Where :math:`current\_epoch=floor(\frac{i}{step\_per\_epoch})`.

    Args:
        min_lr (float): The minimum value of learning rate.
        max_lr (float): The maximum value of learning rate.
        total_step (int): The total number of steps.
        step_per_epoch (int): The number of steps in per epoch.
        decay_epoch (int): Number of epochs to decay over.

    Returns:
        list[float]. The size of list is `total_step`.

    Raises:
        TypeError: If `min_lr` or `max_lr` is not a float.
        TypeError: If `total_step` or `step_per_epoch` or `decay_epoch` is not an int.
        ValueError: If `max_lr` is not greater than 0 or `min_lr` is less than 0.
        ValueError: If `total_step` or `step_per_epoch` or `decay_epoch` is less than 0.
        ValueError: If `min_lr` is greater than or equal to `max_lr`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.nn as nn
        >>>
        >>> min_lr = 0.01
        >>> max_lr = 0.1
        >>> total_step = 6
        >>> step_per_epoch = 2
        >>> decay_epoch = 2
        >>> start_step = 2
        >>> output = nn.cosine_decay_lr_with_start_step(min_lr, max_lr, total_step,
        >>>                                             step_per_epoch, decay_epoch, start_step)
        >>> print(output)
        [0.1, 0.1, 0.05500000000000001, 0.05500000000000001, 0.01, 0.01]
    """
    assert isinstance(min_lr, float) and min_lr >= 0, "min_lr should be equal or bigger than 0"
    assert isinstance(max_lr, float) and max_lr > 0, "max_lr should be bigger than 0"
    assert isinstance(total_step, int) and total_step > 0, "total_step should be bigger than 0"
    assert isinstance(step_per_epoch, int) and step_per_epoch > 0, "step_per_epoch should be bigger than 0"
    assert isinstance(decay_epoch, int) and decay_epoch > 0, "decay_epoch should be bigger than 0"
    if min_lr >= max_lr:
        raise ValueError("For 'cosine_decay_lr_with_start_step', the 'max_lr' must be greater than the 'min_lr', "
                         "but got 'max_lr' value: {}, 'min_lr' value: {}.".format(max_lr, min_lr))
    if start_step >= total_step or start_step < 0:
        raise ValueError("start_step should be less than total step")
    delta = 0.5 * (max_lr - min_lr)
    lr = []
    for i in range(total_step):
        tmp_epoch = min(math.floor(i / step_per_epoch), decay_epoch)
        lr.append(min_lr + delta * (1 + math.cos(math.pi * tmp_epoch / decay_epoch)))
    return lr[start_step:]
