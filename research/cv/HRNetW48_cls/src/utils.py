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
"""Tool functions."""
import math

import numpy as np
from mindspore import Tensor
from mindspore import nn
from mindspore.common import initializer


# Functions for params initialization.
def calculate_fan_in_and_fan_out(shape):
    """
    calculate fan_in and fan_out

    Args:
        shape (tuple): input shape.

    Returns:
        Tuple, a tuple with two elements, the first element is `n_in` and the second element is `n_out`.
    """
    dimensions = len(shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    if dimensions == 2:
        fan_in = shape[1]
        fan_out = shape[0]
    else:
        num_input_fmaps = shape[1]
        num_output_fmaps = shape[0]
        receptive_field_size = 1
        if dimensions > 2:
            receptive_field_size = shape[2] * shape[3]
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def get_conv_bias(cell):
    """Bias initializer for conv."""
    weight = initializer.initializer(initializer.HeUniform(negative_slope=math.sqrt(5)),
                                     cell.weight.shape, cell.weight.dtype)
    fan_in, _ = calculate_fan_in_and_fan_out(weight.shape)
    bound = 1 / math.sqrt(fan_in)
    return initializer.initializer(initializer.Uniform(scale=bound),
                                   cell.bias.shape, cell.bias.dtype)


def params_initializer(config, net):
    """Model parameter initializer."""
    for _, cell in net.cells_and_names():
        if isinstance(cell, nn.Conv2d):
            if config.conv_init == "XavierUniform":
                cell.weight.set_data(initializer.initializer(initializer.XavierUniform(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
            elif config.conv_init == "TruncatedNormal":
                cell.weight.set_data(initializer.initializer(initializer.TruncatedNormal(sigma=0.027),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
            if cell.has_bias:
                cell.bias.set_data(get_conv_bias(cell))

        if isinstance(cell, nn.BatchNorm2d):
            cell.gamma.set_data(initializer.initializer(1,
                                                        cell.gamma.shape,
                                                        cell.gamma.dtype))
            cell.beta.set_data(initializer.initializer(0,
                                                       cell.beta.shape,
                                                       cell.beta.dtype))

        if isinstance(cell, nn.Dense):
            if config.dense_init == "TruncatedNormal":
                cell.weight.set_data(initializer.initializer(initializer.TruncatedNormal(sigma=0.027),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
            elif config.dense_init == "RandomNormal":
                in_channel = cell.in_channels
                out_channel = cell.out_channels
                weight = np.random.normal(loc=0, scale=0.01, size=out_channel * in_channel)
                weight = Tensor(np.reshape(weight, (out_channel, in_channel)), dtype=cell.weight.dtype)
                cell.weight.set_data(weight)


# Functions for learning rate generation.
def get_linear_lr(base_lr, total_epoch, spe, lr_init, lr_end, warmup_epoch=0):
    """Get learning rates decay in linear."""
    lr_each_step = []
    total_steps = spe * total_epoch
    warmup_steps = spe * warmup_epoch
    for i in range(total_steps):
        if i < warmup_steps:
            lr = lr_init + (base_lr - lr_init) * i / warmup_steps
        else:
            lr = base_lr - (base_lr - lr_end) * (i - warmup_steps) / (total_steps - warmup_steps)
        lr_each_step.append(lr)
    return lr_each_step
