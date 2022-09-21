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
"""Head"""
import math
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.tensor import Tensor
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer
import numpy as np

class ScaleExp(nn.Cell):
    def __init__(self, init_value=1.0):
        super(ScaleExp, self).__init__()
        self.scale = mindspore.Parameter(mindspore.Tensor([init_value], dtype=mindspore.float32))

    def construct(self, x):
        return ops.Exp()(x * self.scale)


def bias_init_zeros(shape):
    """Bias init method."""
    return Tensor(np.array(np.zeros(shape).astype(np.float32)))


def _conv(in_channels, out_channels, kernel_size=3, stride=1, padding=0, pad_mode='pad', has_bias=True):
    """Conv2D wrapper."""
    shape = (out_channels, in_channels, kernel_size, kernel_size)
    weights = initializer("normal", shape=shape, dtype=mstype.float32).init_data()
    shape_bias = (out_channels,)
    biass = bias_init_zeros(shape_bias)
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     pad_mode=pad_mode, weight_init=weights, has_bias=has_bias, bias_init=biass)


class ClsCntRegHead(nn.Cell):
    def __init__(self, in_channel, class_num, GN=True, cnt_on_reg=True, prior=0.01):
        super(ClsCntRegHead, self).__init__()
        self.prior = prior
        self.class_num = class_num
        self.cnt_on_reg = cnt_on_reg
        cls_branch = []
        reg_branch = []
        i = 4
        while i:
            i -= 1
            cls_branch.append(
                _conv(in_channel, in_channel, stride=1, pad_mode='pad', kernel_size=3, padding=1, has_bias=True))
            if GN:
                cls_branch.append(nn.GroupNorm(32, in_channel))
            cls_branch.append(nn.ReLU())
            reg_branch.append(
                _conv(in_channel, in_channel, stride=1, pad_mode='pad', kernel_size=3, padding=1, has_bias=True))
            if GN:
                reg_branch.append(nn.GroupNorm(32, in_channel))
            reg_branch.append(nn.ReLU())
        self.cls_conv = nn.SequentialCell(cls_branch)
        self.reg_conv = nn.SequentialCell(reg_branch)
        self.cls_logits = _conv(in_channel, class_num, pad_mode='pad', kernel_size=3, padding=1, has_bias=True)
        self.cnt_logits = _conv(in_channel, 1, pad_mode='pad', kernel_size=3, padding=1, has_bias=True)
        self.reg_pred = _conv(in_channel, 4, pad_mode='pad', kernel_size=3, padding=1, has_bias=True)
        constant_init = mindspore.common.initializer.Constant(-math.log((1 - prior) / prior))
        constant_init(self.cls_logits.bias)
        self.scale_exp = nn.CellList([ScaleExp(1.0) for _ in range(5)])

    def construct(self, inputs):
        '''inputs:[P3~P7]'''
        cls_logits = ()
        cnt_logits = ()
        reg_preds = ()
        for index, P in enumerate(inputs):
            cls_conv_out = self.cls_conv(P)
            reg_conv_out = self.reg_conv(P)
            cls_logits = cls_logits + (self.cls_logits(cls_conv_out),)
            if not self.cnt_on_reg:
                cnt_logits = cnt_logits + (self.cnt_logits(cls_conv_out),)
            else:
                cnt_logits = cnt_logits + (self.cnt_logits(reg_conv_out),)
            reg_preds = reg_preds + (self.scale_exp[index](self.reg_pred(reg_conv_out)),)
        return cls_logits, cnt_logits, reg_preds
