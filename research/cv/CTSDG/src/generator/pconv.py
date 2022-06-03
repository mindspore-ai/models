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
"""Partial convolution"""

from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import ops


class PartialConv2d(nn.Cell):
    """Partial conv2d"""
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=0, bias=False, multi_channel=True):
        super().__init__()
        self.multi_channel = multi_channel

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, pad_mode='pad', padding=padding, has_bias=bias)
        self.mask_update_conv_ops = ops.Conv2D(out_channel=out_channels, kernel_size=kernel_size,
                                               pad_mode='pad', pad=padding, stride=stride)
        if self.multi_channel:
            self.weight_mask_updater = ops.Ones()((out_channels, in_channels,
                                                   kernel_size, kernel_size), mstype.float32)
        else:
            self.weight_mask_updater = ops.Ones()((1, 1, kernel_size, kernel_size), mstype.float32)

        self.slide_winsize = (self.weight_mask_updater.shape[1] * self.weight_mask_updater.shape[2] *
                              self.weight_mask_updater.shape[3])

        self.clip_value_min = Tensor(0, mstype.float32)
        self.clip_value_max = Tensor(1, mstype.float32)

    def construct(self, inp, mask):
        """construct"""
        update_mask = ops.stop_gradient(self.mask_update_conv_ops(mask, self.weight_mask_updater))
        mask_ratio = self.slide_winsize / (update_mask + 1e-8)
        update_mask = ops.clip_by_value(update_mask, self.clip_value_min, self.clip_value_max)
        mask_ratio = ops.Mul()(mask_ratio, update_mask)

        raw_out = self.conv(ops.Mul()(inp, mask))
        _, out_channels, _, _ = raw_out.shape
        if self.conv.bias is not None:
            bias_view = self.conv.bias.view(1, out_channels, 1, 1)
            output = ops.Mul()(raw_out - bias_view, mask_ratio) + bias_view
            output = ops.Mul()(output, update_mask)
        else:
            output = ops.Mul()(raw_out, mask_ratio)
        return output, update_mask


class PConvBNActiv(nn.Cell):
    """PConv-BatchNorm-Activation"""
    def __init__(self, in_channels, out_channels, bn=True,
                 sample='none-3', activation='relu', bias=False):
        super().__init__()

        if sample == 'down-7':
            self.conv = PartialConv2d(in_channels, out_channels, kernel_size=7,
                                      stride=2, padding=3, bias=bias, multi_channel=True)
        elif sample == 'down-5':
            self.conv = PartialConv2d(in_channels, out_channels, kernel_size=5,
                                      stride=2, padding=2, bias=bias, multi_channel=True)
        elif sample == 'down-3':
            self.conv = PartialConv2d(in_channels, out_channels, kernel_size=3,
                                      stride=2, padding=1, bias=bias, multi_channel=True)
        else:
            self.conv = PartialConv2d(in_channels, out_channels, kernel_size=3,
                                      stride=1, padding=1, bias=bias, multi_channel=True)
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)

        self.activation = None
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky':
            self.activation = nn.LeakyReLU(alpha=0.2)

    def construct(self, images, masks):
        """construct"""
        images, masks = self.conv(images, masks)
        if self.bn:
            images = self.bn(images)
        if self.activation:
            images = self.activation(images)

        return images, masks
