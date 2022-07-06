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
"""Contextual Feature Aggregation"""

from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import ops

from src.utils import extract_patches


class RAL(nn.Cell):
    """Region affinity learning"""
    def __init__(self, kernel_size=3, stride=1, rate=2, softmax_scale=10.):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.rate = rate
        self.softmax_scale = softmax_scale
        self.escape_nan = Tensor([1e-4], mstype.float32)
        self.attn_min = Tensor([1e-8], mstype.float32)

    @staticmethod
    def _get_size_conv_transpose2d(x, w, stride, padding, dilation):
        """output size for conv transpose2d"""
        bs, _, h_in, w_in = x.shape
        _, c_out, ks, _ = w.shape
        h_out = (h_in - 1) * stride - 2 * padding + dilation * (ks - 1) + 1
        w_out = (w_in - 1) * stride - 2 * padding + dilation * (ks - 1) + 1
        return bs, c_out, h_out, w_out

    def construct(self, background, foreground):
        """construct"""
        # accelerated calculation
        bs, _, h, w = foreground.shape
        foreground = ops.ResizeBilinear(size=(h // self.rate, w // self.rate), align_corners=True)(foreground)

        foreground_size = foreground.shape
        background_size = background.shape

        background_kernel_size = 2 * self.rate
        background_patches = extract_patches(background, ksize=background_kernel_size,
                                             stride=self.stride * self.rate)
        background_patches = background_patches.view(background_size[0], -1, background_size[1],
                                                     background_kernel_size, background_kernel_size)

        background_patches_list = ops.Split(axis=0, output_num=bs)(background_patches)

        foreground_list = ops.Split(axis=0, output_num=bs)(foreground)
        foreground_patches = extract_patches(foreground,
                                             ksize=self.kernel_size,
                                             stride=self.stride)
        foreground_patches = foreground_patches.view(foreground_size[0], -1, foreground_size[1],
                                                     self.kernel_size, self.kernel_size)
        foreground_patches_list = ops.Split(axis=0, output_num=bs)(foreground_patches)

        output_list = []
        padding = 0 if self.kernel_size == 1 else 1

        for i in range(bs):
            foreground_item = foreground_list[i]
            foreground_patches_item = foreground_patches_list[i]
            background_patches_item = background_patches_list[i]

            foreground_patches_item = foreground_patches_item[0]
            foreground_patches_item_normed = foreground_patches_item / ops.Maximum()(
                ops.Sqrt()(
                    ops.ReduceSum(keep_dims=True)(
                        foreground_patches_item * foreground_patches_item,
                        [1, 2, 3]
                    )
                ),
                self.escape_nan)

            out_ch, _, ks, _ = foreground_patches_item_normed.shape
            score_map = ops.Conv2D(out_ch, ks, pad_mode='pad', pad=padding, stride=1)(
                foreground_item,
                foreground_patches_item_normed
            )
            score_map = score_map.view(1, foreground_size[2] // self.stride * foreground_size[3] // self.stride,
                                       foreground_size[2], foreground_size[3])
            attention_map = ops.Softmax(axis=1)(score_map * self.softmax_scale)
            attention_map = ops.clip_by_value(attention_map, self.attn_min, attention_map.max())

            background_patches_item = background_patches_item[0]
            out_ch, _, ks, _ = background_patches_item.shape
            output_shape = self._get_size_conv_transpose2d(attention_map, background_patches_item,
                                                           stride=self.rate, padding=1, dilation=1)
            output_item = ops.Conv2DTranspose(out_ch, ks, pad_mode='pad', pad=1, stride=self.rate)(
                attention_map,
                background_patches_item,
                output_shape
            ) / 4.
            output_list.append(output_item)

        output = ops.Concat(axis=0)(output_list)
        output = output.view(background_size)
        return output


class MSFA(nn.Cell):
    """Multi-scale feature aggregation"""
    def __init__(self, in_channels=64, out_channels=64, dilation_rate_list=(1, 2, 4, 8)):
        super().__init__()

        self.dilation_rate_list = dilation_rate_list
        dilated_convs = []
        for dilation_rate in dilation_rate_list:
            dilated_convs.append(
                nn.SequentialCell(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilation_rate,
                              pad_mode='pad', padding=dilation_rate, has_bias=True),
                    nn.ReLU()
                )
            )
        self.dilated_convs = nn.CellList(dilated_convs)

        self.weight_calc = nn.SequentialCell(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                      pad_mode='pad', padding=1, has_bias=True),
            nn.ReLU(),
            nn.Conv2d(out_channels, len(dilation_rate_list), kernel_size=1, has_bias=True),
            nn.ReLU(),
            nn.Softmax(axis=1)
        )

    def construct(self, x):
        """construct"""
        weight_map = self.weight_calc(x)

        x_feature_list = []
        for i in range(len(self.dilation_rate_list)):
            x_feature_list.append(self.dilated_convs[i](x))

        output = weight_map[:, 0:1, :, :] * x_feature_list[0] + \
                 weight_map[:, 1:2, :, :] * x_feature_list[1] + \
                 weight_map[:, 2:3, :, :] * x_feature_list[2] + \
                 weight_map[:, 3:4, :, :] * x_feature_list[3]

        return output


class CFA(nn.Cell):
    """Contextual Feature Aggregation"""
    def __init__(self, kernel_size=3, stride=1, rate=2, softmax_scale=10.,
                 in_channels=64, out_channels=64, dilation_rate_list=(1, 2, 4, 8)):
        super().__init__()

        self.ral = RAL(kernel_size=kernel_size, stride=stride,
                       rate=rate, softmax_scale=softmax_scale)
        self.msfa = MSFA(in_channels=in_channels, out_channels=out_channels,
                         dilation_rate_list=dilation_rate_list)

    def construct(self, background, foreground):
        """construct"""
        output = self.ral(background, foreground)
        output = self.msfa(output)

        return output
