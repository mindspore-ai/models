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

"""The M-Net of Semantic Human Matting Network"""
import mindspore.nn as nn
from mindspore.ops import operations as P


class M_net(nn.Cell):
    """M-Net architecture: encoder + decoder"""

    def __init__(self):
        super(M_net, self).__init__()
        # -----------------------------------------------------------------
        # encoder
        # ---------------------
        # 1/2 ——> shape scale: (n, c, h, w) ——> (n, c', h * 1/2, w * 1/2)
        self.en_conv_bn_relu_1 = nn.SequentialCell(
            [
                nn.Conv2d(6, 16, 3, 1, padding=1, pad_mode="pad", has_bias=False, weight_init="Uniform"),
                nn.BatchNorm2d(16),
                nn.ReLU(),
            ]
        )
        self.max_pooling_1 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid")

        # 1/4
        self.en_conv_bn_relu_2 = nn.SequentialCell(
            [
                nn.Conv2d(16, 32, 3, 1, padding=1, pad_mode="pad", has_bias=False, weight_init="Uniform"),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            ]
        )
        self.max_pooling_2 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid")

        # 1/8
        self.en_conv_bn_relu_3 = nn.SequentialCell(
            [
                nn.Conv2d(32, 64, 3, 1, padding=1, pad_mode="pad", has_bias=False, weight_init="Uniform"),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            ]
        )
        self.max_pooling_3 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid")

        # 1/16
        self.en_conv_bn_relu_4 = nn.SequentialCell(
            [
                nn.Conv2d(64, 128, 3, 1, padding=1, pad_mode="pad", has_bias=False, weight_init="Uniform"),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            ]
        )
        self.max_pooling_4 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid")

        # -----------------------------------------------------------------
        # decoder
        # ---------------------
        # 1/8
        self.de_conv_bn_relu_1 = nn.SequentialCell(
            [
                nn.Conv2d(128, 128, 3, 1, padding=1, pad_mode="pad", has_bias=False, weight_init="Uniform"),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            ]
        )
        self.deconv_1 = nn.Conv2dTranspose(
            128, 128, 5, 2, pad_mode="pad", padding=2, has_bias=False, weight_init="Uniform"
        )

        # 1/4
        self.de_conv_bn_relu_2 = nn.SequentialCell(
            [
                nn.Conv2d(128, 64, 3, 1, padding=1, pad_mode="pad", has_bias=False, weight_init="Uniform"),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            ]
        )
        self.deconv_2 = nn.Conv2dTranspose(
            64, 64, 5, 2, pad_mode="pad", padding=2, has_bias=False, weight_init="Uniform"
        )

        # 1/2
        self.de_conv_bn_relu_3 = nn.SequentialCell(
            [
                nn.Conv2d(64, 32, 3, 1, padding=1, pad_mode="pad", has_bias=False, weight_init="Uniform"),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            ]
        )
        self.deconv_3 = nn.Conv2dTranspose(
            32, 32, 5, 2, pad_mode="pad", padding=2, has_bias=False, weight_init="Uniform"
        )

        # 1/1
        self.de_conv_bn_relu_4 = nn.SequentialCell(
            [
                nn.Conv2d(32, 16, 3, 1, padding=1, pad_mode="pad", has_bias=False, weight_init="Uniform"),
                nn.BatchNorm2d(16),
                nn.ReLU(),
            ]
        )
        self.deconv_4 = nn.Conv2dTranspose(
            16, 16, 5, 2, pad_mode="pad", padding=2, has_bias=False, weight_init="Uniform"
        )

        self.conv = nn.Conv2d(16, 1, 5, 1, padding=2, pad_mode="pad", has_bias=False, weight_init="Uniform")
        self.pad = P.Pad(((0, 0), (0, 0), (1, 1), (1, 1)))
        self.pad_tp = P.Pad(((0, 0), (0, 0), (0, 1), (0, 1)))

    def construct(self, inp):
        # -----------------------------------------------------------------
        # encoder
        # ---------------------
        x = self.en_conv_bn_relu_1(inp)
        x = self.pad(x)
        x = self.max_pooling_1(x)

        x = self.en_conv_bn_relu_2(x)
        x = self.pad(x)
        x = self.max_pooling_2(x)

        x = self.en_conv_bn_relu_3(x)
        x = self.pad(x)
        x = self.max_pooling_3(x)

        x = self.en_conv_bn_relu_4(x)
        x = self.pad(x)
        x = self.max_pooling_4(x)

        # -----------------------------------------------------------------
        # decoder
        # ---------------------
        x = self.de_conv_bn_relu_1(x)
        x = self.pad_tp(x)
        x = self.deconv_1(x)
        x = x[:, :, :-1, :-1]

        x = self.de_conv_bn_relu_2(x)
        x = self.pad_tp(x)
        x = self.deconv_2(x)
        x = x[:, :, :-1, :-1]

        x = self.de_conv_bn_relu_3(x)
        x = self.pad_tp(x)
        x = self.deconv_3(x)
        x = x[:, :, :-1, :-1]

        x = self.de_conv_bn_relu_4(x)
        x = self.pad_tp(x)
        x = self.deconv_4(x)
        x = x[:, :, :-1, :-1]

        # raw alpha pred
        out = self.conv(x)

        return out
