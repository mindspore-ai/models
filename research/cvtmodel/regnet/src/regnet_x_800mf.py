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

import mindspore.ops as P
from mindspore import nn


class Module1(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_0_kernel_size, conv2d_0_stride,
                 conv2d_0_padding, conv2d_0_pad_mode, conv2d_0_group):
        super(Module1, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=conv2d_0_kernel_size,
                                  stride=conv2d_0_stride,
                                  padding=conv2d_0_padding,
                                  pad_mode=conv2d_0_pad_mode,
                                  dilation=(1, 1),
                                  group=conv2d_0_group,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        return opt_relu_1


class Module9(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, module1_0_conv2d_0_in_channels,
                 module1_0_conv2d_0_out_channels, module1_0_conv2d_0_kernel_size, module1_0_conv2d_0_stride,
                 module1_0_conv2d_0_padding, module1_0_conv2d_0_pad_mode, module1_0_conv2d_0_group,
                 module1_1_conv2d_0_in_channels, module1_1_conv2d_0_out_channels, module1_1_conv2d_0_kernel_size,
                 module1_1_conv2d_0_stride, module1_1_conv2d_0_padding, module1_1_conv2d_0_pad_mode,
                 module1_1_conv2d_0_group):
        super(Module9, self).__init__()
        self.module1_0 = Module1(conv2d_0_in_channels=module1_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module1_0_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module1_0_conv2d_0_kernel_size,
                                 conv2d_0_stride=module1_0_conv2d_0_stride,
                                 conv2d_0_padding=module1_0_conv2d_0_padding,
                                 conv2d_0_pad_mode=module1_0_conv2d_0_pad_mode,
                                 conv2d_0_group=module1_0_conv2d_0_group)
        self.module1_1 = Module1(conv2d_0_in_channels=module1_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module1_1_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module1_1_conv2d_0_kernel_size,
                                 conv2d_0_stride=module1_1_conv2d_0_stride,
                                 conv2d_0_padding=module1_1_conv2d_0_padding,
                                 conv2d_0_pad_mode=module1_1_conv2d_0_pad_mode,
                                 conv2d_0_group=module1_1_conv2d_0_group)
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)

    def construct(self, x):
        module1_0_opt = self.module1_0(x)
        module1_1_opt = self.module1_1(module1_0_opt)
        opt_conv2d_0 = self.conv2d_0(module1_1_opt)
        return opt_conv2d_0


class Module0(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_2_in_channels, conv2d_2_out_channels,
                 conv2d_2_group, conv2d_4_in_channels, conv2d_4_out_channels):
        super(Module0, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()
        self.conv2d_2 = nn.Conv2d(in_channels=conv2d_2_in_channels,
                                  out_channels=conv2d_2_out_channels,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=conv2d_2_group,
                                  has_bias=True)
        self.relu_3 = nn.ReLU()
        self.conv2d_4 = nn.Conv2d(in_channels=conv2d_4_in_channels,
                                  out_channels=conv2d_4_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_6 = nn.ReLU()

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_conv2d_2 = self.conv2d_2(opt_relu_1)
        opt_relu_3 = self.relu_3(opt_conv2d_2)
        opt_conv2d_4 = self.conv2d_4(opt_relu_3)
        opt_add_5 = P.Add()(x, opt_conv2d_4)
        opt_relu_6 = self.relu_6(opt_add_5)
        return opt_relu_6


class Module5(nn.Cell):
    def __init__(self, module0_0_conv2d_0_in_channels, module0_0_conv2d_0_out_channels, module0_0_conv2d_2_in_channels,
                 module0_0_conv2d_2_out_channels, module0_0_conv2d_2_group, module0_0_conv2d_4_in_channels,
                 module0_0_conv2d_4_out_channels, module0_1_conv2d_0_in_channels, module0_1_conv2d_0_out_channels,
                 module0_1_conv2d_2_in_channels, module0_1_conv2d_2_out_channels, module0_1_conv2d_2_group,
                 module0_1_conv2d_4_in_channels, module0_1_conv2d_4_out_channels):
        super(Module5, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=module0_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_0_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_0_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_0_conv2d_2_out_channels,
                                 conv2d_2_group=module0_0_conv2d_2_group,
                                 conv2d_4_in_channels=module0_0_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_0_conv2d_4_out_channels)
        self.module0_1 = Module0(conv2d_0_in_channels=module0_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_1_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_1_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_1_conv2d_2_out_channels,
                                 conv2d_2_group=module0_1_conv2d_2_group,
                                 conv2d_4_in_channels=module0_1_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_1_conv2d_4_out_channels)

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        module0_1_opt = self.module0_1(module0_0_opt)
        return module0_1_opt


class Module11(nn.Cell):
    def __init__(self, module0_0_conv2d_0_in_channels, module0_0_conv2d_0_out_channels, module0_0_conv2d_2_in_channels,
                 module0_0_conv2d_2_out_channels, module0_0_conv2d_2_group, module0_0_conv2d_4_in_channels,
                 module0_0_conv2d_4_out_channels, module0_1_conv2d_0_in_channels, module0_1_conv2d_0_out_channels,
                 module0_1_conv2d_2_in_channels, module0_1_conv2d_2_out_channels, module0_1_conv2d_2_group,
                 module0_1_conv2d_4_in_channels, module0_1_conv2d_4_out_channels, module0_2_conv2d_0_in_channels,
                 module0_2_conv2d_0_out_channels, module0_2_conv2d_2_in_channels, module0_2_conv2d_2_out_channels,
                 module0_2_conv2d_2_group, module0_2_conv2d_4_in_channels, module0_2_conv2d_4_out_channels,
                 module0_3_conv2d_0_in_channels, module0_3_conv2d_0_out_channels, module0_3_conv2d_2_in_channels,
                 module0_3_conv2d_2_out_channels, module0_3_conv2d_2_group, module0_3_conv2d_4_in_channels,
                 module0_3_conv2d_4_out_channels):
        super(Module11, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=module0_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_0_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_0_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_0_conv2d_2_out_channels,
                                 conv2d_2_group=module0_0_conv2d_2_group,
                                 conv2d_4_in_channels=module0_0_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_0_conv2d_4_out_channels)
        self.module0_1 = Module0(conv2d_0_in_channels=module0_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_1_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_1_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_1_conv2d_2_out_channels,
                                 conv2d_2_group=module0_1_conv2d_2_group,
                                 conv2d_4_in_channels=module0_1_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_1_conv2d_4_out_channels)
        self.module0_2 = Module0(conv2d_0_in_channels=module0_2_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_2_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_2_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_2_conv2d_2_out_channels,
                                 conv2d_2_group=module0_2_conv2d_2_group,
                                 conv2d_4_in_channels=module0_2_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_2_conv2d_4_out_channels)
        self.module0_3 = Module0(conv2d_0_in_channels=module0_3_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_3_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_3_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_3_conv2d_2_out_channels,
                                 conv2d_2_group=module0_3_conv2d_2_group,
                                 conv2d_4_in_channels=module0_3_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_3_conv2d_4_out_channels)

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        module0_1_opt = self.module0_1(module0_0_opt)
        module0_2_opt = self.module0_2(module0_1_opt)
        module0_3_opt = self.module0_3(module0_2_opt)
        return module0_3_opt


class MindSporeModel(nn.Cell):
    def __init__(self):
        super(MindSporeModel, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=3,
                                  out_channels=32,
                                  kernel_size=(3, 3),
                                  stride=(2, 2),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()
        self.conv2d_2 = nn.Conv2d(in_channels=32,
                                  out_channels=64,
                                  kernel_size=(1, 1),
                                  stride=(2, 2),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.module9_0 = Module9(conv2d_0_in_channels=64,
                                 conv2d_0_out_channels=64,
                                 module1_0_conv2d_0_in_channels=32,
                                 module1_0_conv2d_0_out_channels=64,
                                 module1_0_conv2d_0_kernel_size=(1, 1),
                                 module1_0_conv2d_0_stride=(1, 1),
                                 module1_0_conv2d_0_padding=0,
                                 module1_0_conv2d_0_pad_mode="valid",
                                 module1_0_conv2d_0_group=1,
                                 module1_1_conv2d_0_in_channels=64,
                                 module1_1_conv2d_0_out_channels=64,
                                 module1_1_conv2d_0_kernel_size=(3, 3),
                                 module1_1_conv2d_0_stride=(2, 2),
                                 module1_1_conv2d_0_padding=(1, 1, 1, 1),
                                 module1_1_conv2d_0_pad_mode="pad",
                                 module1_1_conv2d_0_group=4)
        self.relu_9 = nn.ReLU()
        self.conv2d_10 = nn.Conv2d(in_channels=64,
                                   out_channels=128,
                                   kernel_size=(1, 1),
                                   stride=(2, 2),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.module9_1 = Module9(conv2d_0_in_channels=128,
                                 conv2d_0_out_channels=128,
                                 module1_0_conv2d_0_in_channels=64,
                                 module1_0_conv2d_0_out_channels=128,
                                 module1_0_conv2d_0_kernel_size=(1, 1),
                                 module1_0_conv2d_0_stride=(1, 1),
                                 module1_0_conv2d_0_padding=0,
                                 module1_0_conv2d_0_pad_mode="valid",
                                 module1_0_conv2d_0_group=1,
                                 module1_1_conv2d_0_in_channels=128,
                                 module1_1_conv2d_0_out_channels=128,
                                 module1_1_conv2d_0_kernel_size=(3, 3),
                                 module1_1_conv2d_0_stride=(2, 2),
                                 module1_1_conv2d_0_padding=(1, 1, 1, 1),
                                 module1_1_conv2d_0_pad_mode="pad",
                                 module1_1_conv2d_0_group=8)
        self.relu_17 = nn.ReLU()
        self.module5_0 = Module5(module0_0_conv2d_0_in_channels=128,
                                 module0_0_conv2d_0_out_channels=128,
                                 module0_0_conv2d_2_in_channels=128,
                                 module0_0_conv2d_2_out_channels=128,
                                 module0_0_conv2d_2_group=8,
                                 module0_0_conv2d_4_in_channels=128,
                                 module0_0_conv2d_4_out_channels=128,
                                 module0_1_conv2d_0_in_channels=128,
                                 module0_1_conv2d_0_out_channels=128,
                                 module0_1_conv2d_2_in_channels=128,
                                 module0_1_conv2d_2_out_channels=128,
                                 module0_1_conv2d_2_group=8,
                                 module0_1_conv2d_4_in_channels=128,
                                 module0_1_conv2d_4_out_channels=128)
        self.conv2d_32 = nn.Conv2d(in_channels=128,
                                   out_channels=288,
                                   kernel_size=(1, 1),
                                   stride=(2, 2),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.module9_2 = Module9(conv2d_0_in_channels=288,
                                 conv2d_0_out_channels=288,
                                 module1_0_conv2d_0_in_channels=128,
                                 module1_0_conv2d_0_out_channels=288,
                                 module1_0_conv2d_0_kernel_size=(1, 1),
                                 module1_0_conv2d_0_stride=(1, 1),
                                 module1_0_conv2d_0_padding=0,
                                 module1_0_conv2d_0_pad_mode="valid",
                                 module1_0_conv2d_0_group=1,
                                 module1_1_conv2d_0_in_channels=288,
                                 module1_1_conv2d_0_out_channels=288,
                                 module1_1_conv2d_0_kernel_size=(3, 3),
                                 module1_1_conv2d_0_stride=(2, 2),
                                 module1_1_conv2d_0_padding=(1, 1, 1, 1),
                                 module1_1_conv2d_0_pad_mode="pad",
                                 module1_1_conv2d_0_group=18)
        self.relu_39 = nn.ReLU()
        self.module11_0 = Module11(module0_0_conv2d_0_in_channels=288,
                                   module0_0_conv2d_0_out_channels=288,
                                   module0_0_conv2d_2_in_channels=288,
                                   module0_0_conv2d_2_out_channels=288,
                                   module0_0_conv2d_2_group=18,
                                   module0_0_conv2d_4_in_channels=288,
                                   module0_0_conv2d_4_out_channels=288,
                                   module0_1_conv2d_0_in_channels=288,
                                   module0_1_conv2d_0_out_channels=288,
                                   module0_1_conv2d_2_in_channels=288,
                                   module0_1_conv2d_2_out_channels=288,
                                   module0_1_conv2d_2_group=18,
                                   module0_1_conv2d_4_in_channels=288,
                                   module0_1_conv2d_4_out_channels=288,
                                   module0_2_conv2d_0_in_channels=288,
                                   module0_2_conv2d_0_out_channels=288,
                                   module0_2_conv2d_2_in_channels=288,
                                   module0_2_conv2d_2_out_channels=288,
                                   module0_2_conv2d_2_group=18,
                                   module0_2_conv2d_4_in_channels=288,
                                   module0_2_conv2d_4_out_channels=288,
                                   module0_3_conv2d_0_in_channels=288,
                                   module0_3_conv2d_0_out_channels=288,
                                   module0_3_conv2d_2_in_channels=288,
                                   module0_3_conv2d_2_out_channels=288,
                                   module0_3_conv2d_2_group=18,
                                   module0_3_conv2d_4_in_channels=288,
                                   module0_3_conv2d_4_out_channels=288)
        self.module5_1 = Module5(module0_0_conv2d_0_in_channels=288,
                                 module0_0_conv2d_0_out_channels=288,
                                 module0_0_conv2d_2_in_channels=288,
                                 module0_0_conv2d_2_out_channels=288,
                                 module0_0_conv2d_2_group=18,
                                 module0_0_conv2d_4_in_channels=288,
                                 module0_0_conv2d_4_out_channels=288,
                                 module0_1_conv2d_0_in_channels=288,
                                 module0_1_conv2d_0_out_channels=288,
                                 module0_1_conv2d_2_in_channels=288,
                                 module0_1_conv2d_2_out_channels=288,
                                 module0_1_conv2d_2_group=18,
                                 module0_1_conv2d_4_in_channels=288,
                                 module0_1_conv2d_4_out_channels=288)
        self.conv2d_82 = nn.Conv2d(in_channels=288,
                                   out_channels=672,
                                   kernel_size=(1, 1),
                                   stride=(2, 2),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.module9_3 = Module9(conv2d_0_in_channels=672,
                                 conv2d_0_out_channels=672,
                                 module1_0_conv2d_0_in_channels=288,
                                 module1_0_conv2d_0_out_channels=672,
                                 module1_0_conv2d_0_kernel_size=(1, 1),
                                 module1_0_conv2d_0_stride=(1, 1),
                                 module1_0_conv2d_0_padding=0,
                                 module1_0_conv2d_0_pad_mode="valid",
                                 module1_0_conv2d_0_group=1,
                                 module1_1_conv2d_0_in_channels=672,
                                 module1_1_conv2d_0_out_channels=672,
                                 module1_1_conv2d_0_kernel_size=(3, 3),
                                 module1_1_conv2d_0_stride=(2, 2),
                                 module1_1_conv2d_0_padding=(1, 1, 1, 1),
                                 module1_1_conv2d_0_pad_mode="pad",
                                 module1_1_conv2d_0_group=42)
        self.relu_89 = nn.ReLU()
        self.module11_1 = Module11(module0_0_conv2d_0_in_channels=672,
                                   module0_0_conv2d_0_out_channels=672,
                                   module0_0_conv2d_2_in_channels=672,
                                   module0_0_conv2d_2_out_channels=672,
                                   module0_0_conv2d_2_group=42,
                                   module0_0_conv2d_4_in_channels=672,
                                   module0_0_conv2d_4_out_channels=672,
                                   module0_1_conv2d_0_in_channels=672,
                                   module0_1_conv2d_0_out_channels=672,
                                   module0_1_conv2d_2_in_channels=672,
                                   module0_1_conv2d_2_out_channels=672,
                                   module0_1_conv2d_2_group=42,
                                   module0_1_conv2d_4_in_channels=672,
                                   module0_1_conv2d_4_out_channels=672,
                                   module0_2_conv2d_0_in_channels=672,
                                   module0_2_conv2d_0_out_channels=672,
                                   module0_2_conv2d_2_in_channels=672,
                                   module0_2_conv2d_2_out_channels=672,
                                   module0_2_conv2d_2_group=42,
                                   module0_2_conv2d_4_in_channels=672,
                                   module0_2_conv2d_4_out_channels=672,
                                   module0_3_conv2d_0_in_channels=672,
                                   module0_3_conv2d_0_out_channels=672,
                                   module0_3_conv2d_2_in_channels=672,
                                   module0_3_conv2d_2_out_channels=672,
                                   module0_3_conv2d_2_group=42,
                                   module0_3_conv2d_4_in_channels=672,
                                   module0_3_conv2d_4_out_channels=672)
        self.avgpool2d_118 = nn.AvgPool2d(kernel_size=(7, 7))
        self.flatten_119 = nn.Flatten()
        self.dense_120 = nn.Dense(in_channels=672, out_channels=1000, has_bias=True)

    def construct(self, input_1):
        opt_conv2d_0 = self.conv2d_0(input_1)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_conv2d_2 = self.conv2d_2(opt_relu_1)
        module9_0_opt = self.module9_0(opt_relu_1)
        opt_add_8 = P.Add()(opt_conv2d_2, module9_0_opt)
        opt_relu_9 = self.relu_9(opt_add_8)
        opt_conv2d_10 = self.conv2d_10(opt_relu_9)
        module9_1_opt = self.module9_1(opt_relu_9)
        opt_add_16 = P.Add()(opt_conv2d_10, module9_1_opt)
        opt_relu_17 = self.relu_17(opt_add_16)
        module5_0_opt = self.module5_0(opt_relu_17)
        opt_conv2d_32 = self.conv2d_32(module5_0_opt)
        module9_2_opt = self.module9_2(module5_0_opt)
        opt_add_38 = P.Add()(opt_conv2d_32, module9_2_opt)
        opt_relu_39 = self.relu_39(opt_add_38)
        module11_0_opt = self.module11_0(opt_relu_39)
        module5_1_opt = self.module5_1(module11_0_opt)
        opt_conv2d_82 = self.conv2d_82(module5_1_opt)
        module9_3_opt = self.module9_3(module5_1_opt)
        opt_add_88 = P.Add()(opt_conv2d_82, module9_3_opt)
        opt_relu_89 = self.relu_89(opt_add_88)
        module11_1_opt = self.module11_1(opt_relu_89)
        opt_avgpool2d_118 = self.avgpool2d_118(module11_1_opt)
        opt_flatten_119 = self.flatten_119(opt_avgpool2d_118)
        opt_dense_120 = self.dense_120(opt_flatten_119)
        return opt_dense_120
