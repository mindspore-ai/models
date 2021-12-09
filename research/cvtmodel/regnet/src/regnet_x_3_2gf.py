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


class Module2(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_0_kernel_size, conv2d_0_stride,
                 conv2d_0_padding, conv2d_0_pad_mode, conv2d_0_group):
        super(Module2, self).__init__()
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


class Module7(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, module2_0_conv2d_0_in_channels,
                 module2_0_conv2d_0_out_channels, module2_0_conv2d_0_kernel_size, module2_0_conv2d_0_stride,
                 module2_0_conv2d_0_padding, module2_0_conv2d_0_pad_mode, module2_0_conv2d_0_group,
                 module2_1_conv2d_0_in_channels, module2_1_conv2d_0_out_channels, module2_1_conv2d_0_kernel_size,
                 module2_1_conv2d_0_stride, module2_1_conv2d_0_padding, module2_1_conv2d_0_pad_mode,
                 module2_1_conv2d_0_group):
        super(Module7, self).__init__()
        self.module2_0 = Module2(conv2d_0_in_channels=module2_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module2_0_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module2_0_conv2d_0_kernel_size,
                                 conv2d_0_stride=module2_0_conv2d_0_stride,
                                 conv2d_0_padding=module2_0_conv2d_0_padding,
                                 conv2d_0_pad_mode=module2_0_conv2d_0_pad_mode,
                                 conv2d_0_group=module2_0_conv2d_0_group)
        self.module2_1 = Module2(conv2d_0_in_channels=module2_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module2_1_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module2_1_conv2d_0_kernel_size,
                                 conv2d_0_stride=module2_1_conv2d_0_stride,
                                 conv2d_0_padding=module2_1_conv2d_0_padding,
                                 conv2d_0_pad_mode=module2_1_conv2d_0_pad_mode,
                                 conv2d_0_group=module2_1_conv2d_0_group)
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
        module2_0_opt = self.module2_0(x)
        module2_1_opt = self.module2_1(module2_0_opt)
        opt_conv2d_0 = self.conv2d_0(module2_1_opt)
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


class Module9(nn.Cell):
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
        super(Module9, self).__init__()
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


class Module8(nn.Cell):
    def __init__(self):
        super(Module8, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=432,
                                 conv2d_0_out_channels=432,
                                 conv2d_2_in_channels=432,
                                 conv2d_2_out_channels=432,
                                 conv2d_2_group=9,
                                 conv2d_4_in_channels=432,
                                 conv2d_4_out_channels=432)
        self.module0_1 = Module0(conv2d_0_in_channels=432,
                                 conv2d_0_out_channels=432,
                                 conv2d_2_in_channels=432,
                                 conv2d_2_out_channels=432,
                                 conv2d_2_group=9,
                                 conv2d_4_in_channels=432,
                                 conv2d_4_out_channels=432)

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        module0_1_opt = self.module0_1(module0_0_opt)
        return module0_1_opt


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
                                  out_channels=96,
                                  kernel_size=(1, 1),
                                  stride=(2, 2),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.module7_0 = Module7(conv2d_0_in_channels=96,
                                 conv2d_0_out_channels=96,
                                 module2_0_conv2d_0_in_channels=32,
                                 module2_0_conv2d_0_out_channels=96,
                                 module2_0_conv2d_0_kernel_size=(1, 1),
                                 module2_0_conv2d_0_stride=(1, 1),
                                 module2_0_conv2d_0_padding=0,
                                 module2_0_conv2d_0_pad_mode="valid",
                                 module2_0_conv2d_0_group=1,
                                 module2_1_conv2d_0_in_channels=96,
                                 module2_1_conv2d_0_out_channels=96,
                                 module2_1_conv2d_0_kernel_size=(3, 3),
                                 module2_1_conv2d_0_stride=(2, 2),
                                 module2_1_conv2d_0_padding=(1, 1, 1, 1),
                                 module2_1_conv2d_0_pad_mode="pad",
                                 module2_1_conv2d_0_group=2)
        self.relu_9 = nn.ReLU()
        self.module0_0 = Module0(conv2d_0_in_channels=96,
                                 conv2d_0_out_channels=96,
                                 conv2d_2_in_channels=96,
                                 conv2d_2_out_channels=96,
                                 conv2d_2_group=2,
                                 conv2d_4_in_channels=96,
                                 conv2d_4_out_channels=96)
        self.conv2d_17 = nn.Conv2d(in_channels=96,
                                   out_channels=192,
                                   kernel_size=(1, 1),
                                   stride=(2, 2),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.module7_1 = Module7(conv2d_0_in_channels=192,
                                 conv2d_0_out_channels=192,
                                 module2_0_conv2d_0_in_channels=96,
                                 module2_0_conv2d_0_out_channels=192,
                                 module2_0_conv2d_0_kernel_size=(1, 1),
                                 module2_0_conv2d_0_stride=(1, 1),
                                 module2_0_conv2d_0_padding=0,
                                 module2_0_conv2d_0_pad_mode="valid",
                                 module2_0_conv2d_0_group=1,
                                 module2_1_conv2d_0_in_channels=192,
                                 module2_1_conv2d_0_out_channels=192,
                                 module2_1_conv2d_0_kernel_size=(3, 3),
                                 module2_1_conv2d_0_stride=(2, 2),
                                 module2_1_conv2d_0_padding=(1, 1, 1, 1),
                                 module2_1_conv2d_0_pad_mode="pad",
                                 module2_1_conv2d_0_group=4)
        self.relu_24 = nn.ReLU()
        self.module9_0 = Module9(module0_0_conv2d_0_in_channels=192,
                                 module0_0_conv2d_0_out_channels=192,
                                 module0_0_conv2d_2_in_channels=192,
                                 module0_0_conv2d_2_out_channels=192,
                                 module0_0_conv2d_2_group=4,
                                 module0_0_conv2d_4_in_channels=192,
                                 module0_0_conv2d_4_out_channels=192,
                                 module0_1_conv2d_0_in_channels=192,
                                 module0_1_conv2d_0_out_channels=192,
                                 module0_1_conv2d_2_in_channels=192,
                                 module0_1_conv2d_2_out_channels=192,
                                 module0_1_conv2d_2_group=4,
                                 module0_1_conv2d_4_in_channels=192,
                                 module0_1_conv2d_4_out_channels=192,
                                 module0_2_conv2d_0_in_channels=192,
                                 module0_2_conv2d_0_out_channels=192,
                                 module0_2_conv2d_2_in_channels=192,
                                 module0_2_conv2d_2_out_channels=192,
                                 module0_2_conv2d_2_group=4,
                                 module0_2_conv2d_4_in_channels=192,
                                 module0_2_conv2d_4_out_channels=192,
                                 module0_3_conv2d_0_in_channels=192,
                                 module0_3_conv2d_0_out_channels=192,
                                 module0_3_conv2d_2_in_channels=192,
                                 module0_3_conv2d_2_out_channels=192,
                                 module0_3_conv2d_2_group=4,
                                 module0_3_conv2d_4_in_channels=192,
                                 module0_3_conv2d_4_out_channels=192)
        self.module0_1 = Module0(conv2d_0_in_channels=192,
                                 conv2d_0_out_channels=192,
                                 conv2d_2_in_channels=192,
                                 conv2d_2_out_channels=192,
                                 conv2d_2_group=4,
                                 conv2d_4_in_channels=192,
                                 conv2d_4_out_channels=192)
        self.conv2d_60 = nn.Conv2d(in_channels=192,
                                   out_channels=432,
                                   kernel_size=(1, 1),
                                   stride=(2, 2),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.module7_2 = Module7(conv2d_0_in_channels=432,
                                 conv2d_0_out_channels=432,
                                 module2_0_conv2d_0_in_channels=192,
                                 module2_0_conv2d_0_out_channels=432,
                                 module2_0_conv2d_0_kernel_size=(1, 1),
                                 module2_0_conv2d_0_stride=(1, 1),
                                 module2_0_conv2d_0_padding=0,
                                 module2_0_conv2d_0_pad_mode="valid",
                                 module2_0_conv2d_0_group=1,
                                 module2_1_conv2d_0_in_channels=432,
                                 module2_1_conv2d_0_out_channels=432,
                                 module2_1_conv2d_0_kernel_size=(3, 3),
                                 module2_1_conv2d_0_stride=(2, 2),
                                 module2_1_conv2d_0_padding=(1, 1, 1, 1),
                                 module2_1_conv2d_0_pad_mode="pad",
                                 module2_1_conv2d_0_group=9)
        self.relu_67 = nn.ReLU()
        self.module9_1 = Module9(module0_0_conv2d_0_in_channels=432,
                                 module0_0_conv2d_0_out_channels=432,
                                 module0_0_conv2d_2_in_channels=432,
                                 module0_0_conv2d_2_out_channels=432,
                                 module0_0_conv2d_2_group=9,
                                 module0_0_conv2d_4_in_channels=432,
                                 module0_0_conv2d_4_out_channels=432,
                                 module0_1_conv2d_0_in_channels=432,
                                 module0_1_conv2d_0_out_channels=432,
                                 module0_1_conv2d_2_in_channels=432,
                                 module0_1_conv2d_2_out_channels=432,
                                 module0_1_conv2d_2_group=9,
                                 module0_1_conv2d_4_in_channels=432,
                                 module0_1_conv2d_4_out_channels=432,
                                 module0_2_conv2d_0_in_channels=432,
                                 module0_2_conv2d_0_out_channels=432,
                                 module0_2_conv2d_2_in_channels=432,
                                 module0_2_conv2d_2_out_channels=432,
                                 module0_2_conv2d_2_group=9,
                                 module0_2_conv2d_4_in_channels=432,
                                 module0_2_conv2d_4_out_channels=432,
                                 module0_3_conv2d_0_in_channels=432,
                                 module0_3_conv2d_0_out_channels=432,
                                 module0_3_conv2d_2_in_channels=432,
                                 module0_3_conv2d_2_out_channels=432,
                                 module0_3_conv2d_2_group=9,
                                 module0_3_conv2d_4_in_channels=432,
                                 module0_3_conv2d_4_out_channels=432)
        self.module9_2 = Module9(module0_0_conv2d_0_in_channels=432,
                                 module0_0_conv2d_0_out_channels=432,
                                 module0_0_conv2d_2_in_channels=432,
                                 module0_0_conv2d_2_out_channels=432,
                                 module0_0_conv2d_2_group=9,
                                 module0_0_conv2d_4_in_channels=432,
                                 module0_0_conv2d_4_out_channels=432,
                                 module0_1_conv2d_0_in_channels=432,
                                 module0_1_conv2d_0_out_channels=432,
                                 module0_1_conv2d_2_in_channels=432,
                                 module0_1_conv2d_2_out_channels=432,
                                 module0_1_conv2d_2_group=9,
                                 module0_1_conv2d_4_in_channels=432,
                                 module0_1_conv2d_4_out_channels=432,
                                 module0_2_conv2d_0_in_channels=432,
                                 module0_2_conv2d_0_out_channels=432,
                                 module0_2_conv2d_2_in_channels=432,
                                 module0_2_conv2d_2_out_channels=432,
                                 module0_2_conv2d_2_group=9,
                                 module0_2_conv2d_4_in_channels=432,
                                 module0_2_conv2d_4_out_channels=432,
                                 module0_3_conv2d_0_in_channels=432,
                                 module0_3_conv2d_0_out_channels=432,
                                 module0_3_conv2d_2_in_channels=432,
                                 module0_3_conv2d_2_out_channels=432,
                                 module0_3_conv2d_2_group=9,
                                 module0_3_conv2d_4_in_channels=432,
                                 module0_3_conv2d_4_out_channels=432)
        self.module9_3 = Module9(module0_0_conv2d_0_in_channels=432,
                                 module0_0_conv2d_0_out_channels=432,
                                 module0_0_conv2d_2_in_channels=432,
                                 module0_0_conv2d_2_out_channels=432,
                                 module0_0_conv2d_2_group=9,
                                 module0_0_conv2d_4_in_channels=432,
                                 module0_0_conv2d_4_out_channels=432,
                                 module0_1_conv2d_0_in_channels=432,
                                 module0_1_conv2d_0_out_channels=432,
                                 module0_1_conv2d_2_in_channels=432,
                                 module0_1_conv2d_2_out_channels=432,
                                 module0_1_conv2d_2_group=9,
                                 module0_1_conv2d_4_in_channels=432,
                                 module0_1_conv2d_4_out_channels=432,
                                 module0_2_conv2d_0_in_channels=432,
                                 module0_2_conv2d_0_out_channels=432,
                                 module0_2_conv2d_2_in_channels=432,
                                 module0_2_conv2d_2_out_channels=432,
                                 module0_2_conv2d_2_group=9,
                                 module0_2_conv2d_4_in_channels=432,
                                 module0_2_conv2d_4_out_channels=432,
                                 module0_3_conv2d_0_in_channels=432,
                                 module0_3_conv2d_0_out_channels=432,
                                 module0_3_conv2d_2_in_channels=432,
                                 module0_3_conv2d_2_out_channels=432,
                                 module0_3_conv2d_2_group=9,
                                 module0_3_conv2d_4_in_channels=432,
                                 module0_3_conv2d_4_out_channels=432)
        self.module8_0 = Module8()
        self.conv2d_166 = nn.Conv2d(in_channels=432,
                                    out_channels=1008,
                                    kernel_size=(1, 1),
                                    stride=(2, 2),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module7_3 = Module7(conv2d_0_in_channels=1008,
                                 conv2d_0_out_channels=1008,
                                 module2_0_conv2d_0_in_channels=432,
                                 module2_0_conv2d_0_out_channels=1008,
                                 module2_0_conv2d_0_kernel_size=(1, 1),
                                 module2_0_conv2d_0_stride=(1, 1),
                                 module2_0_conv2d_0_padding=0,
                                 module2_0_conv2d_0_pad_mode="valid",
                                 module2_0_conv2d_0_group=1,
                                 module2_1_conv2d_0_in_channels=1008,
                                 module2_1_conv2d_0_out_channels=1008,
                                 module2_1_conv2d_0_kernel_size=(3, 3),
                                 module2_1_conv2d_0_stride=(2, 2),
                                 module2_1_conv2d_0_padding=(1, 1, 1, 1),
                                 module2_1_conv2d_0_pad_mode="pad",
                                 module2_1_conv2d_0_group=21)
        self.relu_173 = nn.ReLU()
        self.module0_2 = Module0(conv2d_0_in_channels=1008,
                                 conv2d_0_out_channels=1008,
                                 conv2d_2_in_channels=1008,
                                 conv2d_2_out_channels=1008,
                                 conv2d_2_group=21,
                                 conv2d_4_in_channels=1008,
                                 conv2d_4_out_channels=1008)
        self.avgpool2d_181 = nn.AvgPool2d(kernel_size=(7, 7))
        self.flatten_182 = nn.Flatten()
        self.dense_183 = nn.Dense(in_channels=1008, out_channels=1000, has_bias=True)

    def construct(self, input_1):
        opt_conv2d_0 = self.conv2d_0(input_1)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_conv2d_2 = self.conv2d_2(opt_relu_1)
        module7_0_opt = self.module7_0(opt_relu_1)
        opt_add_8 = P.Add()(opt_conv2d_2, module7_0_opt)
        opt_relu_9 = self.relu_9(opt_add_8)
        module0_0_opt = self.module0_0(opt_relu_9)
        opt_conv2d_17 = self.conv2d_17(module0_0_opt)
        module7_1_opt = self.module7_1(module0_0_opt)
        opt_add_23 = P.Add()(opt_conv2d_17, module7_1_opt)
        opt_relu_24 = self.relu_24(opt_add_23)
        module9_0_opt = self.module9_0(opt_relu_24)
        module0_1_opt = self.module0_1(module9_0_opt)
        opt_conv2d_60 = self.conv2d_60(module0_1_opt)
        module7_2_opt = self.module7_2(module0_1_opt)
        opt_add_66 = P.Add()(opt_conv2d_60, module7_2_opt)
        opt_relu_67 = self.relu_67(opt_add_66)
        module9_1_opt = self.module9_1(opt_relu_67)
        module9_2_opt = self.module9_2(module9_1_opt)
        module9_3_opt = self.module9_3(module9_2_opt)
        module8_0_opt = self.module8_0(module9_3_opt)
        opt_conv2d_166 = self.conv2d_166(module8_0_opt)
        module7_3_opt = self.module7_3(module8_0_opt)
        opt_add_172 = P.Add()(opt_conv2d_166, module7_3_opt)
        opt_relu_173 = self.relu_173(opt_add_172)
        module0_2_opt = self.module0_2(opt_relu_173)
        opt_avgpool2d_181 = self.avgpool2d_181(module0_2_opt)
        opt_flatten_182 = self.flatten_182(opt_avgpool2d_181)
        opt_dense_183 = self.dense_183(opt_flatten_182)
        return opt_dense_183
