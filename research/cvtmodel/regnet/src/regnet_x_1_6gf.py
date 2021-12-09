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


class Module6(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, module1_0_conv2d_0_in_channels,
                 module1_0_conv2d_0_out_channels, module1_0_conv2d_0_kernel_size, module1_0_conv2d_0_stride,
                 module1_0_conv2d_0_padding, module1_0_conv2d_0_pad_mode, module1_0_conv2d_0_group,
                 module1_1_conv2d_0_in_channels, module1_1_conv2d_0_out_channels, module1_1_conv2d_0_kernel_size,
                 module1_1_conv2d_0_stride, module1_1_conv2d_0_padding, module1_1_conv2d_0_pad_mode,
                 module1_1_conv2d_0_group):
        super(Module6, self).__init__()
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


class Module11(nn.Cell):
    def __init__(self, module0_0_conv2d_0_in_channels, module0_0_conv2d_0_out_channels, module0_0_conv2d_2_in_channels,
                 module0_0_conv2d_2_out_channels, module0_0_conv2d_2_group, module0_0_conv2d_4_in_channels,
                 module0_0_conv2d_4_out_channels, module0_1_conv2d_0_in_channels, module0_1_conv2d_0_out_channels,
                 module0_1_conv2d_2_in_channels, module0_1_conv2d_2_out_channels, module0_1_conv2d_2_group,
                 module0_1_conv2d_4_in_channels, module0_1_conv2d_4_out_channels, module0_2_conv2d_0_in_channels,
                 module0_2_conv2d_0_out_channels, module0_2_conv2d_2_in_channels, module0_2_conv2d_2_out_channels,
                 module0_2_conv2d_2_group, module0_2_conv2d_4_in_channels, module0_2_conv2d_4_out_channels):
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

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        module0_1_opt = self.module0_1(module0_0_opt)
        module0_2_opt = self.module0_2(module0_1_opt)
        return module0_2_opt


class Module8(nn.Cell):
    def __init__(self, module0_0_conv2d_0_in_channels, module0_0_conv2d_0_out_channels, module0_0_conv2d_2_in_channels,
                 module0_0_conv2d_2_out_channels, module0_0_conv2d_2_group, module0_0_conv2d_4_in_channels,
                 module0_0_conv2d_4_out_channels, module0_1_conv2d_0_in_channels, module0_1_conv2d_0_out_channels,
                 module0_1_conv2d_2_in_channels, module0_1_conv2d_2_out_channels, module0_1_conv2d_2_group,
                 module0_1_conv2d_4_in_channels, module0_1_conv2d_4_out_channels):
        super(Module8, self).__init__()
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
                                  out_channels=72,
                                  kernel_size=(1, 1),
                                  stride=(2, 2),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.module6_0 = Module6(conv2d_0_in_channels=72,
                                 conv2d_0_out_channels=72,
                                 module1_0_conv2d_0_in_channels=32,
                                 module1_0_conv2d_0_out_channels=72,
                                 module1_0_conv2d_0_kernel_size=(1, 1),
                                 module1_0_conv2d_0_stride=(1, 1),
                                 module1_0_conv2d_0_padding=0,
                                 module1_0_conv2d_0_pad_mode="valid",
                                 module1_0_conv2d_0_group=1,
                                 module1_1_conv2d_0_in_channels=72,
                                 module1_1_conv2d_0_out_channels=72,
                                 module1_1_conv2d_0_kernel_size=(3, 3),
                                 module1_1_conv2d_0_stride=(2, 2),
                                 module1_1_conv2d_0_padding=(1, 1, 1, 1),
                                 module1_1_conv2d_0_pad_mode="pad",
                                 module1_1_conv2d_0_group=3)
        self.relu_9 = nn.ReLU()
        self.module0_0 = Module0(conv2d_0_in_channels=72,
                                 conv2d_0_out_channels=72,
                                 conv2d_2_in_channels=72,
                                 conv2d_2_out_channels=72,
                                 conv2d_2_group=3,
                                 conv2d_4_in_channels=72,
                                 conv2d_4_out_channels=72)
        self.conv2d_17 = nn.Conv2d(in_channels=72,
                                   out_channels=168,
                                   kernel_size=(1, 1),
                                   stride=(2, 2),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.module6_1 = Module6(conv2d_0_in_channels=168,
                                 conv2d_0_out_channels=168,
                                 module1_0_conv2d_0_in_channels=72,
                                 module1_0_conv2d_0_out_channels=168,
                                 module1_0_conv2d_0_kernel_size=(1, 1),
                                 module1_0_conv2d_0_stride=(1, 1),
                                 module1_0_conv2d_0_padding=0,
                                 module1_0_conv2d_0_pad_mode="valid",
                                 module1_0_conv2d_0_group=1,
                                 module1_1_conv2d_0_in_channels=168,
                                 module1_1_conv2d_0_out_channels=168,
                                 module1_1_conv2d_0_kernel_size=(3, 3),
                                 module1_1_conv2d_0_stride=(2, 2),
                                 module1_1_conv2d_0_padding=(1, 1, 1, 1),
                                 module1_1_conv2d_0_pad_mode="pad",
                                 module1_1_conv2d_0_group=7)
        self.relu_24 = nn.ReLU()
        self.module11_0 = Module11(module0_0_conv2d_0_in_channels=168,
                                   module0_0_conv2d_0_out_channels=168,
                                   module0_0_conv2d_2_in_channels=168,
                                   module0_0_conv2d_2_out_channels=168,
                                   module0_0_conv2d_2_group=7,
                                   module0_0_conv2d_4_in_channels=168,
                                   module0_0_conv2d_4_out_channels=168,
                                   module0_1_conv2d_0_in_channels=168,
                                   module0_1_conv2d_0_out_channels=168,
                                   module0_1_conv2d_2_in_channels=168,
                                   module0_1_conv2d_2_out_channels=168,
                                   module0_1_conv2d_2_group=7,
                                   module0_1_conv2d_4_in_channels=168,
                                   module0_1_conv2d_4_out_channels=168,
                                   module0_2_conv2d_0_in_channels=168,
                                   module0_2_conv2d_0_out_channels=168,
                                   module0_2_conv2d_2_in_channels=168,
                                   module0_2_conv2d_2_out_channels=168,
                                   module0_2_conv2d_2_group=7,
                                   module0_2_conv2d_4_in_channels=168,
                                   module0_2_conv2d_4_out_channels=168)
        self.conv2d_46 = nn.Conv2d(in_channels=168,
                                   out_channels=408,
                                   kernel_size=(1, 1),
                                   stride=(2, 2),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.module6_2 = Module6(conv2d_0_in_channels=408,
                                 conv2d_0_out_channels=408,
                                 module1_0_conv2d_0_in_channels=168,
                                 module1_0_conv2d_0_out_channels=408,
                                 module1_0_conv2d_0_kernel_size=(1, 1),
                                 module1_0_conv2d_0_stride=(1, 1),
                                 module1_0_conv2d_0_padding=0,
                                 module1_0_conv2d_0_pad_mode="valid",
                                 module1_0_conv2d_0_group=1,
                                 module1_1_conv2d_0_in_channels=408,
                                 module1_1_conv2d_0_out_channels=408,
                                 module1_1_conv2d_0_kernel_size=(3, 3),
                                 module1_1_conv2d_0_stride=(2, 2),
                                 module1_1_conv2d_0_padding=(1, 1, 1, 1),
                                 module1_1_conv2d_0_pad_mode="pad",
                                 module1_1_conv2d_0_group=17)
        self.relu_53 = nn.ReLU()
        self.module8_0 = Module8(module0_0_conv2d_0_in_channels=408,
                                 module0_0_conv2d_0_out_channels=408,
                                 module0_0_conv2d_2_in_channels=408,
                                 module0_0_conv2d_2_out_channels=408,
                                 module0_0_conv2d_2_group=17,
                                 module0_0_conv2d_4_in_channels=408,
                                 module0_0_conv2d_4_out_channels=408,
                                 module0_1_conv2d_0_in_channels=408,
                                 module0_1_conv2d_0_out_channels=408,
                                 module0_1_conv2d_2_in_channels=408,
                                 module0_1_conv2d_2_out_channels=408,
                                 module0_1_conv2d_2_group=17,
                                 module0_1_conv2d_4_in_channels=408,
                                 module0_1_conv2d_4_out_channels=408)
        self.module8_1 = Module8(module0_0_conv2d_0_in_channels=408,
                                 module0_0_conv2d_0_out_channels=408,
                                 module0_0_conv2d_2_in_channels=408,
                                 module0_0_conv2d_2_out_channels=408,
                                 module0_0_conv2d_2_group=17,
                                 module0_0_conv2d_4_in_channels=408,
                                 module0_0_conv2d_4_out_channels=408,
                                 module0_1_conv2d_0_in_channels=408,
                                 module0_1_conv2d_0_out_channels=408,
                                 module0_1_conv2d_2_in_channels=408,
                                 module0_1_conv2d_2_out_channels=408,
                                 module0_1_conv2d_2_group=17,
                                 module0_1_conv2d_4_in_channels=408,
                                 module0_1_conv2d_4_out_channels=408)
        self.module8_2 = Module8(module0_0_conv2d_0_in_channels=408,
                                 module0_0_conv2d_0_out_channels=408,
                                 module0_0_conv2d_2_in_channels=408,
                                 module0_0_conv2d_2_out_channels=408,
                                 module0_0_conv2d_2_group=17,
                                 module0_0_conv2d_4_in_channels=408,
                                 module0_0_conv2d_4_out_channels=408,
                                 module0_1_conv2d_0_in_channels=408,
                                 module0_1_conv2d_0_out_channels=408,
                                 module0_1_conv2d_2_in_channels=408,
                                 module0_1_conv2d_2_out_channels=408,
                                 module0_1_conv2d_2_group=17,
                                 module0_1_conv2d_4_in_channels=408,
                                 module0_1_conv2d_4_out_channels=408)
        self.module11_1 = Module11(module0_0_conv2d_0_in_channels=408,
                                   module0_0_conv2d_0_out_channels=408,
                                   module0_0_conv2d_2_in_channels=408,
                                   module0_0_conv2d_2_out_channels=408,
                                   module0_0_conv2d_2_group=17,
                                   module0_0_conv2d_4_in_channels=408,
                                   module0_0_conv2d_4_out_channels=408,
                                   module0_1_conv2d_0_in_channels=408,
                                   module0_1_conv2d_0_out_channels=408,
                                   module0_1_conv2d_2_in_channels=408,
                                   module0_1_conv2d_2_out_channels=408,
                                   module0_1_conv2d_2_group=17,
                                   module0_1_conv2d_4_in_channels=408,
                                   module0_1_conv2d_4_out_channels=408,
                                   module0_2_conv2d_0_in_channels=408,
                                   module0_2_conv2d_0_out_channels=408,
                                   module0_2_conv2d_2_in_channels=408,
                                   module0_2_conv2d_2_out_channels=408,
                                   module0_2_conv2d_2_group=17,
                                   module0_2_conv2d_4_in_channels=408,
                                   module0_2_conv2d_4_out_channels=408)
        self.conv2d_117 = nn.Conv2d(in_channels=408,
                                    out_channels=912,
                                    kernel_size=(1, 1),
                                    stride=(2, 2),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module6_3 = Module6(conv2d_0_in_channels=912,
                                 conv2d_0_out_channels=912,
                                 module1_0_conv2d_0_in_channels=408,
                                 module1_0_conv2d_0_out_channels=912,
                                 module1_0_conv2d_0_kernel_size=(1, 1),
                                 module1_0_conv2d_0_stride=(1, 1),
                                 module1_0_conv2d_0_padding=0,
                                 module1_0_conv2d_0_pad_mode="valid",
                                 module1_0_conv2d_0_group=1,
                                 module1_1_conv2d_0_in_channels=912,
                                 module1_1_conv2d_0_out_channels=912,
                                 module1_1_conv2d_0_kernel_size=(3, 3),
                                 module1_1_conv2d_0_stride=(2, 2),
                                 module1_1_conv2d_0_padding=(1, 1, 1, 1),
                                 module1_1_conv2d_0_pad_mode="pad",
                                 module1_1_conv2d_0_group=38)
        self.relu_124 = nn.ReLU()
        self.module0_1 = Module0(conv2d_0_in_channels=912,
                                 conv2d_0_out_channels=912,
                                 conv2d_2_in_channels=912,
                                 conv2d_2_out_channels=912,
                                 conv2d_2_group=38,
                                 conv2d_4_in_channels=912,
                                 conv2d_4_out_channels=912)
        self.avgpool2d_132 = nn.AvgPool2d(kernel_size=(7, 7))
        self.flatten_133 = nn.Flatten()
        self.dense_134 = nn.Dense(in_channels=912, out_channels=1000, has_bias=True)

    def construct(self, input_1):
        opt_conv2d_0 = self.conv2d_0(input_1)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_conv2d_2 = self.conv2d_2(opt_relu_1)
        module6_0_opt = self.module6_0(opt_relu_1)
        opt_add_8 = P.Add()(opt_conv2d_2, module6_0_opt)
        opt_relu_9 = self.relu_9(opt_add_8)
        module0_0_opt = self.module0_0(opt_relu_9)
        opt_conv2d_17 = self.conv2d_17(module0_0_opt)
        module6_1_opt = self.module6_1(module0_0_opt)
        opt_add_23 = P.Add()(opt_conv2d_17, module6_1_opt)
        opt_relu_24 = self.relu_24(opt_add_23)
        module11_0_opt = self.module11_0(opt_relu_24)
        opt_conv2d_46 = self.conv2d_46(module11_0_opt)
        module6_2_opt = self.module6_2(module11_0_opt)
        opt_add_52 = P.Add()(opt_conv2d_46, module6_2_opt)
        opt_relu_53 = self.relu_53(opt_add_52)
        module8_0_opt = self.module8_0(opt_relu_53)
        module8_1_opt = self.module8_1(module8_0_opt)
        module8_2_opt = self.module8_2(module8_1_opt)
        module11_1_opt = self.module11_1(module8_2_opt)
        opt_conv2d_117 = self.conv2d_117(module11_1_opt)
        module6_3_opt = self.module6_3(module11_1_opt)
        opt_add_123 = P.Add()(opt_conv2d_117, module6_3_opt)
        opt_relu_124 = self.relu_124(opt_add_123)
        module0_1_opt = self.module0_1(opt_relu_124)
        opt_avgpool2d_132 = self.avgpool2d_132(module0_1_opt)
        opt_flatten_133 = self.flatten_133(opt_avgpool2d_132)
        opt_dense_134 = self.dense_134(opt_flatten_133)
        return opt_dense_134
