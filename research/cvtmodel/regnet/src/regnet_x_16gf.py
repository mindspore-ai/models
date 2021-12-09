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


class Module3(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_0_kernel_size, conv2d_0_stride,
                 conv2d_0_padding, conv2d_0_pad_mode, conv2d_0_group):
        super(Module3, self).__init__()
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


class Module8(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, module3_0_conv2d_0_in_channels,
                 module3_0_conv2d_0_out_channels, module3_0_conv2d_0_kernel_size, module3_0_conv2d_0_stride,
                 module3_0_conv2d_0_padding, module3_0_conv2d_0_pad_mode, module3_0_conv2d_0_group,
                 module3_1_conv2d_0_in_channels, module3_1_conv2d_0_out_channels, module3_1_conv2d_0_kernel_size,
                 module3_1_conv2d_0_stride, module3_1_conv2d_0_padding, module3_1_conv2d_0_pad_mode,
                 module3_1_conv2d_0_group):
        super(Module8, self).__init__()
        self.module3_0 = Module3(conv2d_0_in_channels=module3_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module3_0_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module3_0_conv2d_0_kernel_size,
                                 conv2d_0_stride=module3_0_conv2d_0_stride,
                                 conv2d_0_padding=module3_0_conv2d_0_padding,
                                 conv2d_0_pad_mode=module3_0_conv2d_0_pad_mode,
                                 conv2d_0_group=module3_0_conv2d_0_group)
        self.module3_1 = Module3(conv2d_0_in_channels=module3_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module3_1_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module3_1_conv2d_0_kernel_size,
                                 conv2d_0_stride=module3_1_conv2d_0_stride,
                                 conv2d_0_padding=module3_1_conv2d_0_padding,
                                 conv2d_0_pad_mode=module3_1_conv2d_0_pad_mode,
                                 conv2d_0_group=module3_1_conv2d_0_group)
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
        module3_0_opt = self.module3_0(x)
        module3_1_opt = self.module3_1(module3_0_opt)
        opt_conv2d_0 = self.conv2d_0(module3_1_opt)
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


class Module10(nn.Cell):
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
        super(Module10, self).__init__()
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
                                  out_channels=256,
                                  kernel_size=(1, 1),
                                  stride=(2, 2),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.module8_0 = Module8(conv2d_0_in_channels=256,
                                 conv2d_0_out_channels=256,
                                 module3_0_conv2d_0_in_channels=32,
                                 module3_0_conv2d_0_out_channels=256,
                                 module3_0_conv2d_0_kernel_size=(1, 1),
                                 module3_0_conv2d_0_stride=(1, 1),
                                 module3_0_conv2d_0_padding=0,
                                 module3_0_conv2d_0_pad_mode="valid",
                                 module3_0_conv2d_0_group=1,
                                 module3_1_conv2d_0_in_channels=256,
                                 module3_1_conv2d_0_out_channels=256,
                                 module3_1_conv2d_0_kernel_size=(3, 3),
                                 module3_1_conv2d_0_stride=(2, 2),
                                 module3_1_conv2d_0_padding=(1, 1, 1, 1),
                                 module3_1_conv2d_0_pad_mode="pad",
                                 module3_1_conv2d_0_group=2)
        self.relu_9 = nn.ReLU()
        self.module0_0 = Module0(conv2d_0_in_channels=256,
                                 conv2d_0_out_channels=256,
                                 conv2d_2_in_channels=256,
                                 conv2d_2_out_channels=256,
                                 conv2d_2_group=2,
                                 conv2d_4_in_channels=256,
                                 conv2d_4_out_channels=256)
        self.conv2d_17 = nn.Conv2d(in_channels=256,
                                   out_channels=512,
                                   kernel_size=(1, 1),
                                   stride=(2, 2),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.module8_1 = Module8(conv2d_0_in_channels=512,
                                 conv2d_0_out_channels=512,
                                 module3_0_conv2d_0_in_channels=256,
                                 module3_0_conv2d_0_out_channels=512,
                                 module3_0_conv2d_0_kernel_size=(1, 1),
                                 module3_0_conv2d_0_stride=(1, 1),
                                 module3_0_conv2d_0_padding=0,
                                 module3_0_conv2d_0_pad_mode="valid",
                                 module3_0_conv2d_0_group=1,
                                 module3_1_conv2d_0_in_channels=512,
                                 module3_1_conv2d_0_out_channels=512,
                                 module3_1_conv2d_0_kernel_size=(3, 3),
                                 module3_1_conv2d_0_stride=(2, 2),
                                 module3_1_conv2d_0_padding=(1, 1, 1, 1),
                                 module3_1_conv2d_0_pad_mode="pad",
                                 module3_1_conv2d_0_group=4)
        self.relu_24 = nn.ReLU()
        self.module10_0 = Module10(module0_0_conv2d_0_in_channels=512,
                                   module0_0_conv2d_0_out_channels=512,
                                   module0_0_conv2d_2_in_channels=512,
                                   module0_0_conv2d_2_out_channels=512,
                                   module0_0_conv2d_2_group=4,
                                   module0_0_conv2d_4_in_channels=512,
                                   module0_0_conv2d_4_out_channels=512,
                                   module0_1_conv2d_0_in_channels=512,
                                   module0_1_conv2d_0_out_channels=512,
                                   module0_1_conv2d_2_in_channels=512,
                                   module0_1_conv2d_2_out_channels=512,
                                   module0_1_conv2d_2_group=4,
                                   module0_1_conv2d_4_in_channels=512,
                                   module0_1_conv2d_4_out_channels=512,
                                   module0_2_conv2d_0_in_channels=512,
                                   module0_2_conv2d_0_out_channels=512,
                                   module0_2_conv2d_2_in_channels=512,
                                   module0_2_conv2d_2_out_channels=512,
                                   module0_2_conv2d_2_group=4,
                                   module0_2_conv2d_4_in_channels=512,
                                   module0_2_conv2d_4_out_channels=512,
                                   module0_3_conv2d_0_in_channels=512,
                                   module0_3_conv2d_0_out_channels=512,
                                   module0_3_conv2d_2_in_channels=512,
                                   module0_3_conv2d_2_out_channels=512,
                                   module0_3_conv2d_2_group=4,
                                   module0_3_conv2d_4_in_channels=512,
                                   module0_3_conv2d_4_out_channels=512)
        self.module0_1 = Module0(conv2d_0_in_channels=512,
                                 conv2d_0_out_channels=512,
                                 conv2d_2_in_channels=512,
                                 conv2d_2_out_channels=512,
                                 conv2d_2_group=4,
                                 conv2d_4_in_channels=512,
                                 conv2d_4_out_channels=512)
        self.conv2d_60 = nn.Conv2d(in_channels=512,
                                   out_channels=896,
                                   kernel_size=(1, 1),
                                   stride=(2, 2),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.module8_2 = Module8(conv2d_0_in_channels=896,
                                 conv2d_0_out_channels=896,
                                 module3_0_conv2d_0_in_channels=512,
                                 module3_0_conv2d_0_out_channels=896,
                                 module3_0_conv2d_0_kernel_size=(1, 1),
                                 module3_0_conv2d_0_stride=(1, 1),
                                 module3_0_conv2d_0_padding=0,
                                 module3_0_conv2d_0_pad_mode="valid",
                                 module3_0_conv2d_0_group=1,
                                 module3_1_conv2d_0_in_channels=896,
                                 module3_1_conv2d_0_out_channels=896,
                                 module3_1_conv2d_0_kernel_size=(3, 3),
                                 module3_1_conv2d_0_stride=(2, 2),
                                 module3_1_conv2d_0_padding=(1, 1, 1, 1),
                                 module3_1_conv2d_0_pad_mode="pad",
                                 module3_1_conv2d_0_group=7)
        self.relu_67 = nn.ReLU()
        self.module10_1 = Module10(module0_0_conv2d_0_in_channels=896,
                                   module0_0_conv2d_0_out_channels=896,
                                   module0_0_conv2d_2_in_channels=896,
                                   module0_0_conv2d_2_out_channels=896,
                                   module0_0_conv2d_2_group=7,
                                   module0_0_conv2d_4_in_channels=896,
                                   module0_0_conv2d_4_out_channels=896,
                                   module0_1_conv2d_0_in_channels=896,
                                   module0_1_conv2d_0_out_channels=896,
                                   module0_1_conv2d_2_in_channels=896,
                                   module0_1_conv2d_2_out_channels=896,
                                   module0_1_conv2d_2_group=7,
                                   module0_1_conv2d_4_in_channels=896,
                                   module0_1_conv2d_4_out_channels=896,
                                   module0_2_conv2d_0_in_channels=896,
                                   module0_2_conv2d_0_out_channels=896,
                                   module0_2_conv2d_2_in_channels=896,
                                   module0_2_conv2d_2_out_channels=896,
                                   module0_2_conv2d_2_group=7,
                                   module0_2_conv2d_4_in_channels=896,
                                   module0_2_conv2d_4_out_channels=896,
                                   module0_3_conv2d_0_in_channels=896,
                                   module0_3_conv2d_0_out_channels=896,
                                   module0_3_conv2d_2_in_channels=896,
                                   module0_3_conv2d_2_out_channels=896,
                                   module0_3_conv2d_2_group=7,
                                   module0_3_conv2d_4_in_channels=896,
                                   module0_3_conv2d_4_out_channels=896)
        self.module10_2 = Module10(module0_0_conv2d_0_in_channels=896,
                                   module0_0_conv2d_0_out_channels=896,
                                   module0_0_conv2d_2_in_channels=896,
                                   module0_0_conv2d_2_out_channels=896,
                                   module0_0_conv2d_2_group=7,
                                   module0_0_conv2d_4_in_channels=896,
                                   module0_0_conv2d_4_out_channels=896,
                                   module0_1_conv2d_0_in_channels=896,
                                   module0_1_conv2d_0_out_channels=896,
                                   module0_1_conv2d_2_in_channels=896,
                                   module0_1_conv2d_2_out_channels=896,
                                   module0_1_conv2d_2_group=7,
                                   module0_1_conv2d_4_in_channels=896,
                                   module0_1_conv2d_4_out_channels=896,
                                   module0_2_conv2d_0_in_channels=896,
                                   module0_2_conv2d_0_out_channels=896,
                                   module0_2_conv2d_2_in_channels=896,
                                   module0_2_conv2d_2_out_channels=896,
                                   module0_2_conv2d_2_group=7,
                                   module0_2_conv2d_4_in_channels=896,
                                   module0_2_conv2d_4_out_channels=896,
                                   module0_3_conv2d_0_in_channels=896,
                                   module0_3_conv2d_0_out_channels=896,
                                   module0_3_conv2d_2_in_channels=896,
                                   module0_3_conv2d_2_out_channels=896,
                                   module0_3_conv2d_2_group=7,
                                   module0_3_conv2d_4_in_channels=896,
                                   module0_3_conv2d_4_out_channels=896)
        self.module10_3 = Module10(module0_0_conv2d_0_in_channels=896,
                                   module0_0_conv2d_0_out_channels=896,
                                   module0_0_conv2d_2_in_channels=896,
                                   module0_0_conv2d_2_out_channels=896,
                                   module0_0_conv2d_2_group=7,
                                   module0_0_conv2d_4_in_channels=896,
                                   module0_0_conv2d_4_out_channels=896,
                                   module0_1_conv2d_0_in_channels=896,
                                   module0_1_conv2d_0_out_channels=896,
                                   module0_1_conv2d_2_in_channels=896,
                                   module0_1_conv2d_2_out_channels=896,
                                   module0_1_conv2d_2_group=7,
                                   module0_1_conv2d_4_in_channels=896,
                                   module0_1_conv2d_4_out_channels=896,
                                   module0_2_conv2d_0_in_channels=896,
                                   module0_2_conv2d_0_out_channels=896,
                                   module0_2_conv2d_2_in_channels=896,
                                   module0_2_conv2d_2_out_channels=896,
                                   module0_2_conv2d_2_group=7,
                                   module0_2_conv2d_4_in_channels=896,
                                   module0_2_conv2d_4_out_channels=896,
                                   module0_3_conv2d_0_in_channels=896,
                                   module0_3_conv2d_0_out_channels=896,
                                   module0_3_conv2d_2_in_channels=896,
                                   module0_3_conv2d_2_out_channels=896,
                                   module0_3_conv2d_2_group=7,
                                   module0_3_conv2d_4_in_channels=896,
                                   module0_3_conv2d_4_out_channels=896)
        self.conv2d_152 = nn.Conv2d(in_channels=896,
                                    out_channels=2048,
                                    kernel_size=(1, 1),
                                    stride=(2, 2),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module8_3 = Module8(conv2d_0_in_channels=2048,
                                 conv2d_0_out_channels=2048,
                                 module3_0_conv2d_0_in_channels=896,
                                 module3_0_conv2d_0_out_channels=2048,
                                 module3_0_conv2d_0_kernel_size=(1, 1),
                                 module3_0_conv2d_0_stride=(1, 1),
                                 module3_0_conv2d_0_padding=0,
                                 module3_0_conv2d_0_pad_mode="valid",
                                 module3_0_conv2d_0_group=1,
                                 module3_1_conv2d_0_in_channels=2048,
                                 module3_1_conv2d_0_out_channels=2048,
                                 module3_1_conv2d_0_kernel_size=(3, 3),
                                 module3_1_conv2d_0_stride=(2, 2),
                                 module3_1_conv2d_0_padding=(1, 1, 1, 1),
                                 module3_1_conv2d_0_pad_mode="pad",
                                 module3_1_conv2d_0_group=16)
        self.relu_159 = nn.ReLU()
        self.avgpool2d_160 = nn.AvgPool2d(kernel_size=(7, 7))
        self.flatten_161 = nn.Flatten()
        self.dense_162 = nn.Dense(in_channels=2048, out_channels=1000, has_bias=True)

    def construct(self, input_1):
        opt_conv2d_0 = self.conv2d_0(input_1)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_conv2d_2 = self.conv2d_2(opt_relu_1)
        module8_0_opt = self.module8_0(opt_relu_1)
        opt_add_8 = P.Add()(opt_conv2d_2, module8_0_opt)
        opt_relu_9 = self.relu_9(opt_add_8)
        module0_0_opt = self.module0_0(opt_relu_9)
        opt_conv2d_17 = self.conv2d_17(module0_0_opt)
        module8_1_opt = self.module8_1(module0_0_opt)
        opt_add_23 = P.Add()(opt_conv2d_17, module8_1_opt)
        opt_relu_24 = self.relu_24(opt_add_23)
        module10_0_opt = self.module10_0(opt_relu_24)
        module0_1_opt = self.module0_1(module10_0_opt)
        opt_conv2d_60 = self.conv2d_60(module0_1_opt)
        module8_2_opt = self.module8_2(module0_1_opt)
        opt_add_66 = P.Add()(opt_conv2d_60, module8_2_opt)
        opt_relu_67 = self.relu_67(opt_add_66)
        module10_1_opt = self.module10_1(opt_relu_67)
        module10_2_opt = self.module10_2(module10_1_opt)
        module10_3_opt = self.module10_3(module10_2_opt)
        opt_conv2d_152 = self.conv2d_152(module10_3_opt)
        module8_3_opt = self.module8_3(module10_3_opt)
        opt_add_158 = P.Add()(opt_conv2d_152, module8_3_opt)
        opt_relu_159 = self.relu_159(opt_add_158)
        opt_avgpool2d_160 = self.avgpool2d_160(opt_relu_159)
        opt_flatten_161 = self.flatten_161(opt_avgpool2d_160)
        opt_dense_162 = self.dense_162(opt_flatten_161)
        return opt_dense_162
