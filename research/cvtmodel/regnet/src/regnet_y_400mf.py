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


class Module0(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_2_in_channels, conv2d_2_out_channels,
                 conv2d_2_stride, conv2d_2_group, avgpool2d_4_kernel_size, conv2d_5_in_channels, conv2d_5_out_channels,
                 conv2d_7_in_channels, conv2d_7_out_channels):
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
                                  stride=conv2d_2_stride,
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=conv2d_2_group,
                                  has_bias=True)
        self.relu_3 = nn.ReLU()
        self.avgpool2d_4 = nn.AvgPool2d(kernel_size=avgpool2d_4_kernel_size)
        self.conv2d_5 = nn.Conv2d(in_channels=conv2d_5_in_channels,
                                  out_channels=conv2d_5_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_6 = nn.ReLU()
        self.conv2d_7 = nn.Conv2d(in_channels=conv2d_7_in_channels,
                                  out_channels=conv2d_7_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.sigmoid_8 = nn.Sigmoid()

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_conv2d_2 = self.conv2d_2(opt_relu_1)
        opt_relu_3 = self.relu_3(opt_conv2d_2)
        opt_avgpool2d_4 = self.avgpool2d_4(opt_relu_3)
        opt_conv2d_5 = self.conv2d_5(opt_avgpool2d_4)
        opt_relu_6 = self.relu_6(opt_conv2d_5)
        opt_conv2d_7 = self.conv2d_7(opt_relu_6)
        opt_sigmoid_8 = self.sigmoid_8(opt_conv2d_7)
        opt_mul_9 = P.Mul()(opt_sigmoid_8, opt_relu_3)
        return opt_mul_9


class Module1(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, module0_0_conv2d_0_in_channels,
                 module0_0_conv2d_0_out_channels, module0_0_conv2d_2_in_channels, module0_0_conv2d_2_out_channels,
                 module0_0_conv2d_2_stride, module0_0_conv2d_2_group, module0_0_avgpool2d_4_kernel_size,
                 module0_0_conv2d_5_in_channels, module0_0_conv2d_5_out_channels, module0_0_conv2d_7_in_channels,
                 module0_0_conv2d_7_out_channels):
        super(Module1, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=module0_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_0_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_0_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_0_conv2d_2_out_channels,
                                 conv2d_2_stride=module0_0_conv2d_2_stride,
                                 conv2d_2_group=module0_0_conv2d_2_group,
                                 avgpool2d_4_kernel_size=module0_0_avgpool2d_4_kernel_size,
                                 conv2d_5_in_channels=module0_0_conv2d_5_in_channels,
                                 conv2d_5_out_channels=module0_0_conv2d_5_out_channels,
                                 conv2d_7_in_channels=module0_0_conv2d_7_in_channels,
                                 conv2d_7_out_channels=module0_0_conv2d_7_out_channels)
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
        module0_0_opt = self.module0_0(x)
        opt_conv2d_0 = self.conv2d_0(module0_0_opt)
        return opt_conv2d_0


class Module3(nn.Cell):
    def __init__(self):
        super(Module3, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=104,
                                 conv2d_0_out_channels=104,
                                 conv2d_2_in_channels=104,
                                 conv2d_2_out_channels=104,
                                 conv2d_2_stride=(1, 1),
                                 conv2d_2_group=13,
                                 avgpool2d_4_kernel_size=(28, 28),
                                 conv2d_5_in_channels=104,
                                 conv2d_5_out_channels=26,
                                 conv2d_7_in_channels=26,
                                 conv2d_7_out_channels=104)
        self.conv2d_0 = nn.Conv2d(in_channels=104,
                                  out_channels=104,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_2 = nn.ReLU()
        self.module0_1 = Module0(conv2d_0_in_channels=104,
                                 conv2d_0_out_channels=104,
                                 conv2d_2_in_channels=104,
                                 conv2d_2_out_channels=104,
                                 conv2d_2_stride=(1, 1),
                                 conv2d_2_group=13,
                                 avgpool2d_4_kernel_size=(28, 28),
                                 conv2d_5_in_channels=104,
                                 conv2d_5_out_channels=26,
                                 conv2d_7_in_channels=26,
                                 conv2d_7_out_channels=104)
        self.conv2d_3 = nn.Conv2d(in_channels=104,
                                  out_channels=104,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_5 = nn.ReLU()

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        opt_conv2d_0 = self.conv2d_0(module0_0_opt)
        opt_add_1 = P.Add()(x, opt_conv2d_0)
        opt_relu_2 = self.relu_2(opt_add_1)
        module0_1_opt = self.module0_1(opt_relu_2)
        opt_conv2d_3 = self.conv2d_3(module0_1_opt)
        opt_add_4 = P.Add()(opt_relu_2, opt_conv2d_3)
        opt_relu_5 = self.relu_5(opt_add_4)
        return opt_relu_5


class Module6(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_3_in_channels, conv2d_3_out_channels,
                 conv2d_6_in_channels, conv2d_6_out_channels, conv2d_9_in_channels, conv2d_9_out_channels,
                 conv2d_12_in_channels, conv2d_12_out_channels, module0_0_conv2d_0_in_channels,
                 module0_0_conv2d_0_out_channels, module0_0_conv2d_2_in_channels, module0_0_conv2d_2_out_channels,
                 module0_0_conv2d_2_stride, module0_0_conv2d_2_group, module0_0_avgpool2d_4_kernel_size,
                 module0_0_conv2d_5_in_channels, module0_0_conv2d_5_out_channels, module0_0_conv2d_7_in_channels,
                 module0_0_conv2d_7_out_channels, module0_1_conv2d_0_in_channels, module0_1_conv2d_0_out_channels,
                 module0_1_conv2d_2_in_channels, module0_1_conv2d_2_out_channels, module0_1_conv2d_2_stride,
                 module0_1_conv2d_2_group, module0_1_avgpool2d_4_kernel_size, module0_1_conv2d_5_in_channels,
                 module0_1_conv2d_5_out_channels, module0_1_conv2d_7_in_channels, module0_1_conv2d_7_out_channels,
                 module0_2_conv2d_0_in_channels, module0_2_conv2d_0_out_channels, module0_2_conv2d_2_in_channels,
                 module0_2_conv2d_2_out_channels, module0_2_conv2d_2_stride, module0_2_conv2d_2_group,
                 module0_2_avgpool2d_4_kernel_size, module0_2_conv2d_5_in_channels, module0_2_conv2d_5_out_channels,
                 module0_2_conv2d_7_in_channels, module0_2_conv2d_7_out_channels, module0_3_conv2d_0_in_channels,
                 module0_3_conv2d_0_out_channels, module0_3_conv2d_2_in_channels, module0_3_conv2d_2_out_channels,
                 module0_3_conv2d_2_stride, module0_3_conv2d_2_group, module0_3_avgpool2d_4_kernel_size,
                 module0_3_conv2d_5_in_channels, module0_3_conv2d_5_out_channels, module0_3_conv2d_7_in_channels,
                 module0_3_conv2d_7_out_channels, module0_4_conv2d_0_in_channels, module0_4_conv2d_0_out_channels,
                 module0_4_conv2d_2_in_channels, module0_4_conv2d_2_out_channels, module0_4_conv2d_2_stride,
                 module0_4_conv2d_2_group, module0_4_avgpool2d_4_kernel_size, module0_4_conv2d_5_in_channels,
                 module0_4_conv2d_5_out_channels, module0_4_conv2d_7_in_channels, module0_4_conv2d_7_out_channels):
        super(Module6, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=module0_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_0_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_0_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_0_conv2d_2_out_channels,
                                 conv2d_2_stride=module0_0_conv2d_2_stride,
                                 conv2d_2_group=module0_0_conv2d_2_group,
                                 avgpool2d_4_kernel_size=module0_0_avgpool2d_4_kernel_size,
                                 conv2d_5_in_channels=module0_0_conv2d_5_in_channels,
                                 conv2d_5_out_channels=module0_0_conv2d_5_out_channels,
                                 conv2d_7_in_channels=module0_0_conv2d_7_in_channels,
                                 conv2d_7_out_channels=module0_0_conv2d_7_out_channels)
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_2 = nn.ReLU()
        self.module0_1 = Module0(conv2d_0_in_channels=module0_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_1_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_1_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_1_conv2d_2_out_channels,
                                 conv2d_2_stride=module0_1_conv2d_2_stride,
                                 conv2d_2_group=module0_1_conv2d_2_group,
                                 avgpool2d_4_kernel_size=module0_1_avgpool2d_4_kernel_size,
                                 conv2d_5_in_channels=module0_1_conv2d_5_in_channels,
                                 conv2d_5_out_channels=module0_1_conv2d_5_out_channels,
                                 conv2d_7_in_channels=module0_1_conv2d_7_in_channels,
                                 conv2d_7_out_channels=module0_1_conv2d_7_out_channels)
        self.conv2d_3 = nn.Conv2d(in_channels=conv2d_3_in_channels,
                                  out_channels=conv2d_3_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_5 = nn.ReLU()
        self.module0_2 = Module0(conv2d_0_in_channels=module0_2_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_2_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_2_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_2_conv2d_2_out_channels,
                                 conv2d_2_stride=module0_2_conv2d_2_stride,
                                 conv2d_2_group=module0_2_conv2d_2_group,
                                 avgpool2d_4_kernel_size=module0_2_avgpool2d_4_kernel_size,
                                 conv2d_5_in_channels=module0_2_conv2d_5_in_channels,
                                 conv2d_5_out_channels=module0_2_conv2d_5_out_channels,
                                 conv2d_7_in_channels=module0_2_conv2d_7_in_channels,
                                 conv2d_7_out_channels=module0_2_conv2d_7_out_channels)
        self.conv2d_6 = nn.Conv2d(in_channels=conv2d_6_in_channels,
                                  out_channels=conv2d_6_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_8 = nn.ReLU()
        self.module0_3 = Module0(conv2d_0_in_channels=module0_3_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_3_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_3_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_3_conv2d_2_out_channels,
                                 conv2d_2_stride=module0_3_conv2d_2_stride,
                                 conv2d_2_group=module0_3_conv2d_2_group,
                                 avgpool2d_4_kernel_size=module0_3_avgpool2d_4_kernel_size,
                                 conv2d_5_in_channels=module0_3_conv2d_5_in_channels,
                                 conv2d_5_out_channels=module0_3_conv2d_5_out_channels,
                                 conv2d_7_in_channels=module0_3_conv2d_7_in_channels,
                                 conv2d_7_out_channels=module0_3_conv2d_7_out_channels)
        self.conv2d_9 = nn.Conv2d(in_channels=conv2d_9_in_channels,
                                  out_channels=conv2d_9_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_11 = nn.ReLU()
        self.module0_4 = Module0(conv2d_0_in_channels=module0_4_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_4_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_4_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_4_conv2d_2_out_channels,
                                 conv2d_2_stride=module0_4_conv2d_2_stride,
                                 conv2d_2_group=module0_4_conv2d_2_group,
                                 avgpool2d_4_kernel_size=module0_4_avgpool2d_4_kernel_size,
                                 conv2d_5_in_channels=module0_4_conv2d_5_in_channels,
                                 conv2d_5_out_channels=module0_4_conv2d_5_out_channels,
                                 conv2d_7_in_channels=module0_4_conv2d_7_in_channels,
                                 conv2d_7_out_channels=module0_4_conv2d_7_out_channels)
        self.conv2d_12 = nn.Conv2d(in_channels=conv2d_12_in_channels,
                                   out_channels=conv2d_12_out_channels,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_14 = nn.ReLU()

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        opt_conv2d_0 = self.conv2d_0(module0_0_opt)
        opt_add_1 = P.Add()(x, opt_conv2d_0)
        opt_relu_2 = self.relu_2(opt_add_1)
        module0_1_opt = self.module0_1(opt_relu_2)
        opt_conv2d_3 = self.conv2d_3(module0_1_opt)
        opt_add_4 = P.Add()(opt_relu_2, opt_conv2d_3)
        opt_relu_5 = self.relu_5(opt_add_4)
        module0_2_opt = self.module0_2(opt_relu_5)
        opt_conv2d_6 = self.conv2d_6(module0_2_opt)
        opt_add_7 = P.Add()(opt_relu_5, opt_conv2d_6)
        opt_relu_8 = self.relu_8(opt_add_7)
        module0_3_opt = self.module0_3(opt_relu_8)
        opt_conv2d_9 = self.conv2d_9(module0_3_opt)
        opt_add_10 = P.Add()(opt_relu_8, opt_conv2d_9)
        opt_relu_11 = self.relu_11(opt_add_10)
        module0_4_opt = self.module0_4(opt_relu_11)
        opt_conv2d_12 = self.conv2d_12(module0_4_opt)
        opt_add_13 = P.Add()(opt_relu_11, opt_conv2d_12)
        opt_relu_14 = self.relu_14(opt_add_13)
        return opt_relu_14


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
                                  out_channels=48,
                                  kernel_size=(1, 1),
                                  stride=(2, 2),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.module1_0 = Module1(conv2d_0_in_channels=48,
                                 conv2d_0_out_channels=48,
                                 module0_0_conv2d_0_in_channels=32,
                                 module0_0_conv2d_0_out_channels=48,
                                 module0_0_conv2d_2_in_channels=48,
                                 module0_0_conv2d_2_out_channels=48,
                                 module0_0_conv2d_2_stride=(2, 2),
                                 module0_0_conv2d_2_group=6,
                                 module0_0_avgpool2d_4_kernel_size=(56, 56),
                                 module0_0_conv2d_5_in_channels=48,
                                 module0_0_conv2d_5_out_channels=8,
                                 module0_0_conv2d_7_in_channels=8,
                                 module0_0_conv2d_7_out_channels=48)
        self.relu_15 = nn.ReLU()
        self.conv2d_16 = nn.Conv2d(in_channels=48,
                                   out_channels=104,
                                   kernel_size=(1, 1),
                                   stride=(2, 2),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.module1_1 = Module1(conv2d_0_in_channels=104,
                                 conv2d_0_out_channels=104,
                                 module0_0_conv2d_0_in_channels=48,
                                 module0_0_conv2d_0_out_channels=104,
                                 module0_0_conv2d_2_in_channels=104,
                                 module0_0_conv2d_2_out_channels=104,
                                 module0_0_conv2d_2_stride=(2, 2),
                                 module0_0_conv2d_2_group=13,
                                 module0_0_avgpool2d_4_kernel_size=(28, 28),
                                 module0_0_conv2d_5_in_channels=104,
                                 module0_0_conv2d_5_out_channels=12,
                                 module0_0_conv2d_7_in_channels=12,
                                 module0_0_conv2d_7_out_channels=104)
        self.relu_29 = nn.ReLU()
        self.module3_0 = Module3()
        self.conv2d_56 = nn.Conv2d(in_channels=104,
                                   out_channels=208,
                                   kernel_size=(1, 1),
                                   stride=(2, 2),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.module1_2 = Module1(conv2d_0_in_channels=208,
                                 conv2d_0_out_channels=208,
                                 module0_0_conv2d_0_in_channels=104,
                                 module0_0_conv2d_0_out_channels=208,
                                 module0_0_conv2d_2_in_channels=208,
                                 module0_0_conv2d_2_out_channels=208,
                                 module0_0_conv2d_2_stride=(2, 2),
                                 module0_0_conv2d_2_group=26,
                                 module0_0_avgpool2d_4_kernel_size=(14, 14),
                                 module0_0_conv2d_5_in_channels=208,
                                 module0_0_conv2d_5_out_channels=26,
                                 module0_0_conv2d_7_in_channels=26,
                                 module0_0_conv2d_7_out_channels=208)
        self.relu_69 = nn.ReLU()
        self.module6_0 = Module6(conv2d_0_in_channels=208,
                                 conv2d_0_out_channels=208,
                                 conv2d_3_in_channels=208,
                                 conv2d_3_out_channels=208,
                                 conv2d_6_in_channels=208,
                                 conv2d_6_out_channels=208,
                                 conv2d_9_in_channels=208,
                                 conv2d_9_out_channels=208,
                                 conv2d_12_in_channels=208,
                                 conv2d_12_out_channels=208,
                                 module0_0_conv2d_0_in_channels=208,
                                 module0_0_conv2d_0_out_channels=208,
                                 module0_0_conv2d_2_in_channels=208,
                                 module0_0_conv2d_2_out_channels=208,
                                 module0_0_conv2d_2_stride=(1, 1),
                                 module0_0_conv2d_2_group=26,
                                 module0_0_avgpool2d_4_kernel_size=(14, 14),
                                 module0_0_conv2d_5_in_channels=208,
                                 module0_0_conv2d_5_out_channels=52,
                                 module0_0_conv2d_7_in_channels=52,
                                 module0_0_conv2d_7_out_channels=208,
                                 module0_1_conv2d_0_in_channels=208,
                                 module0_1_conv2d_0_out_channels=208,
                                 module0_1_conv2d_2_in_channels=208,
                                 module0_1_conv2d_2_out_channels=208,
                                 module0_1_conv2d_2_stride=(1, 1),
                                 module0_1_conv2d_2_group=26,
                                 module0_1_avgpool2d_4_kernel_size=(14, 14),
                                 module0_1_conv2d_5_in_channels=208,
                                 module0_1_conv2d_5_out_channels=52,
                                 module0_1_conv2d_7_in_channels=52,
                                 module0_1_conv2d_7_out_channels=208,
                                 module0_2_conv2d_0_in_channels=208,
                                 module0_2_conv2d_0_out_channels=208,
                                 module0_2_conv2d_2_in_channels=208,
                                 module0_2_conv2d_2_out_channels=208,
                                 module0_2_conv2d_2_stride=(1, 1),
                                 module0_2_conv2d_2_group=26,
                                 module0_2_avgpool2d_4_kernel_size=(14, 14),
                                 module0_2_conv2d_5_in_channels=208,
                                 module0_2_conv2d_5_out_channels=52,
                                 module0_2_conv2d_7_in_channels=52,
                                 module0_2_conv2d_7_out_channels=208,
                                 module0_3_conv2d_0_in_channels=208,
                                 module0_3_conv2d_0_out_channels=208,
                                 module0_3_conv2d_2_in_channels=208,
                                 module0_3_conv2d_2_out_channels=208,
                                 module0_3_conv2d_2_stride=(1, 1),
                                 module0_3_conv2d_2_group=26,
                                 module0_3_avgpool2d_4_kernel_size=(14, 14),
                                 module0_3_conv2d_5_in_channels=208,
                                 module0_3_conv2d_5_out_channels=52,
                                 module0_3_conv2d_7_in_channels=52,
                                 module0_3_conv2d_7_out_channels=208,
                                 module0_4_conv2d_0_in_channels=208,
                                 module0_4_conv2d_0_out_channels=208,
                                 module0_4_conv2d_2_in_channels=208,
                                 module0_4_conv2d_2_out_channels=208,
                                 module0_4_conv2d_2_stride=(1, 1),
                                 module0_4_conv2d_2_group=26,
                                 module0_4_avgpool2d_4_kernel_size=(14, 14),
                                 module0_4_conv2d_5_in_channels=208,
                                 module0_4_conv2d_5_out_channels=52,
                                 module0_4_conv2d_7_in_channels=52,
                                 module0_4_conv2d_7_out_channels=208)
        self.conv2d_135 = nn.Conv2d(in_channels=208,
                                    out_channels=440,
                                    kernel_size=(1, 1),
                                    stride=(2, 2),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module1_3 = Module1(conv2d_0_in_channels=440,
                                 conv2d_0_out_channels=440,
                                 module0_0_conv2d_0_in_channels=208,
                                 module0_0_conv2d_0_out_channels=440,
                                 module0_0_conv2d_2_in_channels=440,
                                 module0_0_conv2d_2_out_channels=440,
                                 module0_0_conv2d_2_stride=(2, 2),
                                 module0_0_conv2d_2_group=55,
                                 module0_0_avgpool2d_4_kernel_size=(7, 7),
                                 module0_0_conv2d_5_in_channels=440,
                                 module0_0_conv2d_5_out_channels=52,
                                 module0_0_conv2d_7_in_channels=52,
                                 module0_0_conv2d_7_out_channels=440)
        self.relu_148 = nn.ReLU()
        self.module6_1 = Module6(conv2d_0_in_channels=440,
                                 conv2d_0_out_channels=440,
                                 conv2d_3_in_channels=440,
                                 conv2d_3_out_channels=440,
                                 conv2d_6_in_channels=440,
                                 conv2d_6_out_channels=440,
                                 conv2d_9_in_channels=440,
                                 conv2d_9_out_channels=440,
                                 conv2d_12_in_channels=440,
                                 conv2d_12_out_channels=440,
                                 module0_0_conv2d_0_in_channels=440,
                                 module0_0_conv2d_0_out_channels=440,
                                 module0_0_conv2d_2_in_channels=440,
                                 module0_0_conv2d_2_out_channels=440,
                                 module0_0_conv2d_2_stride=(1, 1),
                                 module0_0_conv2d_2_group=55,
                                 module0_0_avgpool2d_4_kernel_size=(7, 7),
                                 module0_0_conv2d_5_in_channels=440,
                                 module0_0_conv2d_5_out_channels=110,
                                 module0_0_conv2d_7_in_channels=110,
                                 module0_0_conv2d_7_out_channels=440,
                                 module0_1_conv2d_0_in_channels=440,
                                 module0_1_conv2d_0_out_channels=440,
                                 module0_1_conv2d_2_in_channels=440,
                                 module0_1_conv2d_2_out_channels=440,
                                 module0_1_conv2d_2_stride=(1, 1),
                                 module0_1_conv2d_2_group=55,
                                 module0_1_avgpool2d_4_kernel_size=(7, 7),
                                 module0_1_conv2d_5_in_channels=440,
                                 module0_1_conv2d_5_out_channels=110,
                                 module0_1_conv2d_7_in_channels=110,
                                 module0_1_conv2d_7_out_channels=440,
                                 module0_2_conv2d_0_in_channels=440,
                                 module0_2_conv2d_0_out_channels=440,
                                 module0_2_conv2d_2_in_channels=440,
                                 module0_2_conv2d_2_out_channels=440,
                                 module0_2_conv2d_2_stride=(1, 1),
                                 module0_2_conv2d_2_group=55,
                                 module0_2_avgpool2d_4_kernel_size=(7, 7),
                                 module0_2_conv2d_5_in_channels=440,
                                 module0_2_conv2d_5_out_channels=110,
                                 module0_2_conv2d_7_in_channels=110,
                                 module0_2_conv2d_7_out_channels=440,
                                 module0_3_conv2d_0_in_channels=440,
                                 module0_3_conv2d_0_out_channels=440,
                                 module0_3_conv2d_2_in_channels=440,
                                 module0_3_conv2d_2_out_channels=440,
                                 module0_3_conv2d_2_stride=(1, 1),
                                 module0_3_conv2d_2_group=55,
                                 module0_3_avgpool2d_4_kernel_size=(7, 7),
                                 module0_3_conv2d_5_in_channels=440,
                                 module0_3_conv2d_5_out_channels=110,
                                 module0_3_conv2d_7_in_channels=110,
                                 module0_3_conv2d_7_out_channels=440,
                                 module0_4_conv2d_0_in_channels=440,
                                 module0_4_conv2d_0_out_channels=440,
                                 module0_4_conv2d_2_in_channels=440,
                                 module0_4_conv2d_2_out_channels=440,
                                 module0_4_conv2d_2_stride=(1, 1),
                                 module0_4_conv2d_2_group=55,
                                 module0_4_avgpool2d_4_kernel_size=(7, 7),
                                 module0_4_conv2d_5_in_channels=440,
                                 module0_4_conv2d_5_out_channels=110,
                                 module0_4_conv2d_7_in_channels=110,
                                 module0_4_conv2d_7_out_channels=440)
        self.avgpool2d_214 = nn.AvgPool2d(kernel_size=(7, 7))
        self.flatten_215 = nn.Flatten()
        self.dense_216 = nn.Dense(in_channels=440, out_channels=1000, has_bias=True)

    def construct(self, input_1):
        opt_conv2d_0 = self.conv2d_0(input_1)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_conv2d_2 = self.conv2d_2(opt_relu_1)
        module1_0_opt = self.module1_0(opt_relu_1)
        opt_add_14 = P.Add()(opt_conv2d_2, module1_0_opt)
        opt_relu_15 = self.relu_15(opt_add_14)
        opt_conv2d_16 = self.conv2d_16(opt_relu_15)
        module1_1_opt = self.module1_1(opt_relu_15)
        opt_add_28 = P.Add()(opt_conv2d_16, module1_1_opt)
        opt_relu_29 = self.relu_29(opt_add_28)
        module3_0_opt = self.module3_0(opt_relu_29)
        opt_conv2d_56 = self.conv2d_56(module3_0_opt)
        module1_2_opt = self.module1_2(module3_0_opt)
        opt_add_68 = P.Add()(opt_conv2d_56, module1_2_opt)
        opt_relu_69 = self.relu_69(opt_add_68)
        module6_0_opt = self.module6_0(opt_relu_69)
        opt_conv2d_135 = self.conv2d_135(module6_0_opt)
        module1_3_opt = self.module1_3(module6_0_opt)
        opt_add_147 = P.Add()(opt_conv2d_135, module1_3_opt)
        opt_relu_148 = self.relu_148(opt_add_147)
        module6_1_opt = self.module6_1(opt_relu_148)
        opt_avgpool2d_214 = self.avgpool2d_214(module6_1_opt)
        opt_flatten_215 = self.flatten_215(opt_avgpool2d_214)
        opt_dense_216 = self.dense_216(opt_flatten_215)
        return opt_dense_216
