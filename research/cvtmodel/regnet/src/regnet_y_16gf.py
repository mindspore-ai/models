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


class Module2(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, module0_0_conv2d_0_in_channels,
                 module0_0_conv2d_0_out_channels, module0_0_conv2d_2_in_channels, module0_0_conv2d_2_out_channels,
                 module0_0_conv2d_2_stride, module0_0_conv2d_2_group, module0_0_avgpool2d_4_kernel_size,
                 module0_0_conv2d_5_in_channels, module0_0_conv2d_5_out_channels, module0_0_conv2d_7_in_channels,
                 module0_0_conv2d_7_out_channels):
        super(Module2, self).__init__()
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

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        opt_conv2d_0 = self.conv2d_0(module0_0_opt)
        opt_add_1 = P.Add()(x, opt_conv2d_0)
        opt_relu_2 = self.relu_2(opt_add_1)
        return opt_relu_2


class Module3(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_3_in_channels, conv2d_3_out_channels,
                 module0_0_conv2d_0_in_channels, module0_0_conv2d_0_out_channels, module0_0_conv2d_2_in_channels,
                 module0_0_conv2d_2_out_channels, module0_0_conv2d_2_stride, module0_0_conv2d_2_group,
                 module0_0_avgpool2d_4_kernel_size, module0_0_conv2d_5_in_channels, module0_0_conv2d_5_out_channels,
                 module0_0_conv2d_7_in_channels, module0_0_conv2d_7_out_channels, module0_1_conv2d_0_in_channels,
                 module0_1_conv2d_0_out_channels, module0_1_conv2d_2_in_channels, module0_1_conv2d_2_out_channels,
                 module0_1_conv2d_2_stride, module0_1_conv2d_2_group, module0_1_avgpool2d_4_kernel_size,
                 module0_1_conv2d_5_in_channels, module0_1_conv2d_5_out_channels, module0_1_conv2d_7_in_channels,
                 module0_1_conv2d_7_out_channels):
        super(Module3, self).__init__()
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


class Module4(nn.Cell):
    def __init__(self, module0_0_conv2d_0_in_channels, module0_0_conv2d_0_out_channels, module0_0_conv2d_2_in_channels,
                 module0_0_conv2d_2_out_channels, module0_0_conv2d_2_stride, module0_0_conv2d_2_group,
                 module0_0_avgpool2d_4_kernel_size, module0_0_conv2d_5_in_channels, module0_0_conv2d_5_out_channels,
                 module0_0_conv2d_7_in_channels, module0_0_conv2d_7_out_channels, module0_1_conv2d_0_in_channels,
                 module0_1_conv2d_0_out_channels, module0_1_conv2d_2_in_channels, module0_1_conv2d_2_out_channels,
                 module0_1_conv2d_2_stride, module0_1_conv2d_2_group, module0_1_avgpool2d_4_kernel_size,
                 module0_1_conv2d_5_in_channels, module0_1_conv2d_5_out_channels, module0_1_conv2d_7_in_channels,
                 module0_1_conv2d_7_out_channels, module0_2_conv2d_0_in_channels, module0_2_conv2d_0_out_channels,
                 module0_2_conv2d_2_in_channels, module0_2_conv2d_2_out_channels, module0_2_conv2d_2_stride,
                 module0_2_conv2d_2_group, module0_2_avgpool2d_4_kernel_size, module0_2_conv2d_5_in_channels,
                 module0_2_conv2d_5_out_channels, module0_2_conv2d_7_in_channels, module0_2_conv2d_7_out_channels,
                 module0_3_conv2d_0_in_channels, module0_3_conv2d_0_out_channels, module0_3_conv2d_2_in_channels,
                 module0_3_conv2d_2_out_channels, module0_3_conv2d_2_stride, module0_3_conv2d_2_group,
                 module0_3_avgpool2d_4_kernel_size, module0_3_conv2d_5_in_channels, module0_3_conv2d_5_out_channels,
                 module0_3_conv2d_7_in_channels, module0_3_conv2d_7_out_channels):
        super(Module4, self).__init__()
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
        self.conv2d_0 = nn.Conv2d(in_channels=1232,
                                  out_channels=1232,
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
        self.conv2d_3 = nn.Conv2d(in_channels=1232,
                                  out_channels=1232,
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
        self.conv2d_6 = nn.Conv2d(in_channels=1232,
                                  out_channels=1232,
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
        self.conv2d_9 = nn.Conv2d(in_channels=1232,
                                  out_channels=1232,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_11 = nn.ReLU()

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
        return opt_relu_11


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
                                  out_channels=224,
                                  kernel_size=(1, 1),
                                  stride=(2, 2),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.module1_0 = Module1(conv2d_0_in_channels=224,
                                 conv2d_0_out_channels=224,
                                 module0_0_conv2d_0_in_channels=32,
                                 module0_0_conv2d_0_out_channels=224,
                                 module0_0_conv2d_2_in_channels=224,
                                 module0_0_conv2d_2_out_channels=224,
                                 module0_0_conv2d_2_stride=(2, 2),
                                 module0_0_conv2d_2_group=2,
                                 module0_0_avgpool2d_4_kernel_size=(56, 56),
                                 module0_0_conv2d_5_in_channels=224,
                                 module0_0_conv2d_5_out_channels=8,
                                 module0_0_conv2d_7_in_channels=8,
                                 module0_0_conv2d_7_out_channels=224)
        self.relu_15 = nn.ReLU()
        self.module2_0 = Module2(conv2d_0_in_channels=224,
                                 conv2d_0_out_channels=224,
                                 module0_0_conv2d_0_in_channels=224,
                                 module0_0_conv2d_0_out_channels=224,
                                 module0_0_conv2d_2_in_channels=224,
                                 module0_0_conv2d_2_out_channels=224,
                                 module0_0_conv2d_2_stride=(1, 1),
                                 module0_0_conv2d_2_group=2,
                                 module0_0_avgpool2d_4_kernel_size=(56, 56),
                                 module0_0_conv2d_5_in_channels=224,
                                 module0_0_conv2d_5_out_channels=56,
                                 module0_0_conv2d_7_in_channels=56,
                                 module0_0_conv2d_7_out_channels=224)
        self.conv2d_29 = nn.Conv2d(in_channels=224,
                                   out_channels=448,
                                   kernel_size=(1, 1),
                                   stride=(2, 2),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.module1_1 = Module1(conv2d_0_in_channels=448,
                                 conv2d_0_out_channels=448,
                                 module0_0_conv2d_0_in_channels=224,
                                 module0_0_conv2d_0_out_channels=448,
                                 module0_0_conv2d_2_in_channels=448,
                                 module0_0_conv2d_2_out_channels=448,
                                 module0_0_conv2d_2_stride=(2, 2),
                                 module0_0_conv2d_2_group=4,
                                 module0_0_avgpool2d_4_kernel_size=(28, 28),
                                 module0_0_conv2d_5_in_channels=448,
                                 module0_0_conv2d_5_out_channels=56,
                                 module0_0_conv2d_7_in_channels=56,
                                 module0_0_conv2d_7_out_channels=448)
        self.relu_42 = nn.ReLU()
        self.module3_0 = Module3(conv2d_0_in_channels=448,
                                 conv2d_0_out_channels=448,
                                 conv2d_3_in_channels=448,
                                 conv2d_3_out_channels=448,
                                 module0_0_conv2d_0_in_channels=448,
                                 module0_0_conv2d_0_out_channels=448,
                                 module0_0_conv2d_2_in_channels=448,
                                 module0_0_conv2d_2_out_channels=448,
                                 module0_0_conv2d_2_stride=(1, 1),
                                 module0_0_conv2d_2_group=4,
                                 module0_0_avgpool2d_4_kernel_size=(28, 28),
                                 module0_0_conv2d_5_in_channels=448,
                                 module0_0_conv2d_5_out_channels=112,
                                 module0_0_conv2d_7_in_channels=112,
                                 module0_0_conv2d_7_out_channels=448,
                                 module0_1_conv2d_0_in_channels=448,
                                 module0_1_conv2d_0_out_channels=448,
                                 module0_1_conv2d_2_in_channels=448,
                                 module0_1_conv2d_2_out_channels=448,
                                 module0_1_conv2d_2_stride=(1, 1),
                                 module0_1_conv2d_2_group=4,
                                 module0_1_avgpool2d_4_kernel_size=(28, 28),
                                 module0_1_conv2d_5_in_channels=448,
                                 module0_1_conv2d_5_out_channels=112,
                                 module0_1_conv2d_7_in_channels=112,
                                 module0_1_conv2d_7_out_channels=448)
        self.module2_1 = Module2(conv2d_0_in_channels=448,
                                 conv2d_0_out_channels=448,
                                 module0_0_conv2d_0_in_channels=448,
                                 module0_0_conv2d_0_out_channels=448,
                                 module0_0_conv2d_2_in_channels=448,
                                 module0_0_conv2d_2_out_channels=448,
                                 module0_0_conv2d_2_stride=(1, 1),
                                 module0_0_conv2d_2_group=4,
                                 module0_0_avgpool2d_4_kernel_size=(28, 28),
                                 module0_0_conv2d_5_in_channels=448,
                                 module0_0_conv2d_5_out_channels=112,
                                 module0_0_conv2d_7_in_channels=112,
                                 module0_0_conv2d_7_out_channels=448)
        self.conv2d_82 = nn.Conv2d(in_channels=448,
                                   out_channels=1232,
                                   kernel_size=(1, 1),
                                   stride=(2, 2),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.module1_2 = Module1(conv2d_0_in_channels=1232,
                                 conv2d_0_out_channels=1232,
                                 module0_0_conv2d_0_in_channels=448,
                                 module0_0_conv2d_0_out_channels=1232,
                                 module0_0_conv2d_2_in_channels=1232,
                                 module0_0_conv2d_2_out_channels=1232,
                                 module0_0_conv2d_2_stride=(2, 2),
                                 module0_0_conv2d_2_group=11,
                                 module0_0_avgpool2d_4_kernel_size=(14, 14),
                                 module0_0_conv2d_5_in_channels=1232,
                                 module0_0_conv2d_5_out_channels=112,
                                 module0_0_conv2d_7_in_channels=112,
                                 module0_0_conv2d_7_out_channels=1232)
        self.relu_95 = nn.ReLU()
        self.module4_0 = Module4(module0_0_conv2d_0_in_channels=1232,
                                 module0_0_conv2d_0_out_channels=1232,
                                 module0_0_conv2d_2_in_channels=1232,
                                 module0_0_conv2d_2_out_channels=1232,
                                 module0_0_conv2d_2_stride=(1, 1),
                                 module0_0_conv2d_2_group=11,
                                 module0_0_avgpool2d_4_kernel_size=(14, 14),
                                 module0_0_conv2d_5_in_channels=1232,
                                 module0_0_conv2d_5_out_channels=308,
                                 module0_0_conv2d_7_in_channels=308,
                                 module0_0_conv2d_7_out_channels=1232,
                                 module0_1_conv2d_0_in_channels=1232,
                                 module0_1_conv2d_0_out_channels=1232,
                                 module0_1_conv2d_2_in_channels=1232,
                                 module0_1_conv2d_2_out_channels=1232,
                                 module0_1_conv2d_2_stride=(1, 1),
                                 module0_1_conv2d_2_group=11,
                                 module0_1_avgpool2d_4_kernel_size=(14, 14),
                                 module0_1_conv2d_5_in_channels=1232,
                                 module0_1_conv2d_5_out_channels=308,
                                 module0_1_conv2d_7_in_channels=308,
                                 module0_1_conv2d_7_out_channels=1232,
                                 module0_2_conv2d_0_in_channels=1232,
                                 module0_2_conv2d_0_out_channels=1232,
                                 module0_2_conv2d_2_in_channels=1232,
                                 module0_2_conv2d_2_out_channels=1232,
                                 module0_2_conv2d_2_stride=(1, 1),
                                 module0_2_conv2d_2_group=11,
                                 module0_2_avgpool2d_4_kernel_size=(14, 14),
                                 module0_2_conv2d_5_in_channels=1232,
                                 module0_2_conv2d_5_out_channels=308,
                                 module0_2_conv2d_7_in_channels=308,
                                 module0_2_conv2d_7_out_channels=1232,
                                 module0_3_conv2d_0_in_channels=1232,
                                 module0_3_conv2d_0_out_channels=1232,
                                 module0_3_conv2d_2_in_channels=1232,
                                 module0_3_conv2d_2_out_channels=1232,
                                 module0_3_conv2d_2_stride=(1, 1),
                                 module0_3_conv2d_2_group=11,
                                 module0_3_avgpool2d_4_kernel_size=(14, 14),
                                 module0_3_conv2d_5_in_channels=1232,
                                 module0_3_conv2d_5_out_channels=308,
                                 module0_3_conv2d_7_in_channels=308,
                                 module0_3_conv2d_7_out_channels=1232)
        self.module4_1 = Module4(module0_0_conv2d_0_in_channels=1232,
                                 module0_0_conv2d_0_out_channels=1232,
                                 module0_0_conv2d_2_in_channels=1232,
                                 module0_0_conv2d_2_out_channels=1232,
                                 module0_0_conv2d_2_stride=(1, 1),
                                 module0_0_conv2d_2_group=11,
                                 module0_0_avgpool2d_4_kernel_size=(14, 14),
                                 module0_0_conv2d_5_in_channels=1232,
                                 module0_0_conv2d_5_out_channels=308,
                                 module0_0_conv2d_7_in_channels=308,
                                 module0_0_conv2d_7_out_channels=1232,
                                 module0_1_conv2d_0_in_channels=1232,
                                 module0_1_conv2d_0_out_channels=1232,
                                 module0_1_conv2d_2_in_channels=1232,
                                 module0_1_conv2d_2_out_channels=1232,
                                 module0_1_conv2d_2_stride=(1, 1),
                                 module0_1_conv2d_2_group=11,
                                 module0_1_avgpool2d_4_kernel_size=(14, 14),
                                 module0_1_conv2d_5_in_channels=1232,
                                 module0_1_conv2d_5_out_channels=308,
                                 module0_1_conv2d_7_in_channels=308,
                                 module0_1_conv2d_7_out_channels=1232,
                                 module0_2_conv2d_0_in_channels=1232,
                                 module0_2_conv2d_0_out_channels=1232,
                                 module0_2_conv2d_2_in_channels=1232,
                                 module0_2_conv2d_2_out_channels=1232,
                                 module0_2_conv2d_2_stride=(1, 1),
                                 module0_2_conv2d_2_group=11,
                                 module0_2_avgpool2d_4_kernel_size=(14, 14),
                                 module0_2_conv2d_5_in_channels=1232,
                                 module0_2_conv2d_5_out_channels=308,
                                 module0_2_conv2d_7_in_channels=308,
                                 module0_2_conv2d_7_out_channels=1232,
                                 module0_3_conv2d_0_in_channels=1232,
                                 module0_3_conv2d_0_out_channels=1232,
                                 module0_3_conv2d_2_in_channels=1232,
                                 module0_3_conv2d_2_out_channels=1232,
                                 module0_3_conv2d_2_stride=(1, 1),
                                 module0_3_conv2d_2_group=11,
                                 module0_3_avgpool2d_4_kernel_size=(14, 14),
                                 module0_3_conv2d_5_in_channels=1232,
                                 module0_3_conv2d_5_out_channels=308,
                                 module0_3_conv2d_7_in_channels=308,
                                 module0_3_conv2d_7_out_channels=1232)
        self.module3_1 = Module3(conv2d_0_in_channels=1232,
                                 conv2d_0_out_channels=1232,
                                 conv2d_3_in_channels=1232,
                                 conv2d_3_out_channels=1232,
                                 module0_0_conv2d_0_in_channels=1232,
                                 module0_0_conv2d_0_out_channels=1232,
                                 module0_0_conv2d_2_in_channels=1232,
                                 module0_0_conv2d_2_out_channels=1232,
                                 module0_0_conv2d_2_stride=(1, 1),
                                 module0_0_conv2d_2_group=11,
                                 module0_0_avgpool2d_4_kernel_size=(14, 14),
                                 module0_0_conv2d_5_in_channels=1232,
                                 module0_0_conv2d_5_out_channels=308,
                                 module0_0_conv2d_7_in_channels=308,
                                 module0_0_conv2d_7_out_channels=1232,
                                 module0_1_conv2d_0_in_channels=1232,
                                 module0_1_conv2d_0_out_channels=1232,
                                 module0_1_conv2d_2_in_channels=1232,
                                 module0_1_conv2d_2_out_channels=1232,
                                 module0_1_conv2d_2_stride=(1, 1),
                                 module0_1_conv2d_2_group=11,
                                 module0_1_avgpool2d_4_kernel_size=(14, 14),
                                 module0_1_conv2d_5_in_channels=1232,
                                 module0_1_conv2d_5_out_channels=308,
                                 module0_1_conv2d_7_in_channels=308,
                                 module0_1_conv2d_7_out_channels=1232)
        self.conv2d_226 = nn.Conv2d(in_channels=1232,
                                    out_channels=3024,
                                    kernel_size=(1, 1),
                                    stride=(2, 2),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module1_3 = Module1(conv2d_0_in_channels=3024,
                                 conv2d_0_out_channels=3024,
                                 module0_0_conv2d_0_in_channels=1232,
                                 module0_0_conv2d_0_out_channels=3024,
                                 module0_0_conv2d_2_in_channels=3024,
                                 module0_0_conv2d_2_out_channels=3024,
                                 module0_0_conv2d_2_stride=(2, 2),
                                 module0_0_conv2d_2_group=27,
                                 module0_0_avgpool2d_4_kernel_size=(7, 7),
                                 module0_0_conv2d_5_in_channels=3024,
                                 module0_0_conv2d_5_out_channels=308,
                                 module0_0_conv2d_7_in_channels=308,
                                 module0_0_conv2d_7_out_channels=3024)
        self.relu_239 = nn.ReLU()
        self.avgpool2d_240 = nn.AvgPool2d(kernel_size=(7, 7))
        self.flatten_241 = nn.Flatten()
        self.dense_242 = nn.Dense(in_channels=3024, out_channels=1000, has_bias=True)

    def construct(self, input_1):
        opt_conv2d_0 = self.conv2d_0(input_1)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_conv2d_2 = self.conv2d_2(opt_relu_1)
        module1_0_opt = self.module1_0(opt_relu_1)
        opt_add_14 = P.Add()(opt_conv2d_2, module1_0_opt)
        opt_relu_15 = self.relu_15(opt_add_14)
        module2_0_opt = self.module2_0(opt_relu_15)
        opt_conv2d_29 = self.conv2d_29(module2_0_opt)
        module1_1_opt = self.module1_1(module2_0_opt)
        opt_add_41 = P.Add()(opt_conv2d_29, module1_1_opt)
        opt_relu_42 = self.relu_42(opt_add_41)
        module3_0_opt = self.module3_0(opt_relu_42)
        module2_1_opt = self.module2_1(module3_0_opt)
        opt_conv2d_82 = self.conv2d_82(module2_1_opt)
        module1_2_opt = self.module1_2(module2_1_opt)
        opt_add_94 = P.Add()(opt_conv2d_82, module1_2_opt)
        opt_relu_95 = self.relu_95(opt_add_94)
        module4_0_opt = self.module4_0(opt_relu_95)
        module4_1_opt = self.module4_1(module4_0_opt)
        module3_1_opt = self.module3_1(module4_1_opt)
        opt_conv2d_226 = self.conv2d_226(module3_1_opt)
        module1_3_opt = self.module1_3(module3_1_opt)
        opt_add_238 = P.Add()(opt_conv2d_226, module1_3_opt)
        opt_relu_239 = self.relu_239(opt_add_238)
        opt_avgpool2d_240 = self.avgpool2d_240(opt_relu_239)
        opt_flatten_241 = self.flatten_241(opt_avgpool2d_240)
        opt_dense_242 = self.dense_242(opt_flatten_241)
        return opt_dense_242
