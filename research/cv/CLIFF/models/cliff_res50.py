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

import mindspore.ops as P
from mindspore import nn


class Module9(nn.Cell):

    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_0_kernel_size, conv2d_0_stride,
                 conv2d_0_padding, conv2d_0_pad_mode):
        super(Module9, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=conv2d_0_kernel_size,
                                  stride=conv2d_0_stride,
                                  padding=conv2d_0_padding,
                                  pad_mode=conv2d_0_pad_mode,
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        return opt_relu_1


class Module18(nn.Cell):

    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, module9_0_conv2d_0_in_channels,
                 module9_0_conv2d_0_out_channels, module9_0_conv2d_0_kernel_size, module9_0_conv2d_0_stride,
                 module9_0_conv2d_0_padding, module9_0_conv2d_0_pad_mode, module9_1_conv2d_0_in_channels,
                 module9_1_conv2d_0_out_channels, module9_1_conv2d_0_kernel_size, module9_1_conv2d_0_stride,
                 module9_1_conv2d_0_padding, module9_1_conv2d_0_pad_mode):
        super(Module18, self).__init__()
        self.module9_0 = Module9(conv2d_0_in_channels=module9_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module9_0_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module9_0_conv2d_0_kernel_size,
                                 conv2d_0_stride=module9_0_conv2d_0_stride,
                                 conv2d_0_padding=module9_0_conv2d_0_padding,
                                 conv2d_0_pad_mode=module9_0_conv2d_0_pad_mode)
        self.module9_1 = Module9(conv2d_0_in_channels=module9_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module9_1_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module9_1_conv2d_0_kernel_size,
                                 conv2d_0_stride=module9_1_conv2d_0_stride,
                                 conv2d_0_padding=module9_1_conv2d_0_padding,
                                 conv2d_0_pad_mode=module9_1_conv2d_0_pad_mode)
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
        module9_0_opt = self.module9_0(x)
        module9_1_opt = self.module9_1(module9_0_opt)
        opt_conv2d_0 = self.conv2d_0(module9_1_opt)
        return opt_conv2d_0


class Module0(nn.Cell):

    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_0_stride, conv2d_3_in_channels,
                 conv2d_3_out_channels, conv2d_5_in_channels, conv2d_5_out_channels, conv2d_7_in_channels,
                 conv2d_7_out_channels):
        super(Module0, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(1, 1),
                                  stride=conv2d_0_stride,
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_2 = nn.ReLU()
        self.conv2d_3 = nn.Conv2d(in_channels=conv2d_3_in_channels,
                                  out_channels=conv2d_3_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_4 = nn.ReLU()
        self.conv2d_5 = nn.Conv2d(in_channels=conv2d_5_in_channels,
                                  out_channels=conv2d_5_out_channels,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
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
        self.relu_9 = nn.ReLU()

    def construct(self, x, x0):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_add_1 = P.Add()(x0, opt_conv2d_0)
        opt_relu_2 = self.relu_2(opt_add_1)
        opt_conv2d_3 = self.conv2d_3(opt_relu_2)
        opt_relu_4 = self.relu_4(opt_conv2d_3)
        opt_conv2d_5 = self.conv2d_5(opt_relu_4)
        opt_relu_6 = self.relu_6(opt_conv2d_5)
        opt_conv2d_7 = self.conv2d_7(opt_relu_6)
        opt_add_8 = P.Add()(opt_conv2d_7, opt_relu_2)
        opt_relu_9 = self.relu_9(opt_add_8)
        return opt_relu_9


class Module25(nn.Cell):

    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, module9_0_conv2d_0_in_channels,
                 module9_0_conv2d_0_out_channels, module9_0_conv2d_0_kernel_size, module9_0_conv2d_0_stride,
                 module9_0_conv2d_0_padding, module9_0_conv2d_0_pad_mode, module9_1_conv2d_0_in_channels,
                 module9_1_conv2d_0_out_channels, module9_1_conv2d_0_kernel_size, module9_1_conv2d_0_stride,
                 module9_1_conv2d_0_padding, module9_1_conv2d_0_pad_mode):
        super(Module25, self).__init__()
        self.module9_0 = Module9(conv2d_0_in_channels=module9_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module9_0_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module9_0_conv2d_0_kernel_size,
                                 conv2d_0_stride=module9_0_conv2d_0_stride,
                                 conv2d_0_padding=module9_0_conv2d_0_padding,
                                 conv2d_0_pad_mode=module9_0_conv2d_0_pad_mode)
        self.module9_1 = Module9(conv2d_0_in_channels=module9_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module9_1_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module9_1_conv2d_0_kernel_size,
                                 conv2d_0_stride=module9_1_conv2d_0_stride,
                                 conv2d_0_padding=module9_1_conv2d_0_padding,
                                 conv2d_0_pad_mode=module9_1_conv2d_0_pad_mode)
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
        module9_0_opt = self.module9_0(x)
        module9_1_opt = self.module9_1(module9_0_opt)
        opt_conv2d_0 = self.conv2d_0(module9_1_opt)
        opt_add_1 = P.Add()(opt_conv2d_0, x)
        opt_relu_2 = self.relu_2(opt_add_1)
        return opt_relu_2


class Module36(nn.Cell):

    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_2_in_channels, conv2d_2_out_channels,
                 conv2d_4_in_channels, conv2d_4_out_channels, module0_0_conv2d_0_in_channels,
                 module0_0_conv2d_0_out_channels, module0_0_conv2d_0_stride, module0_0_conv2d_3_in_channels,
                 module0_0_conv2d_3_out_channels, module0_0_conv2d_5_in_channels, module0_0_conv2d_5_out_channels,
                 module0_0_conv2d_7_in_channels, module0_0_conv2d_7_out_channels, module9_0_conv2d_0_in_channels,
                 module9_0_conv2d_0_out_channels, module9_0_conv2d_0_kernel_size, module9_0_conv2d_0_stride,
                 module9_0_conv2d_0_padding, module9_0_conv2d_0_pad_mode, module9_1_conv2d_0_in_channels,
                 module9_1_conv2d_0_out_channels, module9_1_conv2d_0_kernel_size, module9_1_conv2d_0_stride,
                 module9_1_conv2d_0_padding, module9_1_conv2d_0_pad_mode, module0_1_conv2d_0_in_channels,
                 module0_1_conv2d_0_out_channels, module0_1_conv2d_0_stride, module0_1_conv2d_3_in_channels,
                 module0_1_conv2d_3_out_channels, module0_1_conv2d_5_in_channels, module0_1_conv2d_5_out_channels,
                 module0_1_conv2d_7_in_channels, module0_1_conv2d_7_out_channels, module0_2_conv2d_0_in_channels,
                 module0_2_conv2d_0_out_channels, module0_2_conv2d_0_stride, module0_2_conv2d_3_in_channels,
                 module0_2_conv2d_3_out_channels, module0_2_conv2d_5_in_channels, module0_2_conv2d_5_out_channels,
                 module0_2_conv2d_7_in_channels, module0_2_conv2d_7_out_channels):
        super(Module36, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=module0_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_0_conv2d_0_out_channels,
                                 conv2d_0_stride=module0_0_conv2d_0_stride,
                                 conv2d_3_in_channels=module0_0_conv2d_3_in_channels,
                                 conv2d_3_out_channels=module0_0_conv2d_3_out_channels,
                                 conv2d_5_in_channels=module0_0_conv2d_5_in_channels,
                                 conv2d_5_out_channels=module0_0_conv2d_5_out_channels,
                                 conv2d_7_in_channels=module0_0_conv2d_7_in_channels,
                                 conv2d_7_out_channels=module0_0_conv2d_7_out_channels)
        self.module9_0 = Module9(conv2d_0_in_channels=module9_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module9_0_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module9_0_conv2d_0_kernel_size,
                                 conv2d_0_stride=module9_0_conv2d_0_stride,
                                 conv2d_0_padding=module9_0_conv2d_0_padding,
                                 conv2d_0_pad_mode=module9_0_conv2d_0_pad_mode)
        self.module9_1 = Module9(conv2d_0_in_channels=module9_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module9_1_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module9_1_conv2d_0_kernel_size,
                                 conv2d_0_stride=module9_1_conv2d_0_stride,
                                 conv2d_0_padding=module9_1_conv2d_0_padding,
                                 conv2d_0_pad_mode=module9_1_conv2d_0_pad_mode)
        self.module0_1 = Module0(conv2d_0_in_channels=module0_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_1_conv2d_0_out_channels,
                                 conv2d_0_stride=module0_1_conv2d_0_stride,
                                 conv2d_3_in_channels=module0_1_conv2d_3_in_channels,
                                 conv2d_3_out_channels=module0_1_conv2d_3_out_channels,
                                 conv2d_5_in_channels=module0_1_conv2d_5_in_channels,
                                 conv2d_5_out_channels=module0_1_conv2d_5_out_channels,
                                 conv2d_7_in_channels=module0_1_conv2d_7_in_channels,
                                 conv2d_7_out_channels=module0_1_conv2d_7_out_channels)
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
                                  stride=(2, 2),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
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
        self.module0_2 = Module0(conv2d_0_in_channels=module0_2_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_2_conv2d_0_out_channels,
                                 conv2d_0_stride=module0_2_conv2d_0_stride,
                                 conv2d_3_in_channels=module0_2_conv2d_3_in_channels,
                                 conv2d_3_out_channels=module0_2_conv2d_3_out_channels,
                                 conv2d_5_in_channels=module0_2_conv2d_5_in_channels,
                                 conv2d_5_out_channels=module0_2_conv2d_5_out_channels,
                                 conv2d_7_in_channels=module0_2_conv2d_7_in_channels,
                                 conv2d_7_out_channels=module0_2_conv2d_7_out_channels)

    def construct(self, x, x0):
        module0_0_opt = self.module0_0(x, x0)
        module9_0_opt = self.module9_0(module0_0_opt)
        module9_1_opt = self.module9_1(module9_0_opt)
        module0_1_opt = self.module0_1(module9_1_opt, module0_0_opt)
        opt_conv2d_0 = self.conv2d_0(module0_1_opt)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_conv2d_2 = self.conv2d_2(opt_relu_1)
        opt_relu_3 = self.relu_3(opt_conv2d_2)
        opt_conv2d_4 = self.conv2d_4(opt_relu_3)
        module0_2_opt = self.module0_2(module0_1_opt, opt_conv2d_4)
        return module0_2_opt


class Module13(nn.Cell):

    def __init__(self):
        super(Module13, self).__init__()
        self.module9_0 = Module9(conv2d_0_in_channels=1024,
                                 conv2d_0_out_channels=256,
                                 conv2d_0_kernel_size=(1, 1),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=0,
                                 conv2d_0_pad_mode="valid")
        self.module9_1 = Module9(conv2d_0_in_channels=256,
                                 conv2d_0_out_channels=256,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad")

    def construct(self, x):
        module9_0_opt = self.module9_0(x)
        module9_1_opt = self.module9_1(module9_0_opt)
        return module9_1_opt


class Module29(nn.Cell):

    def __init__(self):
        super(Module29, self).__init__()
        self.dense_0 = nn.Dense(in_channels=2208, out_channels=1024, has_bias=True)
        self.dense_1 = nn.Dense(in_channels=1024, out_channels=1024, has_bias=True)

    def construct(self, x):
        opt_dense_0 = self.dense_0(x)
        opt_dense_1 = self.dense_1(opt_dense_0)
        return opt_dense_1


class Module35(nn.Cell):

    def __init__(self):
        super(Module35, self).__init__()
        self.concat_0 = P.Concat(axis=1)
        self.module29_0 = Module29()

    def construct(self, x, x0, x1, x2, x3):
        opt_concat_0 = self.concat_0((x, x0, x1, x2, x3))
        module29_0_opt = self.module29_0(opt_concat_0)
        return module29_0_opt


class MindSporeModel(nn.Cell):

    def __init__(self):
        super(MindSporeModel, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2),
                                  padding=(3, 3, 3, 3), pad_mode="pad", dilation=(1, 1), group=1, has_bias=True)
        self.relu_1 = nn.ReLU()
        self.pad_maxpool2d_2 = nn.Pad(paddings=((0, 0), (0, 0), (1, 0), (1, 0)))
        self.maxpool2d_2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.module18_0 = Module18(conv2d_0_in_channels=64, conv2d_0_out_channels=256,
                                   module9_0_conv2d_0_in_channels=64, module9_0_conv2d_0_out_channels=64,
                                   module9_0_conv2d_0_kernel_size=(1, 1), module9_0_conv2d_0_stride=(1, 1),
                                   module9_0_conv2d_0_padding=0, module9_0_conv2d_0_pad_mode="valid",
                                   module9_1_conv2d_0_in_channels=64, module9_1_conv2d_0_out_channels=64,
                                   module9_1_conv2d_0_kernel_size=(3, 3), module9_1_conv2d_0_stride=(1, 1),
                                   module9_1_conv2d_0_padding=(1, 1, 1, 1), module9_1_conv2d_0_pad_mode="pad")
        self.module0_0 = Module0(conv2d_0_in_channels=64, conv2d_0_out_channels=256, conv2d_0_stride=(1, 1),
                                 conv2d_3_in_channels=256, conv2d_3_out_channels=64, conv2d_5_in_channels=64,
                                 conv2d_5_out_channels=64, conv2d_7_in_channels=64, conv2d_7_out_channels=256)
        self.module25_0 = Module25(conv2d_0_in_channels=64, conv2d_0_out_channels=256,
                                   module9_0_conv2d_0_in_channels=256, module9_0_conv2d_0_out_channels=64,
                                   module9_0_conv2d_0_kernel_size=(1, 1), module9_0_conv2d_0_stride=(1, 1),
                                   module9_0_conv2d_0_padding=0, module9_0_conv2d_0_pad_mode="valid",
                                   module9_1_conv2d_0_in_channels=64, module9_1_conv2d_0_out_channels=64,
                                   module9_1_conv2d_0_kernel_size=(3, 3), module9_1_conv2d_0_stride=(1, 1),
                                   module9_1_conv2d_0_padding=(1, 1, 1, 1), module9_1_conv2d_0_pad_mode="pad")
        self.module18_1 = Module18(conv2d_0_in_channels=128, conv2d_0_out_channels=512,
                                   module9_0_conv2d_0_in_channels=256, module9_0_conv2d_0_out_channels=128,
                                   module9_0_conv2d_0_kernel_size=(1, 1), module9_0_conv2d_0_stride=(1, 1),
                                   module9_0_conv2d_0_padding=0, module9_0_conv2d_0_pad_mode="valid",
                                   module9_1_conv2d_0_in_channels=128, module9_1_conv2d_0_out_channels=128,
                                   module9_1_conv2d_0_kernel_size=(3, 3), module9_1_conv2d_0_stride=(2, 2),
                                   module9_1_conv2d_0_padding=(1, 1, 1, 1), module9_1_conv2d_0_pad_mode="pad")
        self.module36_0 = Module36(conv2d_0_in_channels=512, conv2d_0_out_channels=256, conv2d_2_in_channels=256,
                                   conv2d_2_out_channels=256, conv2d_4_in_channels=256, conv2d_4_out_channels=1024,
                                   module0_0_conv2d_0_in_channels=256, module0_0_conv2d_0_out_channels=512,
                                   module0_0_conv2d_0_stride=(2, 2), module0_0_conv2d_3_in_channels=512,
                                   module0_0_conv2d_3_out_channels=128, module0_0_conv2d_5_in_channels=128,
                                   module0_0_conv2d_5_out_channels=128, module0_0_conv2d_7_in_channels=128,
                                   module0_0_conv2d_7_out_channels=512, module9_0_conv2d_0_in_channels=512,
                                   module9_0_conv2d_0_out_channels=128, module9_0_conv2d_0_kernel_size=(1, 1),
                                   module9_0_conv2d_0_stride=(1, 1), module9_0_conv2d_0_padding=0,
                                   module9_0_conv2d_0_pad_mode="valid", module9_1_conv2d_0_in_channels=128,
                                   module9_1_conv2d_0_out_channels=128, module9_1_conv2d_0_kernel_size=(3, 3),
                                   module9_1_conv2d_0_stride=(1, 1), module9_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module9_1_conv2d_0_pad_mode="pad", module0_1_conv2d_0_in_channels=128,
                                   module0_1_conv2d_0_out_channels=512, module0_1_conv2d_0_stride=(1, 1),
                                   module0_1_conv2d_3_in_channels=512, module0_1_conv2d_3_out_channels=128,
                                   module0_1_conv2d_5_in_channels=128, module0_1_conv2d_5_out_channels=128,
                                   module0_1_conv2d_7_in_channels=128, module0_1_conv2d_7_out_channels=512,
                                   module0_2_conv2d_0_in_channels=512, module0_2_conv2d_0_out_channels=1024,
                                   module0_2_conv2d_0_stride=(2, 2), module0_2_conv2d_3_in_channels=1024,
                                   module0_2_conv2d_3_out_channels=256, module0_2_conv2d_5_in_channels=256,
                                   module0_2_conv2d_5_out_channels=256, module0_2_conv2d_7_in_channels=256,
                                   module0_2_conv2d_7_out_channels=1024)
        self.module13_0 = Module13()
        self.module36_1 = Module36(conv2d_0_in_channels=1024, conv2d_0_out_channels=512, conv2d_2_in_channels=512,
                                   conv2d_2_out_channels=512, conv2d_4_in_channels=512, conv2d_4_out_channels=2048,
                                   module0_0_conv2d_0_in_channels=256, module0_0_conv2d_0_out_channels=1024,
                                   module0_0_conv2d_0_stride=(1, 1), module0_0_conv2d_3_in_channels=1024,
                                   module0_0_conv2d_3_out_channels=256, module0_0_conv2d_5_in_channels=256,
                                   module0_0_conv2d_5_out_channels=256, module0_0_conv2d_7_in_channels=256,
                                   module0_0_conv2d_7_out_channels=1024, module9_0_conv2d_0_in_channels=1024,
                                   module9_0_conv2d_0_out_channels=256, module9_0_conv2d_0_kernel_size=(1, 1),
                                   module9_0_conv2d_0_stride=(1, 1), module9_0_conv2d_0_padding=0,
                                   module9_0_conv2d_0_pad_mode="valid", module9_1_conv2d_0_in_channels=256,
                                   module9_1_conv2d_0_out_channels=256, module9_1_conv2d_0_kernel_size=(3, 3),
                                   module9_1_conv2d_0_stride=(1, 1), module9_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module9_1_conv2d_0_pad_mode="pad", module0_1_conv2d_0_in_channels=256,
                                   module0_1_conv2d_0_out_channels=1024, module0_1_conv2d_0_stride=(1, 1),
                                   module0_1_conv2d_3_in_channels=1024, module0_1_conv2d_3_out_channels=256,
                                   module0_1_conv2d_5_in_channels=256, module0_1_conv2d_5_out_channels=256,
                                   module0_1_conv2d_7_in_channels=256, module0_1_conv2d_7_out_channels=1024,
                                   module0_2_conv2d_0_in_channels=1024, module0_2_conv2d_0_out_channels=2048,
                                   module0_2_conv2d_0_stride=(2, 2), module0_2_conv2d_3_in_channels=2048,
                                   module0_2_conv2d_3_out_channels=512, module0_2_conv2d_5_in_channels=512,
                                   module0_2_conv2d_5_out_channels=512, module0_2_conv2d_7_in_channels=512,
                                   module0_2_conv2d_7_out_channels=2048)
        self.module25_1 = Module25(conv2d_0_in_channels=512, conv2d_0_out_channels=2048,
                                   module9_0_conv2d_0_in_channels=2048, module9_0_conv2d_0_out_channels=512,
                                   module9_0_conv2d_0_kernel_size=(1, 1), module9_0_conv2d_0_stride=(1, 1),
                                   module9_0_conv2d_0_padding=0, module9_0_conv2d_0_pad_mode="valid",
                                   module9_1_conv2d_0_in_channels=512, module9_1_conv2d_0_out_channels=512,
                                   module9_1_conv2d_0_kernel_size=(3, 3), module9_1_conv2d_0_stride=(1, 1),
                                   module9_1_conv2d_0_padding=(1, 1, 1, 1), module9_1_conv2d_0_pad_mode="pad")
        self.avgpool2d_119 = nn.AvgPool2d(kernel_size=(8, 6))
        self.flatten_120 = nn.Flatten()
        self.concat_121 = P.Concat(axis=1)
        self.module29_0 = Module29()
        self.dense_124 = nn.Dense(in_channels=1024, out_channels=144, has_bias=True)
        self.dense_125 = nn.Dense(in_channels=1024, out_channels=10, has_bias=True)
        self.dense_126 = nn.Dense(in_channels=1024, out_channels=3, has_bias=True)
        self.module35_0 = Module35()
        self.dense_133 = nn.Dense(in_channels=1024, out_channels=144, has_bias=True)
        self.dense_134 = nn.Dense(in_channels=1024, out_channels=10, has_bias=True)
        self.dense_135 = nn.Dense(in_channels=1024, out_channels=3, has_bias=True)
        self.module35_1 = Module35()
        self.dense_142 = nn.Dense(in_channels=1024, out_channels=144, has_bias=True)
        self.dense_143 = nn.Dense(in_channels=1024, out_channels=10, has_bias=True)
        self.dense_144 = nn.Dense(in_channels=1024, out_channels=3, has_bias=True)

    def construct(self, inp, x0, x1, x2, x3):
        opt_conv2d_0 = self.conv2d_0(inp)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_maxpool2d_2 = self.pad_maxpool2d_2(opt_relu_1)
        opt_maxpool2d_2 = self.maxpool2d_2(opt_maxpool2d_2)
        module18_0_opt = self.module18_0(opt_maxpool2d_2)
        module0_0_opt = self.module0_0(opt_maxpool2d_2, module18_0_opt)
        module25_0_opt = self.module25_0(module0_0_opt)
        module18_1_opt = self.module18_1(module25_0_opt)
        module36_0_opt = self.module36_0(module25_0_opt, module18_1_opt)
        module13_0_opt = self.module13_0(module36_0_opt)
        module36_1_opt = self.module36_1(module13_0_opt, module36_0_opt)
        module25_1_opt = self.module25_1(module36_1_opt)
        opt_avgpool2d_119 = self.avgpool2d_119(module25_1_opt)
        opt_flatten_120 = self.flatten_120(opt_avgpool2d_119)
        opt_concat_121 = self.concat_121((opt_flatten_120, x0, x1, x2, x3))
        module29_0_opt = self.module29_0(opt_concat_121)
        opt_dense_124 = self.dense_124(module29_0_opt)
        opt_add_127 = P.Add()(opt_dense_124, x1)
        opt_dense_125 = self.dense_125(module29_0_opt)
        opt_add_128 = P.Add()(opt_dense_125, x2)
        opt_dense_126 = self.dense_126(module29_0_opt)
        opt_add_129 = P.Add()(opt_dense_126, x3)
        module35_0_opt = self.module35_0(opt_flatten_120, x0, opt_add_127, opt_add_128, opt_add_129)
        opt_dense_133 = self.dense_133(module35_0_opt)
        opt_add_136 = P.Add()(opt_dense_133, opt_add_127)
        opt_dense_134 = self.dense_134(module35_0_opt)
        opt_add_137 = P.Add()(opt_dense_134, opt_add_128)
        opt_dense_135 = self.dense_135(module35_0_opt)
        opt_add_138 = P.Add()(opt_dense_135, opt_add_129)
        module35_1_opt = self.module35_1(opt_flatten_120, x0, opt_add_136, opt_add_137, opt_add_138)
        opt_dense_142 = self.dense_142(module35_1_opt)
        opt_add_145 = P.Add()(opt_dense_142, opt_add_136)
        opt_dense_143 = self.dense_143(module35_1_opt)
        opt_add_146 = P.Add()(opt_dense_143, opt_add_137)
        opt_dense_144 = self.dense_144(module35_1_opt)
        opt_add_147 = P.Add()(opt_dense_144, opt_add_138)
        return opt_add_145, opt_add_146, opt_add_147
