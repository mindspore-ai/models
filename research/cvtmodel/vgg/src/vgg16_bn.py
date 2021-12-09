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

from mindspore import nn


class Module4(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels):
        super(Module4, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        return opt_relu_1


class Module7(nn.Cell):
    def __init__(self, module4_0_conv2d_0_in_channels, module4_0_conv2d_0_out_channels):
        super(Module7, self).__init__()
        self.module4_0 = Module4(conv2d_0_in_channels=module4_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module4_0_conv2d_0_out_channels)
        self.pad_maxpool2d_0 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.maxpool2d_0 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    def construct(self, x):
        module4_0_opt = self.module4_0(x)
        opt_maxpool2d_0 = self.pad_maxpool2d_0(module4_0_opt)
        opt_maxpool2d_0 = self.maxpool2d_0(opt_maxpool2d_0)
        return opt_maxpool2d_0


class Module0(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_2_in_channels, conv2d_2_out_channels,
                 conv2d_4_in_channels, conv2d_4_out_channels):
        super(Module0, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
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
                                  group=1,
                                  has_bias=True)
        self.relu_3 = nn.ReLU()
        self.conv2d_4 = nn.Conv2d(in_channels=conv2d_4_in_channels,
                                  out_channels=conv2d_4_out_channels,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_5 = nn.ReLU()

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_conv2d_2 = self.conv2d_2(opt_relu_1)
        opt_relu_3 = self.relu_3(opt_conv2d_2)
        opt_conv2d_4 = self.conv2d_4(opt_relu_3)
        opt_relu_5 = self.relu_5(opt_conv2d_4)
        return opt_relu_5


class Module10(nn.Cell):
    def __init__(self, module0_0_conv2d_0_in_channels, module0_0_conv2d_0_out_channels, module0_0_conv2d_2_in_channels,
                 module0_0_conv2d_2_out_channels, module0_0_conv2d_4_in_channels, module0_0_conv2d_4_out_channels):
        super(Module10, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=module0_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_0_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_0_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_0_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_0_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_0_conv2d_4_out_channels)
        self.pad_maxpool2d_0 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.maxpool2d_0 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        opt_maxpool2d_0 = self.pad_maxpool2d_0(module0_0_opt)
        opt_maxpool2d_0 = self.maxpool2d_0(opt_maxpool2d_0)
        return opt_maxpool2d_0


class MindSporeModel(nn.Cell):
    def __init__(self):
        super(MindSporeModel, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=3,
                                  out_channels=64,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()
        self.module7_0 = Module7(module4_0_conv2d_0_in_channels=64, module4_0_conv2d_0_out_channels=64)
        self.module4_0 = Module4(conv2d_0_in_channels=64, conv2d_0_out_channels=128)
        self.module7_1 = Module7(module4_0_conv2d_0_in_channels=128, module4_0_conv2d_0_out_channels=128)
        self.module10_0 = Module10(module0_0_conv2d_0_in_channels=128,
                                   module0_0_conv2d_0_out_channels=256,
                                   module0_0_conv2d_2_in_channels=256,
                                   module0_0_conv2d_2_out_channels=256,
                                   module0_0_conv2d_4_in_channels=256,
                                   module0_0_conv2d_4_out_channels=256)
        self.module10_1 = Module10(module0_0_conv2d_0_in_channels=256,
                                   module0_0_conv2d_0_out_channels=512,
                                   module0_0_conv2d_2_in_channels=512,
                                   module0_0_conv2d_2_out_channels=512,
                                   module0_0_conv2d_4_in_channels=512,
                                   module0_0_conv2d_4_out_channels=512)
        self.module10_2 = Module10(module0_0_conv2d_0_in_channels=512,
                                   module0_0_conv2d_0_out_channels=512,
                                   module0_0_conv2d_2_in_channels=512,
                                   module0_0_conv2d_2_out_channels=512,
                                   module0_0_conv2d_4_in_channels=512,
                                   module0_0_conv2d_4_out_channels=512)
        self.pad_avgpool2d_31 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_31 = nn.AvgPool2d(kernel_size=(1, 1), stride=(1, 1))
        self.flatten_32 = nn.Flatten()
        self.dense_33 = nn.Dense(in_channels=25088, out_channels=4096, has_bias=True)
        self.relu_34 = nn.ReLU()
        self.dense_35 = nn.Dense(in_channels=4096, out_channels=4096, has_bias=True)
        self.relu_36 = nn.ReLU()
        self.dense_37 = nn.Dense(in_channels=4096, out_channels=1000, has_bias=True)

    def construct(self, input_1):
        opt_conv2d_0 = self.conv2d_0(input_1)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        module7_0_opt = self.module7_0(opt_relu_1)
        module4_0_opt = self.module4_0(module7_0_opt)
        module7_1_opt = self.module7_1(module4_0_opt)
        module10_0_opt = self.module10_0(module7_1_opt)
        module10_1_opt = self.module10_1(module10_0_opt)
        module10_2_opt = self.module10_2(module10_1_opt)
        opt_avgpool2d_31 = self.pad_avgpool2d_31(module10_2_opt)
        opt_avgpool2d_31 = self.avgpool2d_31(opt_avgpool2d_31)
        opt_flatten_32 = self.flatten_32(opt_avgpool2d_31)
        opt_dense_33 = self.dense_33(opt_flatten_32)
        opt_relu_34 = self.relu_34(opt_dense_33)
        opt_dense_35 = self.dense_35(opt_relu_34)
        opt_relu_36 = self.relu_36(opt_dense_35)
        opt_dense_37 = self.dense_37(opt_relu_36)
        return opt_dense_37
