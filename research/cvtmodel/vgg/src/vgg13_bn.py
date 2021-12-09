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


class Module12(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels):
        super(Module12, self).__init__()
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


class Module19(nn.Cell):
    def __init__(self, module12_0_conv2d_0_in_channels, module12_0_conv2d_0_out_channels,
                 module12_1_conv2d_0_in_channels, module12_1_conv2d_0_out_channels, module12_2_conv2d_0_in_channels,
                 module12_2_conv2d_0_out_channels):
        super(Module19, self).__init__()
        self.module12_0 = Module12(conv2d_0_in_channels=module12_0_conv2d_0_in_channels,
                                   conv2d_0_out_channels=module12_0_conv2d_0_out_channels)
        self.pad_maxpool2d_0 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.maxpool2d_0 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.module12_1 = Module12(conv2d_0_in_channels=module12_1_conv2d_0_in_channels,
                                   conv2d_0_out_channels=module12_1_conv2d_0_out_channels)
        self.module12_2 = Module12(conv2d_0_in_channels=module12_2_conv2d_0_in_channels,
                                   conv2d_0_out_channels=module12_2_conv2d_0_out_channels)
        self.pad_maxpool2d_1 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.maxpool2d_1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    def construct(self, x):
        module12_0_opt = self.module12_0(x)
        opt_maxpool2d_0 = self.pad_maxpool2d_0(module12_0_opt)
        opt_maxpool2d_0 = self.maxpool2d_0(opt_maxpool2d_0)
        module12_1_opt = self.module12_1(opt_maxpool2d_0)
        module12_2_opt = self.module12_2(module12_1_opt)
        opt_maxpool2d_1 = self.pad_maxpool2d_1(module12_2_opt)
        opt_maxpool2d_1 = self.maxpool2d_1(opt_maxpool2d_1)
        return opt_maxpool2d_1


class Module1(nn.Cell):
    def __init__(self):
        super(Module1, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=128,
                                  out_channels=256,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()
        self.conv2d_2 = nn.Conv2d(in_channels=256,
                                  out_channels=256,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_3 = nn.ReLU()
        self.pad_maxpool2d_4 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.maxpool2d_4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2d_5 = nn.Conv2d(in_channels=256,
                                  out_channels=512,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_6 = nn.ReLU()

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_conv2d_2 = self.conv2d_2(opt_relu_1)
        opt_relu_3 = self.relu_3(opt_conv2d_2)
        opt_maxpool2d_4 = self.pad_maxpool2d_4(opt_relu_3)
        opt_maxpool2d_4 = self.maxpool2d_4(opt_maxpool2d_4)
        opt_conv2d_5 = self.conv2d_5(opt_maxpool2d_4)
        opt_relu_6 = self.relu_6(opt_conv2d_5)
        return opt_relu_6


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
        self.module19_0 = Module19(module12_0_conv2d_0_in_channels=64,
                                   module12_0_conv2d_0_out_channels=64,
                                   module12_1_conv2d_0_in_channels=64,
                                   module12_1_conv2d_0_out_channels=128,
                                   module12_2_conv2d_0_in_channels=128,
                                   module12_2_conv2d_0_out_channels=128)
        self.module1_0 = Module1()
        self.module19_1 = Module19(module12_0_conv2d_0_in_channels=512,
                                   module12_0_conv2d_0_out_channels=512,
                                   module12_1_conv2d_0_in_channels=512,
                                   module12_1_conv2d_0_out_channels=512,
                                   module12_2_conv2d_0_in_channels=512,
                                   module12_2_conv2d_0_out_channels=512)
        self.pad_avgpool2d_25 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_25 = nn.AvgPool2d(kernel_size=(1, 1), stride=(1, 1))
        self.flatten_26 = nn.Flatten()
        self.dense_27 = nn.Dense(in_channels=25088, out_channels=4096, has_bias=True)
        self.relu_28 = nn.ReLU()
        self.dense_29 = nn.Dense(in_channels=4096, out_channels=4096, has_bias=True)
        self.relu_30 = nn.ReLU()
        self.dense_31 = nn.Dense(in_channels=4096, out_channels=1000, has_bias=True)

    def construct(self, input_1):
        opt_conv2d_0 = self.conv2d_0(input_1)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        module19_0_opt = self.module19_0(opt_relu_1)
        module1_0_opt = self.module1_0(module19_0_opt)
        module19_1_opt = self.module19_1(module1_0_opt)
        opt_avgpool2d_25 = self.pad_avgpool2d_25(module19_1_opt)
        opt_avgpool2d_25 = self.avgpool2d_25(opt_avgpool2d_25)
        opt_flatten_26 = self.flatten_26(opt_avgpool2d_25)
        opt_dense_27 = self.dense_27(opt_flatten_26)
        opt_relu_28 = self.relu_28(opt_dense_27)
        opt_dense_29 = self.dense_29(opt_relu_28)
        opt_relu_30 = self.relu_30(opt_dense_29)
        opt_dense_31 = self.dense_31(opt_relu_30)
        return opt_dense_31
