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


class Module3(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels):
        super(Module3, self).__init__()
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


class Module5(nn.Cell):
    def __init__(self, module3_0_conv2d_0_in_channels, module3_0_conv2d_0_out_channels):
        super(Module5, self).__init__()
        self.module3_0 = Module3(conv2d_0_in_channels=module3_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module3_0_conv2d_0_out_channels)
        self.pad_maxpool2d_0 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.maxpool2d_0 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    def construct(self, x):
        module3_0_opt = self.module3_0(x)
        opt_maxpool2d_0 = self.pad_maxpool2d_0(module3_0_opt)
        opt_maxpool2d_0 = self.maxpool2d_0(opt_maxpool2d_0)
        return opt_maxpool2d_0


class Module2(nn.Cell):
    def __init__(self, conv2d_1_in_channels, conv2d_1_out_channels, conv2d_3_in_channels, conv2d_3_out_channels,
                 conv2d_5_in_channels, conv2d_5_out_channels):
        super(Module2, self).__init__()
        self.pad_maxpool2d_0 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.maxpool2d_0 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2d_1 = nn.Conv2d(in_channels=conv2d_1_in_channels,
                                  out_channels=conv2d_1_out_channels,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_2 = nn.ReLU()
        self.conv2d_3 = nn.Conv2d(in_channels=conv2d_3_in_channels,
                                  out_channels=conv2d_3_out_channels,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
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

    def construct(self, x):
        opt_maxpool2d_0 = self.pad_maxpool2d_0(x)
        opt_maxpool2d_0 = self.maxpool2d_0(opt_maxpool2d_0)
        opt_conv2d_1 = self.conv2d_1(opt_maxpool2d_0)
        opt_relu_2 = self.relu_2(opt_conv2d_1)
        opt_conv2d_3 = self.conv2d_3(opt_relu_2)
        opt_relu_4 = self.relu_4(opt_conv2d_3)
        opt_conv2d_5 = self.conv2d_5(opt_relu_4)
        opt_relu_6 = self.relu_6(opt_conv2d_5)
        return opt_relu_6


class Module8(nn.Cell):
    def __init__(self, module3_0_conv2d_0_in_channels, module3_0_conv2d_0_out_channels, module2_0_conv2d_1_in_channels,
                 module2_0_conv2d_1_out_channels, module2_0_conv2d_3_in_channels, module2_0_conv2d_3_out_channels,
                 module2_0_conv2d_5_in_channels, module2_0_conv2d_5_out_channels):
        super(Module8, self).__init__()
        self.module3_0 = Module3(conv2d_0_in_channels=module3_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module3_0_conv2d_0_out_channels)
        self.module2_0 = Module2(conv2d_1_in_channels=module2_0_conv2d_1_in_channels,
                                 conv2d_1_out_channels=module2_0_conv2d_1_out_channels,
                                 conv2d_3_in_channels=module2_0_conv2d_3_in_channels,
                                 conv2d_3_out_channels=module2_0_conv2d_3_out_channels,
                                 conv2d_5_in_channels=module2_0_conv2d_5_in_channels,
                                 conv2d_5_out_channels=module2_0_conv2d_5_out_channels)

    def construct(self, x):
        module3_0_opt = self.module3_0(x)
        module2_0_opt = self.module2_0(module3_0_opt)
        return module2_0_opt


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
        self.module5_0 = Module5(module3_0_conv2d_0_in_channels=64, module3_0_conv2d_0_out_channels=64)
        self.module3_0 = Module3(conv2d_0_in_channels=64, conv2d_0_out_channels=128)
        self.module8_0 = Module8(module3_0_conv2d_0_in_channels=128,
                                 module3_0_conv2d_0_out_channels=128,
                                 module2_0_conv2d_1_in_channels=128,
                                 module2_0_conv2d_1_out_channels=256,
                                 module2_0_conv2d_3_in_channels=256,
                                 module2_0_conv2d_3_out_channels=256,
                                 module2_0_conv2d_5_in_channels=256,
                                 module2_0_conv2d_5_out_channels=256)
        self.module8_1 = Module8(module3_0_conv2d_0_in_channels=256,
                                 module3_0_conv2d_0_out_channels=256,
                                 module2_0_conv2d_1_in_channels=256,
                                 module2_0_conv2d_1_out_channels=512,
                                 module2_0_conv2d_3_in_channels=512,
                                 module2_0_conv2d_3_out_channels=512,
                                 module2_0_conv2d_5_in_channels=512,
                                 module2_0_conv2d_5_out_channels=512)
        self.module8_2 = Module8(module3_0_conv2d_0_in_channels=512,
                                 module3_0_conv2d_0_out_channels=512,
                                 module2_0_conv2d_1_in_channels=512,
                                 module2_0_conv2d_1_out_channels=512,
                                 module2_0_conv2d_3_in_channels=512,
                                 module2_0_conv2d_3_out_channels=512,
                                 module2_0_conv2d_5_in_channels=512,
                                 module2_0_conv2d_5_out_channels=512)
        self.module5_1 = Module5(module3_0_conv2d_0_in_channels=512, module3_0_conv2d_0_out_channels=512)
        self.pad_avgpool2d_37 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_37 = nn.AvgPool2d(kernel_size=(1, 1), stride=(1, 1))
        self.flatten_38 = nn.Flatten()
        self.dense_39 = nn.Dense(in_channels=25088, out_channels=4096, has_bias=True)
        self.relu_40 = nn.ReLU()
        self.dense_41 = nn.Dense(in_channels=4096, out_channels=4096, has_bias=True)
        self.relu_42 = nn.ReLU()
        self.dense_43 = nn.Dense(in_channels=4096, out_channels=1000, has_bias=True)

    def construct(self, input_1):
        opt_conv2d_0 = self.conv2d_0(input_1)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        module5_0_opt = self.module5_0(opt_relu_1)
        module3_0_opt = self.module3_0(module5_0_opt)
        module8_0_opt = self.module8_0(module3_0_opt)
        module8_1_opt = self.module8_1(module8_0_opt)
        module8_2_opt = self.module8_2(module8_1_opt)
        module5_1_opt = self.module5_1(module8_2_opt)
        opt_avgpool2d_37 = self.pad_avgpool2d_37(module5_1_opt)
        opt_avgpool2d_37 = self.avgpool2d_37(opt_avgpool2d_37)
        opt_flatten_38 = self.flatten_38(opt_avgpool2d_37)
        opt_dense_39 = self.dense_39(opt_flatten_38)
        opt_relu_40 = self.relu_40(opt_dense_39)
        opt_dense_41 = self.dense_41(opt_relu_40)
        opt_relu_42 = self.relu_42(opt_dense_41)
        opt_dense_43 = self.dense_43(opt_relu_42)
        return opt_dense_43
