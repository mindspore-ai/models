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


class Module1(nn.Cell):
    def __init__(self, conv2d_1_in_channels, conv2d_1_out_channels, conv2d_3_in_channels, conv2d_3_out_channels):
        super(Module1, self).__init__()
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
        self.pad_maxpool2d_5 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.maxpool2d_5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    def construct(self, x):
        opt_maxpool2d_0 = self.pad_maxpool2d_0(x)
        opt_maxpool2d_0 = self.maxpool2d_0(opt_maxpool2d_0)
        opt_conv2d_1 = self.conv2d_1(opt_maxpool2d_0)
        opt_relu_2 = self.relu_2(opt_conv2d_1)
        opt_conv2d_3 = self.conv2d_3(opt_relu_2)
        opt_relu_4 = self.relu_4(opt_conv2d_3)
        opt_maxpool2d_5 = self.pad_maxpool2d_5(opt_relu_4)
        opt_maxpool2d_5 = self.maxpool2d_5(opt_maxpool2d_5)
        return opt_maxpool2d_5


class Module4(nn.Cell):
    def __init__(self, module3_0_conv2d_0_in_channels, module3_0_conv2d_0_out_channels, module1_0_conv2d_1_in_channels,
                 module1_0_conv2d_1_out_channels, module1_0_conv2d_3_in_channels, module1_0_conv2d_3_out_channels):
        super(Module4, self).__init__()
        self.module3_0 = Module3(conv2d_0_in_channels=module3_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module3_0_conv2d_0_out_channels)
        self.module1_0 = Module1(conv2d_1_in_channels=module1_0_conv2d_1_in_channels,
                                 conv2d_1_out_channels=module1_0_conv2d_1_out_channels,
                                 conv2d_3_in_channels=module1_0_conv2d_3_in_channels,
                                 conv2d_3_out_channels=module1_0_conv2d_3_out_channels)

    def construct(self, x):
        module3_0_opt = self.module3_0(x)
        module1_0_opt = self.module1_0(module3_0_opt)
        return module1_0_opt


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
        self.pad_maxpool2d_2 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.maxpool2d_2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.module4_0 = Module4(module3_0_conv2d_0_in_channels=64,
                                 module3_0_conv2d_0_out_channels=128,
                                 module1_0_conv2d_1_in_channels=128,
                                 module1_0_conv2d_1_out_channels=256,
                                 module1_0_conv2d_3_in_channels=256,
                                 module1_0_conv2d_3_out_channels=256)
        self.module3_0 = Module3(conv2d_0_in_channels=256, conv2d_0_out_channels=512)
        self.module4_1 = Module4(module3_0_conv2d_0_in_channels=512,
                                 module3_0_conv2d_0_out_channels=512,
                                 module1_0_conv2d_1_in_channels=512,
                                 module1_0_conv2d_1_out_channels=512,
                                 module1_0_conv2d_3_in_channels=512,
                                 module1_0_conv2d_3_out_channels=512)
        self.pad_avgpool2d_21 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_21 = nn.AvgPool2d(kernel_size=(1, 1), stride=(1, 1))
        self.flatten_22 = nn.Flatten()
        self.dense_23 = nn.Dense(in_channels=25088, out_channels=4096, has_bias=True)
        self.relu_24 = nn.ReLU()
        self.dense_25 = nn.Dense(in_channels=4096, out_channels=4096, has_bias=True)
        self.relu_26 = nn.ReLU()
        self.dense_27 = nn.Dense(in_channels=4096, out_channels=1000, has_bias=True)

    def construct(self, input_1):
        opt_conv2d_0 = self.conv2d_0(input_1)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_maxpool2d_2 = self.pad_maxpool2d_2(opt_relu_1)
        opt_maxpool2d_2 = self.maxpool2d_2(opt_maxpool2d_2)
        module4_0_opt = self.module4_0(opt_maxpool2d_2)
        module3_0_opt = self.module3_0(module4_0_opt)
        module4_1_opt = self.module4_1(module3_0_opt)
        opt_avgpool2d_21 = self.pad_avgpool2d_21(module4_1_opt)
        opt_avgpool2d_21 = self.avgpool2d_21(opt_avgpool2d_21)
        opt_flatten_22 = self.flatten_22(opt_avgpool2d_21)
        opt_dense_23 = self.dense_23(opt_flatten_22)
        opt_relu_24 = self.relu_24(opt_dense_23)
        opt_dense_25 = self.dense_25(opt_relu_24)
        opt_relu_26 = self.relu_26(opt_dense_25)
        opt_dense_27 = self.dense_27(opt_relu_26)
        return opt_dense_27
