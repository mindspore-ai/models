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
"""melgan model"""
import mindspore.nn as nn
from mindspore.ops import operations as P


class Conv1d_t(nn.Cell):
    """conv1d"""
    def __init__(self, in_channels, out_channels, kernel_size1d, stride=1, pad_mode="valid",
                 padding=0, dilation=1, group=1, has_bias=True, weight_init='xavier_uniform',
                 bias_init='zeros'):
        super(Conv1d_t, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size1d, stride, pad_mode,
                              padding, dilation, group, has_bias, weight_init, bias_init)

    def construct(self, x):
        y = self.conv(x)
        return y


class Conv1d_Transpose(nn.Cell):
    """conv1d transpose"""
    def __init__(self, in_channels, out_channels, kernel_size1d, stride=1, pad_mode="pad",
                 padding=0, dilation=1, group=1, has_bias=True, weight_init='xavier_uniform',
                 bias_init='zeros'):
        super(Conv1d_Transpose, self).__init__()
        self.conv = nn.Conv1dTranspose(in_channels, out_channels, kernel_size1d, stride, pad_mode,
                                       padding, dilation, group, has_bias, weight_init, bias_init)

    def construct(self, x):
        y = self.conv(x)
        return y


class ResStack(nn.Cell):
    """restack"""
    def __init__(self, channel):
        super(ResStack, self).__init__()

        self.block1 = nn.SequentialCell([
            nn.LeakyReLU(),
            nn.Pad(((0, 0), (0, 0), (1, 1))),
            Conv1d_t(channel, channel, kernel_size1d=3, dilation=1),
            nn.LeakyReLU(),
            Conv1d_t(channel, channel, kernel_size1d=1),
        ])
        self.shortcut1 = Conv1d_t(channel, channel, kernel_size1d=1)

        self.block2 = nn.SequentialCell([
            nn.LeakyReLU(),
            nn.Pad(((0, 0), (0, 0), (3, 3))),
            Conv1d_t(channel, channel, kernel_size1d=3, dilation=3),
            nn.LeakyReLU(),
            Conv1d_t(channel, channel, kernel_size1d=1),
        ])
        self.shortcut2 = Conv1d_t(channel, channel, kernel_size1d=1)

        self.block3 = nn.SequentialCell([
            nn.LeakyReLU(),
            nn.Pad(((0, 0), (0, 0), (9, 9))),
            Conv1d_t(channel, channel, kernel_size1d=3, dilation=9),
            nn.LeakyReLU(),
            Conv1d_t(channel, channel, kernel_size1d=1),
        ])
        self.shortcut3 = Conv1d_t(channel, channel, kernel_size1d=1)
    def construct(self, x):
        x = self.block1(x) + self.shortcut1(x)
        x = self.block2(x) + self.shortcut2(x)
        x = self.block3(x) + self.shortcut3(x)
        return x


class Generator(nn.Cell):
    """generator"""
    def __init__(self, alpha):
        super(Generator, self).__init__()
        self.Pad7 = nn.Pad(((0, 0), (0, 0), (3, 3)))
        self.conv1d_1 = Conv1d_t(80, 512, 7, stride=1)
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.slice = P.Slice()
        self.shape = P.Shape()
        self.conv1dtran_1 = Conv1d_Transpose(512, 256, 16, stride=8, padding=4)

        self.rses_1 = ResStack(256)
        self.leaky_relu1 = nn.LeakyReLU(alpha)

        self.conv1dtran_2 = Conv1d_Transpose(256, 128, 16, stride=8, padding=4)

        self.rses_2 = ResStack(128)
        self.leaky_relu2 = nn.LeakyReLU(alpha)

        self.conv1dtran_3 = Conv1d_Transpose(128, 64, 4, stride=2, padding=1)

        self.rses_3 = ResStack(64)
        self.leaky_relu3 = nn.LeakyReLU()

        self.conv1dtran_4 = Conv1d_Transpose(64, 32, 4, stride=2, padding=1)

        self.rses_4 = ResStack(32)
        self.leaky_relu4 = nn.LeakyReLU()

        self.conv1d_2 = Conv1d_t(32, 1, 7, stride=1)  # 32
        self.tanh = nn.Tanh()

    def construct(self, x):
        """forward network"""
        x = self.Pad7(x)
        x = self.conv1d_1(x)
        x = self.leaky_relu(x)

        x = self.conv1dtran_1(x)
        x = self.rses_1(x)
        x = self.leaky_relu1(x)

        x = self.conv1dtran_2(x)
        x = self.rses_2(x)
        x = self.leaky_relu2(x)

        x = self.conv1dtran_3(x)
        x = self.rses_3(x)
        x = self.leaky_relu3(x)

        x = self.conv1dtran_4(x)
        x = self.rses_4(x)
        x = self.leaky_relu4(x)

        x = self.Pad7(x)
        x = self.conv1d_2(x)
        x = self.tanh(x)
        return x


class Discriminator(nn.Cell):
    """discriminator"""
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.SequentialCell([
            nn.Pad(paddings=((0, 0), (0, 0), (7, 7))),
            Conv1d_t(1, 16, kernel_size1d=15, stride=1),
            nn.LeakyReLU(0.2)])
        self.conv2 = nn.SequentialCell([
            Conv1d_t(16, 64, kernel_size1d=41, stride=4, pad_mode="pad", padding=20),
            nn.LeakyReLU(0.2)])
        self.conv3 = nn.SequentialCell([
            Conv1d_t(64, 256, kernel_size1d=41, stride=4, pad_mode="pad", padding=20),
            nn.LeakyReLU(0.2)])
        self.conv4 = nn.SequentialCell([
            Conv1d_t(256, 1024, kernel_size1d=41, stride=4, pad_mode="pad", padding=20),
            nn.LeakyReLU(0.2)])
        self.conv5 = nn.SequentialCell([
            Conv1d_t(1024, 1024, kernel_size1d=11, stride=4, pad_mode="pad", padding=5),
            nn.LeakyReLU(0.2)])
        self.conv6 = nn.SequentialCell([
            Conv1d_t(1024, 1024, kernel_size1d=3, stride=1, pad_mode="pad", padding=1),
            nn.LeakyReLU(0.2)])
        self.conv7 = Conv1d_t(1024, 1, kernel_size1d=3, stride=1, pad_mode="pad", padding=1)


    def construct(self, x):
        x0 = self.conv1(x)
        x1 = self.conv2(x0)
        x2 = self.conv3(x1)
        x3 = self.conv4(x2)
        x4 = self.conv5(x3)
        x5 = self.conv6(x4)
        x6 = self.conv7(x5)
        return x0, x1, x2, x3, x4, x5, x6


class MultiDiscriminator(nn.Cell):
    """multi discriminator"""
    def __init__(self):
        super(MultiDiscriminator, self).__init__()

        self.dis1 = Discriminator()
        self.dis2 = Discriminator()
        self.dis3 = Discriminator()

        self.avg1 = Conv1d_t(1, 1, kernel_size1d=5, stride=2, pad_mode="pad", padding=2)
        self.avg2 = Conv1d_t(1, 1, kernel_size1d=5, stride=2, pad_mode="pad", padding=2)

    def construct(self, x):
        """forward network"""
        y1 = self.dis1(x)

        input1 = self.avg1(x)

        y2 = self.dis2(input1)

        input2 = self.avg2(input1)

        y3 = self.dis3(input2)

        return y1, y2, y3
