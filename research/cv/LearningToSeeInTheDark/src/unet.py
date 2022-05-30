"""
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
"""


import mindspore.nn as nn
from mindspore import ops


class DownBlock(nn.Cell):
    """down size block"""
    def __init__(self, in_channels, out_channels):
        """init"""
        super(DownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), has_bias=True, data_format="NCHW")
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), has_bias=True, data_format="NCHW")
        self.lrelu = nn.LeakyReLU(alpha=0.1)
        self.pool = nn.MaxPool2d((2, 2), stride=2, pad_mode="same")

    def construct(self, x):
        """forward"""
        conv1 = self.lrelu(self.conv1(x))
        conv2 = self.lrelu(self.conv2(conv1))
        pool = self.pool(conv2)

        return pool


class UpBlock(nn.Cell):
    """up size block"""
    def __init__(self, in_channels, out_channels):
        """init"""
        super(UpBlock, self).__init__()
        self.convt = nn.Conv2dTranspose(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.concat = ops.Concat(axis=1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), has_bias=True, data_format="NCHW")
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), has_bias=True, data_format="NCHW")
        self.lrelu = nn.LeakyReLU(alpha=0.1)

    def construct(self, low_feature, add_feature):
        """forward"""
        x = self.convt(low_feature)
        mix = self.concat((x, add_feature))
        conv1 = self.lrelu(self.conv1(mix))
        conv2 = self.lrelu(self.conv2(conv1))

        return conv2


class UNet(nn.Cell):
    """whole u-net"""
    def __init__(self):
        """init"""
        super(UNet, self).__init__()
        self.db1 = DownBlock(4, 32)
        self.db2 = DownBlock(32, 64)
        self.db3 = DownBlock(64, 128)
        self.db4 = DownBlock(128, 256)

        self.conv1 = nn.Conv2d(256, 512, kernel_size=(3, 3), has_bias=True, data_format="NCHW")
        self.conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), has_bias=True, data_format="NCHW")
        self.lrelu = nn.LeakyReLU(alpha=0.2)
        self.pool = nn.MaxPool2d((2, 2), stride=2, pad_mode="same")

        self.up6 = UpBlock(512, 256)
        self.up7 = UpBlock(256, 128)
        self.up8 = UpBlock(128, 64)
        self.up9 = UpBlock(64, 32)

        self.convt1 = nn.Conv2dTranspose(32, 32, kernel_size=2, stride=2)
        self.convt2 = nn.Conv2dTranspose(32, 32, kernel_size=2, stride=2)
        self.conv_out = nn.Conv2d(32, 3, kernel_size=1, has_bias=True)

    def construct(self, x):
        """forward"""
        pool1 = self.db1(x)
        pool2 = self.db2(pool1)
        pool3 = self.db3(pool2)
        pool4 = self.db4(pool3)

        conv1 = self.lrelu(self.conv1(pool4))
        conv2 = self.lrelu(self.conv2(conv1))
        pool5 = self.pool(conv2)

        up6 = self.up6(pool5, pool4)
        up7 = self.up7(up6, pool3)
        up8 = self.up8(up7, pool2)
        up9 = self.up9(up8, pool1)

        up10 = self.convt1(up9)
        up10 = self.convt1(up10)
        out = self.conv_out(up10)

        return out
