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
"""
######################## TCN network ########################
construct TCN network
"""
import mindspore.nn as nn
from mindspore.common.initializer import initializer, Normal

from src.WNConv1d import WNConv1d


class Chomp1d(nn.Cell):
    """chomp1d"""

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def construct(self, x):
        """construct Chomp1d"""
        return x[:, :, :x.shape[2] - self.chomp_size]


class TemporalBlock(nn.Cell):
    """TemporalBlock"""

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        padding1 = padding % 255
        padding2 = padding - padding1
        self.pad = nn.Pad(paddings=((0, 0), (0, 0), (padding2, padding2)), mode="CONSTANT")
        self.conv1 = WNConv1d(dim=0, in_channels=n_inputs, out_channels=n_outputs, kernel_size=kernel_size,
                              stride=stride, pad_mode='pad', padding=padding1, dilation=dilation,
                              weight_init='he_normal')

        self.conv2 = WNConv1d(dim=0, in_channels=n_outputs, out_channels=n_outputs, kernel_size=kernel_size,
                              stride=stride, pad_mode='pad', padding=padding1, dilation=dilation,
                              weight_init='he_normal')

        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weight()

    def init_weight(self):
        if self.downsample is not None:
            self.downsample.weight.set_data(initializer(Normal(0.01, 0), self.downsample.weight.data.shape))

    def construct(self, x):
        """construct temporablock"""
        out = self.pad(x)
        out = self.conv1(out)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.pad(out)
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Cell):
    """TemporalConvNet"""

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        self.network = nn.SequentialCell(*layers)

    def construct(self, x):
        """construct TemporalConvNet"""
        return self.network(x)
