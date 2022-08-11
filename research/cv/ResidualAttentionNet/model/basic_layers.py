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
# ============================================================================
import mindspore.nn as nn
from src.conv2d_ops import _conv1x1_valid, _conv3x3_pad

class ResidualBlock(nn.Cell):
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(num_features=input_channels, momentum=0.9)
        self.relu = nn.ReLU()
        self.conv1 = _conv1x1_valid(in_channels=input_channels, out_channels=int(output_channels / 4), stride=1)
        self.bn2 = nn.BatchNorm2d(num_features=int(output_channels / 4), momentum=0.9)
        self.relu = nn.ReLU()
        self.conv2 = _conv3x3_pad(in_channels=int(output_channels / 4),
                                  out_channels=int(output_channels / 4), stride=stride)
        self.bn3 = nn.BatchNorm2d(num_features=int(output_channels / 4), momentum=0.9)
        self.relu = nn.ReLU()
        self.conv3 = _conv1x1_valid(in_channels=int(output_channels / 4), out_channels=output_channels, stride=1)
        self.conv4 = _conv1x1_valid(in_channels=input_channels, out_channels=output_channels, stride=stride)
    def construct(self, x):
        residual = x
        out = self.bn1(x)
        out1 = self.relu(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if (self.input_channels != self.output_channels) or (self.stride != 1):
            residual = self.conv4(out1)
        out += residual
        return out
