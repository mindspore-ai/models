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
"""projection"""

from mindspore import nn


class Bottleneck(nn.Cell):
    """bottleneck"""
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, pad_mode='pad', padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, inplanes, kernel_size=3, stride=stride, pad_mode='pad', padding=1)
        self.bn2 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU()

    def construct(self, x):
        """construct"""
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += residual
        out = self.relu(out)

        return out


class Feature2Structure(nn.Cell):
    """Feature2Structure"""
    def __init__(self, inplanes=64, planes=16):
        super().__init__()

        self.structure_resolver = Bottleneck(inplanes, planes)
        self.out_layer = nn.SequentialCell(
            nn.Conv2d(64, 1, 1, has_bias=True),
            nn.Sigmoid()
        )

    def construct(self, structure_feature):
        """construct"""
        x = self.structure_resolver(structure_feature)
        structure = self.out_layer(x)
        return structure


class Feature2Texture(nn.Cell):
    """Feature2Texture"""
    def __init__(self, inplanes=64, planes=16):
        super().__init__()

        self.texture_resolver = Bottleneck(inplanes, planes)
        self.out_layer = nn.SequentialCell(
            nn.Conv2d(64, 3, 1, has_bias=True),
            nn.Tanh()
        )

    def construct(self, texture_feature):
        """construct"""
        x = self.texture_resolver(texture_feature)
        texture = self.out_layer(x)
        return texture
