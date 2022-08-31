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

"""Resnet model define"""

import mindspore.nn as nn
from mindspore import load_checkpoint

affine_par = True


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, padding="same", stride=stride, has_bias=False)


class Bottleneck(nn.Cell):
    """
    Bottleneck layer
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, dilation_=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, has_bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par, use_batch_statistics=False)
        for i in self.bn1.get_parameters():
            i.requires_grad = False
        padding = 1
        if dilation_ == 2:
            padding = 2
        elif dilation_ == 4:
            padding = 4
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=padding, pad_mode="pad", has_bias=False,
                               dilation=dilation_)

        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par, use_batch_statistics=False)
        for i in self.bn2.get_parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, has_bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par, use_batch_statistics=False)
        for i in self.bn3.get_parameters():
            i.requires_grad = False
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        """
        forword
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Cell):
    """
    resnet
    """

    def __init__(self, block, layers):
        self.in_planes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, pad_mode="pad",
                               has_bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par, use_batch_statistics=False)
        for i in self.bn1.get_parameters():
            i.requires_grad = False
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        """
        make layer
        """
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.SequentialCell(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, has_bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par, use_batch_statistics=False),
            )
        for i in downsample[1].get_parameters():
            i.requires_grad = False
        layers = [block(self.in_planes, planes, stride, dilation_=dilation, downsample=downsample)]
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, dilation_=dilation))

        return nn.SequentialCell(*layers)

    def load_pretrained_model(self, model_file):
        """
        load pretrained model
        """
        load_checkpoint(model_file, net=self)

    def construct(self, x):
        """
        forward
        """
        tmp_x = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        tmp_x.append(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        tmp_x.append(x)
        x = self.layer2(x)
        tmp_x.append(x)
        x = self.layer3(x)
        tmp_x.append(x)
        x = self.layer4(x)
        tmp_x.append(x)

        return tmp_x


# adding prefix "base" to parameter names for load_checkpoint().
class Tmp(nn.Cell):
    def __init__(self, base):
        super(Tmp, self).__init__()
        self.base = base


def resnet50():
    base = ResNet(Bottleneck, [3, 4, 6, 3])
    return Tmp(base)
