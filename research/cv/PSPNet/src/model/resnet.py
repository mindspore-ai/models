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
""" THE Pretrained model ResNet """
import mindspore.nn as nn


def conv3x3(in_channels, out_channels, stride=1, dilation=1):
    """ 3x3 convolution """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, dilation=dilation, pad_mode="pad",
                     padding=1, has_bias=False)


class BasicBlock(nn.Cell):
    """ basic Block for resnet """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, down_sample_layer=None, BatchNorm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.BatchNorm_layer = BatchNorm_layer
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = self.BatchNorm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = self.BatchNorm_layer(planes)
        self.down_sample_layer = down_sample_layer
        self.stride = stride

    def construct(self, x):
        """ process """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample_layer is not None:
            residual = self.down_sample_layer(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Cell):
    """ bottleneck for ResNet """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, down_sample_layer=None, PSP=0, BatchNorm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.BatchNorm_layer = BatchNorm_layer
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, has_bias=False)
        self.bn1 = self.BatchNorm_layer(planes)
        if PSP == 1:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, pad_mode="pad", padding=2, has_bias=False,
                                   dilation=2)
        elif PSP == 2:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, pad_mode="pad", padding=4, has_bias=False,
                                   dilation=4)

        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, has_bias=False, pad_mode="pad")

        self.bn2 = self.BatchNorm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, has_bias=False)
        self.bn3 = self.BatchNorm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.down_sample_layer = down_sample_layer
        self.stride = stride

    def construct(self, x):
        """ process """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample_layer is not None:
            residual = self.down_sample_layer(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Cell):
    """ ResNet """

    def __init__(self, block, layers, deep_base=False, BatchNorm_layer=nn.BatchNorm2d):
        super(ResNet, self).__init__()
        self.deep_base = deep_base
        self.BatchNorm_layer = BatchNorm_layer
        if not self.deep_base:
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, has_bias=False, pad_mode="pad")
            self.bn1 = self.BatchNorm_layer(64)
        else:
            self.inplanes = 128
            self.conv1 = conv3x3(3, 64, stride=2)
            self.bn1 = self.BatchNorm_layer(64)
            self.conv2 = conv3x3(64, 64)
            self.bn2 = self.BatchNorm_layer(64)
            self.conv3 = conv3x3(64, 128)
            self.bn3 = self.BatchNorm_layer(128)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.layer1 = self._make_layer(block, 64, layers[0], PSP=0)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, PSP=0)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, PSP=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, PSP=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

    def _make_layer(self, block, planes, blocks, PSP, stride=1):
        """ make ResNet layer """
        down_sample_layer = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if PSP == 0:
                down_sample_layer = nn.SequentialCell(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, has_bias=False),
                    self.BatchNorm_layer(planes * block.expansion),
                )
            else:
                down_sample_layer = nn.SequentialCell(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=1, has_bias=False),
                    self.BatchNorm_layer(planes * block.expansion),
                )

        layers = [block(self.inplanes, planes, stride, down_sample_layer, PSP=PSP)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, PSP=PSP))

        return nn.SequentialCell(*layers)

    def construct(self, x):
        """ ResNet process """
        x = self.relu(self.bn1(self.conv1(x)))
        if self.deep_base:
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.

    Args:
        If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model
