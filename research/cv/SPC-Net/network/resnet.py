# Copyright 2023 Huawei Technologies Co., Ltd
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

import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from network.kaiming_normal import kaiming_normal


class Bottleneck(nn.Cell):
    """
    Bottleneck Layer for Resnet
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, iw=0):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, pad_mode='pad')
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

        self.iw = iw
        self.relu = nn.ReLU()

    def construct(self, x_tuple):
        if len(x_tuple) == 2:
            w_arr = x_tuple[1]
            x = x_tuple[0]
        else:
            print("error!!!")
            return []

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

        return [out, w_arr]


class Bottleneck4(nn.Cell):
    """
    Bottleneck Layer for Resnet
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, iw=0):
        super(Bottleneck4, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=2, dilation=2, pad_mode='pad')
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

        self.iw = iw
        self.relu = nn.ReLU()

    def construct(self, x_tuple):
        if len(x_tuple) == 2:
            w_arr = x_tuple[1]
            x = x_tuple[0]
        else:
            print("error!!!")
            return []

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

        return [out, w_arr]


class ResNet(nn.Cell):
    """
    Resnet Global Module for Initialization
    """
    def __init__(self, block, layers, wt_layer=None, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3,
            has_bias=False, pad_mode='pad'
        )   # bias --> has_bias
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pad = nn.Pad(paddings=((0, 0), (0, 0), (1, 0), (1, 0)), mode="CONSTANT")
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid")
        self.layer1 = self._make_layer(block, 64, layers[0], wt_layer=wt_layer[3])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, wt_layer=wt_layer[4])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, wt_layer=wt_layer[5])
        self.layer4 = self._make_layer_4(Bottleneck4, 512, layers[3], stride=2, wt_layer=wt_layer[6])
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Dense(512 * block.expansion, num_classes)
        self.wt_layer = wt_layer
        self._initialize_weights()

    def construct(self, x):
        w_arr = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pad(x)
        x = self.maxpool(x)
        x_tuple = self.layer1([x, w_arr])  # 400
        x_tuple = self.layer2(x_tuple)  # 100
        x_tuple = self.layer3(x_tuple)  # 100
        x_tuple = self.layer4(x_tuple)  # 100
        x = x_tuple[0]
        w_arr = x_tuple[1]
        return x

    def _initialize_weights(self):
        self.init_parameters_data()
        for _, m in self.cells_and_names():
            if isinstance(m, (nn.Conv2d)):
                m.weight.set_data(Tensor(kaiming_normal(m.weight.data.shape, mode="fan_out", nonlinearity='relu')))
                if m.bias is not None:
                    m.bias.set_data(Tensor(np.zeros(m.bias.data.shape, dtype="float32")))
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma.set_data(Tensor(np.ones(m.gamma.data.shape, dtype="float32")))
                m.beta.set_data(Tensor(np.zeros(m.beta.data.shape, dtype="float32")))
            elif isinstance(m, nn.Dense):
                m.weight.set_data(Tensor(np.random.normal(0, 0.01, m.weight.data.shape).astype("float32")))
                if m.bias is not None:
                    m.bias.set_data(Tensor(np.zeros(m.bias.data.shape, dtype="float32")))

    def _make_layer(self, block, planes, blocks, stride=1, wt_layer=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, has_bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, iw=0))
        self.inplanes = planes * block.expansion
        for index in range(1, blocks):
            layers.append(block(
                self.inplanes, planes,
                iw=0 if (wt_layer > 0 and index < blocks - 1) else wt_layer
            ))
        return nn.SequentialCell(*layers)

    def _make_layer_4(self, block, planes, blocks, stride=1, wt_layer=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, has_bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, iw=0))
        self.inplanes = planes * block.expansion
        for index in range(1, blocks):
            layers.append(block(
                self.inplanes, planes,
                iw=0 if (wt_layer > 0 and index < blocks - 1) else wt_layer
            ))
        return nn.SequentialCell(*layers)


def resnet50(pretrained=False, wt_layer=None, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if wt_layer is None:
        wt_layer = [0, 0, 0, 0, 0, 0, 0]
    model = ResNet(Bottleneck, [3, 4, 6, 3], wt_layer=wt_layer, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model
