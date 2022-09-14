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
"""3D-LWNet"""
import mindspore.nn as nn
import mindspore.ops as ops


class Bottleneck1(nn.Cell):
    """
    depth wise convolutional bottleneck expansion=4
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck1, self).__init__()
        self.conv1 = nn.Conv3d(in_channel, out_channel, kernel_size=1, has_bias=False)
        self.bn1 = nn.BatchNorm3d(out_channel, momentum=0.1)
        self.conv2 = nn.Conv3d(out_channel, out_channel, kernel_size=3, stride=stride,
                               padding=1, group=out_channel, has_bias=False)
        self.bn2 = nn.BatchNorm3d(out_channel, momentum=0.1)
        self.conv3 = nn.Conv3d(out_channel, out_channel * 4, kernel_size=1, has_bias=False)
        self.bn3 = nn.BatchNorm3d(out_channel * 4, momentum=0.1)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        """
        construct
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck2(nn.Cell):
    """
    depth wise convolutional bottleneck expansion=2
    """
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck2, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, has_bias=False)
        self.bn1 = nn.BatchNorm3d(planes, momentum=0.1)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, group=planes, has_bias=False)
        self.bn2 = nn.BatchNorm3d(planes, momentum=0.1)
        self.conv3 = nn.Conv3d(planes, planes * 2, kernel_size=1, has_bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 2, momentum=0.1)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        """
        construct
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


class Bottleneck3(nn.Cell):
    """
    depth wise convolutional bottleneck expansion=1/4
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck3, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes*4, kernel_size=1, has_bias=False)
        self.bn1 = nn.BatchNorm3d(planes*4, momentum=0.1)
        self.conv2 = nn.Conv3d(planes*4, planes*4, kernel_size=3, stride=stride,
                               padding=1, pad_mode='pad', has_bias=False)
        self.bn2 = nn.BatchNorm3d(planes*4, momentum=0.1)
        self.conv3 = nn.Conv3d(planes*4, planes, kernel_size=1, has_bias=False)
        self.bn3 = nn.BatchNorm3d(planes, momentum=0.1)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.MaxPool3d = ops.MaxPool3D(kernel_size=2, strides=2, ceil_mode=True, pad_mode="pad")

    def construct(self, x):
        """
        construct
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
            residual = self.MaxPool3d(x)
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck4(nn.Cell):
    """
    depth wise convolutional bottleneck expansion=1/2
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck4, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes*2, kernel_size=1, has_bias=False)
        self.bn1 = nn.BatchNorm3d(planes*2, momentum=0.1)
        self.conv2 = nn.Conv3d(planes*2, planes*2, kernel_size=3, stride=stride,
                               padding=1, group=planes, has_bias=False)
        self.bn2 = nn.BatchNorm3d(planes*2, momentum=0.1)
        self.conv3 = nn.Conv3d(planes*2, planes, kernel_size=1, has_bias=False)
        self.bn3 = nn.BatchNorm3d(planes, momentum=0.1)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        """
        construct
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
    3D-LWNet architecture
    """
    def __init__(self, block, layers, num_classes=1000, dropout_keep_prob=0):
        self.inplanes = 32
        super(ResNet, self).__init__()
        self.conv1 = nn.SequentialCell([
            nn.Conv3d(1, 32, kernel_size=(8, 3, 3), stride=1, padding=0, has_bias=False),
            nn.BatchNorm3d(32, momentum=0.1),
            nn.ReLU(),
        ])
        self.MaxPool3d = ops.MaxPool3D(kernel_size=(2, 3, 3), strides=2)

        self.conv2 = nn.SequentialCell([
            self._make_layer(block, 32, layers[0]),
            self._make_layer(block, 64, layers[1], stride=2),
            self._make_layer(block, 128, layers[2], stride=2),
            self._make_layer(block, 256, layers[3], stride=2),
        ])

        self.AdaptiveAvgPool3d = ops.ReduceMean(keep_dims=True)

        self.fc = nn.Dense(256 * block.expansion, num_classes)

        self.log_softmax = nn.LogSoftmax(axis=1)

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        make_layer
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell([
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, has_bias=False),
                nn.BatchNorm3d(planes * block.expansion, momentum=0.1),
            ])
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.SequentialCell(layers)

    def construct(self, x):
        """
        construct
        """
        x = self.conv1(x)
        x = self.MaxPool3d(x)
        x = self.conv2(x)
        x = self.AdaptiveAvgPool3d(x, (2, 3, 4))
        x = x.squeeze(2).squeeze(2).squeeze(2)
        x = self.fc(x)
        x = self.log_softmax(x)

        return x


def LWNet_1(**kwargs):
    model = ResNet(Bottleneck1, [1, 2, 2, 1], **kwargs)
    return model


def LWNet_2(**kwargs):
    model = ResNet(Bottleneck2, [1, 2, 2, 1], **kwargs)
    return model


def LWNet_3(**kwargs):
    model = ResNet(Bottleneck3, [1, 2, 2, 1], **kwargs)
    return model


def LWNet_4(**kwargs):
    model = ResNet(Bottleneck4, [1, 2, 2, 1], **kwargs)
    return model


def dict_lwnet():
    lwnet = {'LWNet_1': LWNet_1, 'LWNet_2': LWNet_2, 'LWNet_3': LWNet_3, 'LWNet_4': LWNet_4}
    return lwnet
