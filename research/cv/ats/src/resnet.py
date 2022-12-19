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

"""cifar resnet."""


import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.ops as ops


def weight_variable_0(shape):
    zeros = np.zeros(shape).astype(np.float32)
    return Tensor(zeros)


def weight_variable_1(shape):
    ones = np.ones(shape).astype(np.float32)
    return Tensor(ones)


def conv3x3(in_channels, out_channels, stride=1, padding=0):
    if padding == 0:
        pad_mode = "same"
    else:
        pad_mode = "pad"

    return nn.Conv2d(
        in_channels, out_channels,
        kernel_size=3, stride=stride, padding=padding,
        weight_init='XavierUniform',
        has_bias=False, pad_mode=pad_mode
    )


def conv1x1(in_channels, out_channels, stride=1, padding=0):
    if padding == 0:
        pad_mode = "same"
    else:
        pad_mode = "pad"

    return nn.Conv2d(
        in_channels, out_channels,
        kernel_size=1, stride=stride, padding=padding,
        weight_init='XavierUniform',
        has_bias=False, pad_mode=pad_mode
    )


def conv7x7(in_channels, out_channels, stride=1, padding=0):
    if padding == 0:
        pad_mode = "same"
    else:
        pad_mode = "pad"

    return nn.Conv2d(
        in_channels, out_channels,
        kernel_size=7, stride=stride, padding=padding,
        weight_init='XavierUniform',
        has_bias=False, pad_mode=pad_mode
    )


def bn_with_initialize(out_channels):
    shape = (out_channels)
    mean = weight_variable_0(shape)
    var = weight_variable_1(shape)
    beta = weight_variable_0(shape)
    bn = nn.BatchNorm2d(
        out_channels, momentum=0.99, eps=0.00001, gamma_init='Uniform',
        beta_init=beta, moving_mean_init=mean, moving_var_init=var
    )
    return bn


def bn_with_initialize_last(out_channels):
    shape = (out_channels)
    mean = weight_variable_0(shape)
    var = weight_variable_1(shape)
    beta = weight_variable_0(shape)
    bn = nn.BatchNorm2d(
        out_channels, momentum=0.99, eps=0.00001, gamma_init='Uniform',
        beta_init=beta, moving_mean_init=mean, moving_var_init=var
    )
    return bn


def fc_with_initialize(input_channels, out_channels):
    return nn.Dense(
        input_channels, out_channels,
        weight_init='XavierUniform', bias_init='Uniform'
    )


def get_resnet_cfg(res_n_layer):
    cfgs = {
        8: [16, 16, 32, 64],
        14: [16, 16, 32, 64],
        20: [16, 16, 32, 64],
        32: [16, 16, 32, 64],
        44: [16, 16, 32, 64],
        56: [16, 16, 32, 64],
        110: [16, 16, 32, 64],
    }
    assert res_n_layer in cfgs, "res_n_layer=8,14,20,32,44,56,110"

    return cfgs[res_n_layer]


class BasicResidualBlock(nn.Cell):
    expansion = 1

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 downsample=None,
                 is_last=False):
        super(BasicResidualBlock, self).__init__()

        out_chls = out_channels // self.expansion
        self.conv1 = conv3x3(in_channels, out_chls, stride=stride)
        self.bn1 = bn_with_initialize(out_chls)

        self.conv2 = conv3x3(out_chls, out_chls, stride=1)
        self.bn2 = bn_with_initialize(out_chls)

        self.relu = ops.ReLU()
        self.add = ops.Add()

        self.is_last = is_last
        self.downsample = downsample
        self.stride = stride

    def construct(self, x0):
        identity = x0

        out = self.conv1(x0)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x0)

        out = self.add(out, identity)
        out = self.relu(out)

        return out

    def forward(self):
        print("Not implemented...", self.stride)
        return self.stride


class CifarResNet(nn.Cell):

    def __init__(self, res_n_layer, n_classes):
        super(CifarResNet, self).__init__()
        assert (res_n_layer - 2) % 6 == 0, '6n+2, e.g. 20, 32, 44, 56, 110'
        n = (res_n_layer - 2) // 6

        block = BasicResidualBlock

        num_filters = get_resnet_cfg(res_n_layer)

        self.inplanes = num_filters[0]
        self.conv1 = conv3x3(
            3, num_filters[0], stride=1, padding=1
        )
        self.bn1 = bn_with_initialize(num_filters[0])
        self.relu = ops.ReLU()
        self.layer1 = self._make_layer(block, num_filters[1], n)
        self.layer2 = self._make_layer(block, num_filters[2], n, stride=2)
        self.layer3 = self._make_layer(block, num_filters[3], n, stride=2)

        self.avgpool = nn.AvgPool2d(8)

        self.fc = fc_with_initialize(
            num_filters[3] * block.expansion, n_classes
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell([
                conv1x1(
                    self.inplanes, planes * block.expansion, stride=stride
                ),
                bn_with_initialize(planes * block.expansion),
            ])

        layers = list([])
        layers.append(
            block(
                self.inplanes, planes, stride,
                downsample, is_last=(blocks == 1)
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes, is_last=(i == blocks - 1)
                )
            )

        return nn.SequentialCell(layers)

    def construct(self, x0):
        f = self.conv1(x0)
        f = self.bn1(f)
        f = self.relu(f)  # 32x32

        f = self.layer1(f)  # 32x32
        f = self.layer2(f)  # 16x16
        f = self.layer3(f)  # 8x8

        f = self.avgpool(f)
        f = f.view(f.shape[0], -1)
        g = self.fc(f)

        return g

    def forward(self):
        print("Not implemented...", self.inplanes)
        return self.inplanes


if __name__ == "__main__":
    model = CifarResNet(res_n_layer=8, n_classes=100)

    for size in [32]:
        for n_layer in [8, 14, 20, 32, 44, 56, 110]:
            x = Tensor(np.random.randn(2, 3, size, size), mindspore.float32)
            model = CifarResNet(res_n_layer=n_layer, n_classes=100)
            feats, logits = model.construct(x)

            n_params = sum([
                np.prod(param.shape) for param in model.get_parameters()
            ])
            print("Total number of parameters : {}".format(
                n_params
            ))

            print(feats.shape, logits.shape)
