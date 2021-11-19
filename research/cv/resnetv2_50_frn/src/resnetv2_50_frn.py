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
"""ResNetV2_50_FRN model definition"""

import numpy as np

import mindspore
import mindspore.nn as nn
import mindspore.ops.operations as P

from mindspore.common.tensor import Tensor
from mindspore import Parameter

class FilterResponseNormLayer(nn.Cell):
    """
     the implementation of Filter Response Normalization Layer (FRN)
     """
    def __init__(self, num_features, eps=1e-6, method=2):
        super(FilterResponseNormLayer, self).__init__()
        self.num_features = num_features

        self.tau = Parameter(Tensor(np.zeros([1, num_features, 1, 1]), mindspore.float32))
        self.beta = Parameter(Tensor(np.zeros([1, num_features, 1, 1]), mindspore.float32))
        self.gamma = Parameter(Tensor(np.ones([1, num_features, 1, 1]), mindspore.float32))
        self.eps = Parameter(Tensor([eps], mindspore.float32))

        self.reset_parameters()

        self.mean = mindspore.ops.operations.ReduceMean(keep_dims=True)
        self.abs = P.Abs()
        self.sqrt = mindspore.ops.Sqrt()
        self.max = mindspore.ops.Maximum()
        self.relu = mindspore.ops.ReLU()

        self.method = method

    def reset_parameters(self):
        self.tau = Parameter(Tensor(np.zeros([1, self.num_features, 1, 1]), mindspore.float32))
        self.beta = Parameter(Tensor(np.zeros([1, self.num_features, 1, 1]), mindspore.float32))
        self.gamma = Parameter(Tensor(np.ones([1, self.num_features, 1, 1]), mindspore.float32))

    def construct(self, input_data):
        """ construct network """
        nu2 = self.mean(input_data ** 2, (2, 3))
        x = input_data / self.sqrt(nu2 + self.abs(self.eps))
        x = self.gamma * x + self.beta

        #method 1
        if self.method == 1:
            return self.max(x, self.tau)

        #method 2
        #default is method 2
        return self.max(x - self.tau, 0) + self.tau

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride,
                     pad_mode='pad', padding=dilation, dilation=dilation, group=groups, has_bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, stride=stride,
                     pad_mode='pad', has_bias=False)

class PreActBottleneckFRN(nn.Cell):
    """
    PreActBottleneckFRN: the bottleneck block for residual path in PreActResNetFRN.
    """
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, method=2):
        super(PreActBottleneckFRN, self).__init__()

        if norm_layer is None:
            norm_layer = FilterResponseNormLayer

        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.frn1 = norm_layer(inplanes, method=method)
        self.conv1 = conv1x1(inplanes, width)
        self.frn2 = norm_layer(width, method=method)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.frn3 = norm_layer(width, method=method)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        """ construct network """
        identity = x

        out = self.frn1(x)
        out = self.conv1(out)

        out = self.frn2(out)
        out = self.conv2(out)

        out = self.frn3(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        return out

class PreActResNetFRN(nn.Cell):
    """
    PreActResNetFRN: ResNetV2 with FRN Layer
    """
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, method=2):
        super(PreActResNetFRN, self).__init__()

        if norm_layer is None:
            norm_layer = FilterResponseNormLayer
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates whether we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.inplanes, kernel_size=7, stride=2,
                               pad_mode='pad', padding=3, has_bias=False)
        self.frn1 = norm_layer(self.inplanes, method=method)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')

        self.layer1 = self._make_layer(block, 64, layers[0], method=method)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], method=method)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], method=method)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], method=method)
        self.frn2 = norm_layer(self.inplanes, method=method)

        self.avgpool = P.ReduceMean(keep_dims=True)

        self.fc = nn.Dense(in_channels=512 * block.expansion, out_channels=num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, method=2):
        """ make one layer of ResNetV2 """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell([
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion, method=method),
            ])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, method=method))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, method=method))

        return nn.SequentialCell([*layers])

    def construct(self, x):
        """ construct network """
        x = self.conv1(x)
        x = self.frn1(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.frn2(x)

        x = self.avgpool(x, (2, 3))
        x = P.Flatten()(x)
        x = self.fc(x)

        return x

def preact_resnet50_frn(**kwargs):
    """
    Filter Response Normalization Layer:
    Eliminating Batch Dependence in the Training of Deep Neural NetworksNingning.
    https://arxiv.org/abs/1911.09737v2
    """
    return PreActResNetFRN(PreActBottleneckFRN, [3, 4, 6, 3], **kwargs)
