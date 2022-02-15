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
"""Split-Attention"""
from mindspore import nn
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.nn import ReLU

__all__ = ['SplAtConv2d']

conv_weight_init = 'HeUniform'

class GroupConv(nn.Cell):
    """
    group convolution operation.
    Args:
        in_channels (int): Input channels of feature map.
        out_channels (int): Output channels of feature map.
        kernel_size (int): Size of convolution kernel.
        stride (int): Stride size for the group convolution layer.
    Returns:
        tensor, output tensor.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, pad_mode="pad", padding=0, dilation=(1, 1), group=1, has_bias=False):
        super(GroupConv, self).__init__()
        assert in_channels % group == 0 and out_channels % group == 0
        self.group = group
        self.convs = nn.CellList()

        self.op_split = P.Split(axis=1, output_num=self.group)
        self.op_concat = P.Concat(axis=1)
        self.cast = P.Cast()

        for _ in range(group):
            self.convs.append(nn.Conv2d(in_channels//group, out_channels//group,
                                        kernel_size=kernel_size, stride=stride, has_bias=has_bias,
                                        padding=padding, pad_mode=pad_mode, group=1, weight_init=conv_weight_init))


    def construct(self, x):

        if self.group > 1:
            features = (x[:, 0:(x.shape[1] // 2), :, :], x[:, (x.shape[1] // 2):, :, :])
        else:
            features = (x,)
        outputs = ()

        for i in range(self.group):
            if len(features[i].shape) < 4:
                print("error")
            outputs = outputs + (self.convs[i](self.cast(features[i], mstype.float32)),)
        out = self.op_concat(outputs)
        return out

class SplAtConv2d(nn.Cell):
    """Split-Attention Conv2d"""
    def __init__(self, in_channels, channels, kernel_size,
                 stride=(1, 1), padding=(0, 0), dilation=(1, 1),
                 groups=1, bias=True, radix=2, reduction_factor=4,
                 rectify=False, rectify_avg=False, norm_layer=None, **kwargs):
        super(SplAtConv2d, self).__init__()
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.conv = GroupConv(in_channels, channels*radix, kernel_size, stride, pad_mode="pad",
                              padding=padding, dilation=dilation, group=groups*radix, has_bias=bias)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels * radix)
        self.relu = ReLU()
        self.fc1 = GroupConv(channels, inter_channels, 1, group=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = GroupConv(inter_channels, channels*radix, 1, group=self.cardinality)
        self.rsoftmax = rSoftMax(radix, groups)
        self.reshape = P.Reshape()
        self.split = P.Split(axis=1, output_num=self.radix)
        self.split1 = P.Split(axis=1, output_num=2)
        self.sum = P.AddN()
        self.cast = P.Cast()

    def construct(self, x):
        """Split attention construct"""
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        x = self.relu(x)
        batch = x.shape[0]
        splited = ()
        if self.radix > 1:
            if self.radix != 2:
                print("error")
            outputs = ()
            splited = (x[:, 0: (x.shape[1]// 2), :, :], x[:, (x.shape[1]// 2):, :, :])
            for i in range(self.radix):
                outputs = outputs + (self.cast(splited[i], mstype.float32),)
            gap = self.sum(outputs)
        else:
            gap = x
        gap_size = (gap.shape)[-1]
        gap = AdaptiveAvGPool(gap_size)(gap)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten)
        atten = self.reshape(atten, (batch, -1, 1, 1))
        if self.radix > 1:
            attens = (atten[:, 0:(atten.shape[1]//2), :, :], atten[:, (atten.shape[1]//2):, :, :])
        else:
            attens = (atten,)

        if self.radix > 1:
            outputs = ()
            for i in range(self.radix):
                outputs = outputs + (attens[i] * splited[i],)
            out = self.sum(outputs)
        else:
            out = atten * x
        return out

class rSoftMax(nn.Cell):
    """rSoftMax"""
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality
        self.softmax = P.Softmax(axis=1)
        self.reshape1 = P.Reshape()
        self.reshape2 = P.Reshape()
        self.sigmoid = P.Sigmoid()

    def construct(self, x):
        """rSoftMax construct"""
        batch = (x.shape)[0]
        if self.radix > 1:
            x = self.reshape1(x, (batch, self.cardinality, self.radix, -1))
            x = x.transpose((0, 2, 1, 3))
            x = self.softmax(x)
            x = self.reshape2(x, (batch, -1))
        else:
            x = self.sigmoid(x)
        return x

class AdaptiveAvGPool(nn.Cell):
    def __init__(self, input_size):
        super().__init__()
        self.AvgPool2d = nn.AvgPool2d(input_size, stride=1)

    def construct(self, x):
        x = self.AvgPool2d(x)
        return x
