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
"""ProxylessNAS-Mobile model definition"""
from inspect import isfunction
from collections import OrderedDict

import mindspore.nn as nn
import mindspore.ops.operations as P

def get_activation_layer(activation):
    """
    get_activation_layer: to create an activation layer according to the string or function.
    """
    assert activation is not None
    if isfunction(activation):
        return activation()
    if isinstance(activation, str):
        if activation == "relu":
            return nn.ReLU()
        if activation == "relu6":
            return nn.ReLU6()
        if activation == "swish":
            return Swish()
        if activation == "hswish":
            return HSwish(inplace=True)
        if activation == "sigmoid":
            return nn.sigmoid()
        if activation == "hsigmoid":
            return HSigmoid()
        if activation == "identity":
            return Identity()

        raise NotImplementedError()

    assert isinstance(activation, nn.Cell)
    return activation

class ConvBlock(nn.Cell):
    """
    ConvBlock: the convolution block with Batch normalization and activation.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 groups=1,
                 bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation=nn.ReLU()):
        super(ConvBlock, self).__init__()
        self.activate = (activation is not None)
        self.use_bn = use_bn
        self.use_pad = (isinstance(padding, (list, tuple)) and (len(padding) == 4))

        if self.use_pad:
            self.pad = nn.ZeroPad2d(padding=padding)
            padding = 0
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, pad_mode='pad', padding=padding, dilation=dilation,
                              group=groups, has_bias=bias)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(num_features=out_channels, eps=bn_eps, momentum=0.9)
        if self.activate:
            self.activate_layer = get_activation_layer(activation)

    def construct(self, x):
        """ construct network """
        if self.use_pad:
            x = self.pad(x)
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activate:
            x = self.activate_layer(x)
        return x

def conv1x1_block(in_channels,
                  out_channels,
                  stride=1,
                  padding=0,
                  groups=1,
                  bias=False,
                  use_bn=True,
                  bn_eps=1e-5,
                  activation=nn.ReLU()):
    """
    conv1x1_block: the 1x1 version of the ConvBlock.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation)

def conv3x3_block(in_channels,
                  out_channels,
                  stride=1,
                  padding=1,
                  dilation=1,
                  groups=1,
                  bias=False,
                  use_bn=True,
                  bn_eps=1e-5,
                  activation=nn.ReLU()):
    """
    conv3x3_block: the 3x3 version of the ConvBlock.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation)

class ProxylessBlock(nn.Cell):
    """
    ProxylessBlock: the block for residual path in class ProxylessUnit.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 bn_eps,
                 expansion):
        super(ProxylessBlock, self).__init__()
        self.use_bc = (expansion > 1)
        mid_channels = in_channels * expansion

        if self.use_bc:
            self.bc_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                bn_eps=bn_eps,
                activation="relu6")

        padding = (kernel_size - 1) // 2
        self.dw_conv = ConvBlock(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=mid_channels,
            bn_eps=bn_eps,
            activation="relu6")
        self.pw_conv = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            bn_eps=bn_eps,
            activation=None)

    def construct(self, x):
        """ construct network """
        if self.use_bc:
            x = self.bc_conv(x)
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class ProxylessUnit(nn.Cell):
    """
    ProxylessNAS unit.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 bn_eps,
                 expansion,
                 residual,
                 shortcut):
        super(ProxylessUnit, self).__init__()
        assert (residual or shortcut)
        self.residual = residual
        self.shortcut = shortcut

        if self.residual:
            self.body = ProxylessBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bn_eps=bn_eps,
                expansion=expansion)

    def construct(self, x):
        """ construct network """
        if not self.residual:
            return x
        if not self.shortcut:
            return self.body(x)
        identity = x
        x = self.body(x)
        x = identity + x
        return x


class ProxylessNAS(nn.Cell):
    """
    ProxylessNAS: the implementation of ProxylessNAS-Mobile model.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 final_block_channels,
                 residuals,
                 shortcuts,
                 kernel_sizes,
                 expansions,
                 bn_eps=1e-3,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(ProxylessNAS, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        init_block = conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            stride=2,
            bn_eps=bn_eps,
            activation="relu6")

        #stage 1
        unit1 = ProxylessUnit(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            stride=1,
            bn_eps=0.001,
            expansion=1,
            residual=True,
            shortcut=False)

        stage1 = OrderedDict([('unit1', unit1)])

        stage1 = nn.SequentialCell(stage1)

        #stage 2
        unit1 = ProxylessUnit(
            in_channels=16,
            out_channels=32,
            kernel_size=5,
            stride=2,
            bn_eps=0.001,
            expansion=3,
            residual=True,
            shortcut=False)

        unit2 = ProxylessUnit(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            bn_eps=0.001,
            expansion=3,
            residual=True,
            shortcut=True)

        unit3 = ProxylessUnit(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            bn_eps=0.001,
            expansion=3,
            residual=False,
            shortcut=True)

        unit4 = ProxylessUnit(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            bn_eps=0.001,
            expansion=3,
            residual=False,
            shortcut=True)

        stage2 = OrderedDict([('unit1', unit1),
                              ('unit2', unit2),
                              ('unit3', unit3),
                              ('unit4', unit4)
                             ])

        stage2 = nn.SequentialCell(stage2)

        #stage 3
        unit1 = ProxylessUnit(
            in_channels=32,
            out_channels=40,
            kernel_size=7,
            stride=2,
            bn_eps=0.001,
            expansion=3,
            residual=True,
            shortcut=False)

        unit2 = ProxylessUnit(
            in_channels=40,
            out_channels=40,
            kernel_size=3,
            stride=1,
            bn_eps=0.001,
            expansion=3,
            residual=True,
            shortcut=True)

        unit3 = ProxylessUnit(
            in_channels=40,
            out_channels=40,
            kernel_size=5,
            stride=1,
            bn_eps=0.001,
            expansion=3,
            residual=True,
            shortcut=True)

        unit4 = ProxylessUnit(
            in_channels=40,
            out_channels=40,
            kernel_size=5,
            stride=1,
            bn_eps=0.001,
            expansion=3,
            residual=True,
            shortcut=True)

        stage3 = OrderedDict([('unit1', unit1),
                              ('unit2', unit2),
                              ('unit3', unit3),
                              ('unit4', unit4)
                             ])

        stage3 = nn.SequentialCell(stage3)

        #stage 4
        unit1 = ProxylessUnit(
            in_channels=40,
            out_channels=80,
            kernel_size=7,
            stride=2,
            bn_eps=0.001,
            expansion=6,
            residual=True,
            shortcut=False)

        unit2 = ProxylessUnit(
            in_channels=80,
            out_channels=80,
            kernel_size=5,
            stride=1,
            bn_eps=0.001,
            expansion=3,
            residual=True,
            shortcut=True)

        unit3 = ProxylessUnit(
            in_channels=80,
            out_channels=80,
            kernel_size=5,
            stride=1,
            bn_eps=0.001,
            expansion=3,
            residual=True,
            shortcut=True)

        unit4 = ProxylessUnit(
            in_channels=80,
            out_channels=80,
            kernel_size=5,
            stride=1,
            bn_eps=0.001,
            expansion=3,
            residual=True,
            shortcut=True)

        unit5 = ProxylessUnit(
            in_channels=80,
            out_channels=96,
            kernel_size=5,
            stride=1,
            bn_eps=0.001,
            expansion=6,
            residual=True,
            shortcut=False)

        unit6 = ProxylessUnit(
            in_channels=96,
            out_channels=96,
            kernel_size=5,
            stride=1,
            bn_eps=0.001,
            expansion=3,
            residual=True,
            shortcut=True)

        unit7 = ProxylessUnit(
            in_channels=96,
            out_channels=96,
            kernel_size=5,
            stride=1,
            bn_eps=0.001,
            expansion=3,
            residual=True,
            shortcut=True)

        unit8 = ProxylessUnit(
            in_channels=96,
            out_channels=96,
            kernel_size=5,
            stride=1,
            bn_eps=0.001,
            expansion=3,
            residual=True,
            shortcut=True)

        stage4 = OrderedDict([('unit1', unit1),
                              ('unit2', unit2),
                              ('unit3', unit3),
                              ('unit4', unit4),
                              ('unit5', unit5),
                              ('unit6', unit6),
                              ('unit7', unit7),
                              ('unit8', unit8),
                             ])

        stage4 = nn.SequentialCell(stage4)

        #stage 5
        unit1 = ProxylessUnit(
            in_channels=96,
            out_channels=192,
            kernel_size=7,
            stride=2,
            bn_eps=0.001,
            expansion=6,
            residual=True,
            shortcut=False)

        unit2 = ProxylessUnit(
            in_channels=192,
            out_channels=192,
            kernel_size=7,
            stride=1,
            bn_eps=0.001,
            expansion=6,
            residual=True,
            shortcut=True)

        unit3 = ProxylessUnit(
            in_channels=192,
            out_channels=192,
            kernel_size=7,
            stride=1,
            bn_eps=0.001,
            expansion=3,
            residual=True,
            shortcut=True)

        unit4 = ProxylessUnit(
            in_channels=192,
            out_channels=192,
            kernel_size=7,
            stride=1,
            bn_eps=0.001,
            expansion=3,
            residual=True,
            shortcut=True)

        unit5 = ProxylessUnit(
            in_channels=192,
            out_channels=320,
            kernel_size=7,
            stride=1,
            bn_eps=0.001,
            expansion=6,
            residual=True,
            shortcut=False)

        stage5 = OrderedDict([('unit1', unit1),
                              ('unit2', unit2),
                              ('unit3', unit3),
                              ('unit4', unit4),
                              ('unit5', unit5),
                             ])

        stage5 = nn.SequentialCell(stage5)

        final_block = conv1x1_block(
            in_channels=320,
            out_channels=1280,
            bn_eps=bn_eps,
            activation="relu6")

        final_pool = nn.AvgPool2d(kernel_size=7, stride=1, pad_mode='valid')

        #combine from init_block to final_pool
        features = OrderedDict([('init_block', init_block),
                                ('stage1', stage1),
                                ('stage2', stage2),
                                ('stage3', stage3),
                                ('stage4', stage4),
                                ('stage5', stage5),
                                ('final_block', final_block),
                                ('final_pool', final_pool),
                                ])
        self.features = nn.SequentialCell(features)

        #output layer
        in_channels = final_block_channels
        self.output = nn.Dense(in_channels=in_channels, out_channels=num_classes)

    def construct(self, x):
        """ construct network """
        x = self.features(x)

        x = P.Reshape()(x, (P.Shape()(x)[0], -1,))
        x = self.output(x)
        return x

def proxylessnas_mobile(num_classes=1000):
    """
    ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware.
    https://arxiv.org/abs/1812.00332v2
    Parameters:
    ----------
    num_classes : int, default 1000
        Number of classification classes.
    """
    residuals = [[1], [1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
    channels = [[16], [32, 32, 32, 32], [40, 40, 40, 40], [80, 80, 80, 80, 96, 96, 96, 96],
                [192, 192, 192, 192, 320]]
    kernel_sizes = [[3], [5, 3, 3, 3], [7, 3, 5, 5], [7, 5, 5, 5, 5, 5, 5, 5], [7, 7, 7, 7, 7]]
    expansions = [[1], [3, 3, 3, 3], [3, 3, 3, 3], [6, 3, 3, 3, 6, 3, 3, 3], [6, 6, 3, 3, 6]]
    init_block_channels = 32
    final_block_channels = 1280

    shortcuts = [[0], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1, 0, 1, 1, 1], [0, 1, 1, 1, 0]]

    net = ProxylessNAS(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        residuals=residuals,
        shortcuts=shortcuts,
        kernel_sizes=kernel_sizes,
        expansions=expansions,
        num_classes=num_classes
        )

    return net
