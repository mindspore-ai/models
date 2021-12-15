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
"""
This is the 3D-SkipDenseSeg model definition
"""
import math
from collections import OrderedDict
from src.var_init import KaimingNormal
from mindspore.common import initializer as init
import mindspore
import mindspore.nn as nn
import mindspore.ops as OP

class _DenseLayer(nn.Cell):
    """DenseLayer"""
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        cell_list = [
            nn.BatchNorm3d(num_input_features),
            nn.ReLU(),
            nn.Conv3d(num_input_features, bn_size * growth_rate, \
                      kernel_size=1, stride=1, bias_init=False),
            nn.BatchNorm3d(bn_size * growth_rate),
            nn.ReLU(),
            nn.Conv3d(bn_size * growth_rate, growth_rate,\
                      kernel_size=3, stride=1, padding=1, pad_mode='pad', bias_init=False)
        ]
        self.cells_list = nn.SequentialCell(cell_list)
        self.dropout = nn.Dropout(1 - drop_rate)

    def construct(self, x):
        """DenseLayer construct"""
        new_features = self.cells_list(x)
        if self.dropout is not None:
            new_features = self.dropout(new_features)
        return OP.Concat(1)((x, new_features))


class _DenseBlock(nn.Cell):
    """DenseBlock"""
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        cell_list = []
        self.cells_list = nn.SequentialCell(cell_list)
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,\
                                growth_rate, bn_size, drop_rate)
            self.cells_list.append(layer)

    def construct(self, x):
        """_DenseBlock construct"""
        new_features = self.cells_list(x)
        return new_features


class _Transition(nn.Cell):
    """
    Transition block layer
    """
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.features = nn.SequentialCell(OrderedDict([
            ('norm', nn.BatchNorm3d(num_input_features)),
            ('relu', nn.ReLU()),
            ('conv', nn.Conv3d(num_input_features, num_output_features,\
                                          kernel_size=1, stride=1, bias_init=False)),
            ('pool_norm', nn.BatchNorm3d(num_output_features)),
            ('pool_relu', nn.ReLU()),
            ('pool', nn.Conv3d(num_output_features, num_output_features, kernel_size=2, stride=2))
        ]))

    def construct(self, x):
        """_Trasition block mindspore construct"""
        x = self.features(x)
        return x


class DenseNet(nn.Cell):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=16, block_config=(6, 12, 24, 16),
                 num_init_features=32, bn_size=4, drop_rate=0, num_classes=9):
        super(DenseNet, self).__init__()
        #First three convolutions
        self.features = nn.SequentialCell(OrderedDict([
            ('conv0', nn.Conv3d(2, num_init_features, kernel_size=3, stride=1, \
                                padding=1, pad_mode='pad', bias_init=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU()),
            ('conv1', nn.Conv3d(num_init_features, num_init_features, kernel_size=3, stride=1, \
                                padding=1, pad_mode='pad', bias_init=False)),
            ('norm1', nn.BatchNorm3d(num_init_features)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv3d(num_init_features, num_init_features, kernel_size=3, stride=1, \
                                padding=1, pad_mode='pad', bias_init=False)),
        ]))
        self.features_bn = nn.SequentialCell(OrderedDict([
            ('norm2', nn.BatchNorm3d(num_init_features)),
            ('relu2', nn.ReLU()),
        ]))
        self.conv_pool_first = nn.Conv3d(num_init_features, num_init_features, kernel_size=2, \
                                         stride=2, padding=0, bias_init=False)
        #Each denseblock
        num_features = num_init_features
        self.dense_blocks = nn.CellList([])
        self.transit_blocks = nn.CellList([])
        self.upsampling_blocks = nn.CellList([])
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate
            ksize = (2 ** (i + 1)) + 2
            up_block = nn.Conv3dTranspose(in_channels=num_features,
                                          out_channels=num_classes,\
                                          kernel_size=ksize, stride=2 ** (i + 1), \
                                          padding=1, pad_mode='pad', bias_init=False)
            self.upsampling_blocks.append(up_block)
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, \
                                    num_output_features=num_features // 2)
                self.transit_blocks.append(trans)
                num_features = num_features // 2
        #classifier
        self.bn_class = nn.BatchNorm3d(num_classes * 2 + num_init_features)
        self.conv_class = nn.Conv3d(num_classes * 2 + num_init_features,\
                                    num_classes, kernel_size=1, padding=0)
        for cell in self.cells():
            if isinstance(cell, nn.Conv3d):
                cell.weight.set_data(init.initializer(KaimingNormal(\
                      a=math.sqrt(5), mode='fan_out', nonlinearity='relu'),\
                      cell.weight.shape,\
                      cell.weight.dtype))

    def construct(self, x):
        """
        model construct
        """
        first_three_features = self.features(x)
        first_three_features_bn = self.features_bn(first_three_features)
        out = self.conv_pool_first(first_three_features_bn)
        out = self.dense_blocks[0](out)
        up_block1 = self.upsampling_blocks[0](out)
        out = self.transit_blocks[0](out)
        out = self.dense_blocks[1](out)
        up_block2 = self.upsampling_blocks[1](out)
        out = mindspore.ops.Concat(1)((up_block1, up_block2, first_three_features))
        #classifier
        out = self.conv_class(OP.ReLU()(self.bn_class(out)))
        return out
        