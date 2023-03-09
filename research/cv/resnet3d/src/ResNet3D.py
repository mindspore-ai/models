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
"""Resnet3d"""

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import operations as P


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, pad_mode='pad', padding=1, has_bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, pad_mode='same', padding=0, has_bias=False)


class BatchNorm3d(nn.Cell):
    """
    Define BatchNorm3d.
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.9, affine=True, gamma_init='zero',
                 beta_init='zeros', moving_mean_init='zeros', moving_var_init='ones'):
        super(BatchNorm3d, self).__init__()
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.bn2d = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine,
                                   gamma_init=gamma_init, beta_init=beta_init,
                                   moving_mean_init=moving_mean_init, moving_var_init=moving_var_init)

    def construct(self, x):
        x_shape = self.shape(x)
        x = self.reshape(
            x, (x_shape[0], x_shape[1], x_shape[2] * x_shape[3], x_shape[4]))
        bn2d_out = self.bn2d(x)
        bn3d_out = self.reshape(bn2d_out, x_shape)
        return bn3d_out


class SyncBatchNorm3d(nn.Cell):
    """
    Define SyncBatchNorm3d.
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.9, affine=True, gamma_init='zero',
                 beta_init='zeros', moving_mean_init='zeros', moving_var_init='ones'):
        super(SyncBatchNorm3d, self).__init__()
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.bn2d = nn.SyncBatchNorm(num_features, eps=eps, momentum=momentum, affine=affine,
                                     gamma_init=gamma_init, beta_init=beta_init,
                                     moving_mean_init=moving_mean_init, moving_var_init=moving_var_init)

    def construct(self, x):
        x_shape = self.shape(x)
        x = self.reshape(
            x, (x_shape[0], x_shape[1], x_shape[2] * x_shape[3], x_shape[4]))
        bn2d_out = self.bn2d(x)
        bn3d_out = self.reshape(bn2d_out, x_shape)
        return bn3d_out


def _bn(channel, sync=False):
    if sync:
        bn = SyncBatchNorm3d(channel, eps=1e-05, momentum=0.9, affine=True, gamma_init='ones',
                             beta_init='zeros', moving_mean_init='zeros', moving_var_init='ones')
    else:
        bn = BatchNorm3d(channel, eps=1e-05, momentum=0.9, affine=True, gamma_init='ones',
                         beta_init='zeros', moving_mean_init='zeros', moving_var_init='ones')
    return bn


def _bn_last(channel, sync=False):
    if sync:
        bn_last = SyncBatchNorm3d(channel, eps=1e-05, momentum=0.9, affine=True, gamma_init='zero',
                                  beta_init='zeros', moving_mean_init='zeros', moving_var_init='ones')
    else:
        bn_last = BatchNorm3d(channel, eps=1e-05, momentum=0.9, affine=True, gamma_init='zero',
                              beta_init='zeros', moving_mean_init='zeros', moving_var_init='ones')
    return bn_last


class BasicBlock(nn.Cell):
    """
    Define BasicBlock for resnet3d.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, sync=False):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = _bn(planes, sync)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = _bn_last(planes, sync)
        self.downsample = downsample
        self.add = ops.Add()
        self.stride = stride

    def construct(self, x):
        """
        Construct BasicBlock.

        Returns
            BasicBlock for resnet3d.
        """
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add(out, identity)
        out = self.relu(out)

        return out


class Bottleneck(nn.Cell):
    """
    Define Bottleneck for resnet3d.
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, sync=False):
        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = _bn(planes, sync)

        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = _bn(planes, sync)

        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = _bn_last(planes * self.expansion, sync)

        self.relu = nn.ReLU()
        self.downsample = downsample
        self.add = ops.Add()
        self.stride = stride

    def construct(self, x):
        """Construct Bottleneck."""
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

        out = self.add(out, identity)
        out = self.relu(out)

        return out


class ResNet3D(nn.Cell):
    """
    ResNet3D architecture.
    """

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 widen_factor=1.0,
                 n_classes=400,
                 stop_weights_update=False,
                 sync=False
                 ):
        super(ResNet3D, self).__init__()
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        self.stop_weights_update = stop_weights_update
        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3,
                                        conv1_t_size // 2, 3, 3),
                               pad_mode='pad',
                               has_bias=False)
        self.bn1 = _bn(self.in_planes, sync)
        self.relu = nn.ReLU()
        self.maxpool = ops.MaxPool3D(kernel_size=3, strides=2, pad_mode="same")
        self.layer1 = self._make_layer(
            block, block_inplanes[0], layers[0], sync=sync)
        self.layer2 = self._make_layer(
            block, block_inplanes[1], layers[1], stride=2, sync=sync)
        self.layer3 = self._make_layer(
            block, block_inplanes[2], layers[2], stride=2, sync=sync)
        self.layer4 = self._make_layer(
            block, block_inplanes[3], layers[3], stride=2, sync=sync)
        self.mean = ops.ReduceMean(keep_dims=True)
        self.flatten = nn.Flatten()
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Dense(block_inplanes[3] * block.expansion, n_classes)

    def _make_layer(self, block, planes, blocks, stride=1, sync=False):
        """
        Make stage network of ResNet3D.
        """
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.SequentialCell(
                conv1x1x1(self.in_planes, planes * block.expansion, stride),
                _bn(planes * block.expansion, sync)
            )

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample,
                  sync=sync)
        )
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, sync=sync))

        return nn.SequentialCell(layers)

    def construct(self, x):
        """
        Construct ResNet3D.
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if not self.no_max_pool:
            out = self.maxpool(out)

        c1 = self.layer1(out)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.mean(c4, (2, 3, 4))
        c6 = self.flatten(c5)
        # c6 = self.drop(c6)
        out = self.fc(c6)

        return out


def generate_model(**kwargs):
    """
    Get ResNet3D with GloreUnit neural network.
    """
    model = ResNet3D(Bottleneck, [3, 4, 6, 3], [64, 128, 256, 512], **kwargs)
    return model
