# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License Version 2.0(the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0#
#
# Unless required by applicable law or agreed to in writing software
# distributed under the License is distributed on an "AS IS" BASIS
# WITHOUT WARRANT IES OR CONITTONS OF ANY KIND， either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ====================================================================================

"""Define OSNet"""

import os
import warnings
from collections import OrderedDict
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import initializer, HeNormal
from mindspore import load_checkpoint, load_param_into_net


def kaiming_normal(shape, mode='fan_out', nonlinearity='relu'):
    '''initialize weight of conv2d layer.'''
    weight = initializer(HeNormal(mode=mode, nonlinearity=nonlinearity), shape=shape)
    return weight

def _conv2d(in_channels, out_channels, kernel_size, stride, pad_mode, padding, group=1, has_bias=False):
    '''return conv2d layer with initialized weight'''
    if in_channels % group == 0 and out_channels % group == 0:
        weight_shape = [out_channels, in_channels//group, kernel_size, kernel_size]
    else:
        raise ValueError("In_ channels:{} and out_channels:{} must be divisible by the number of groups:{}."
                         .format(in_channels, out_channels, group))
    weight = kaiming_normal(weight_shape)
    conv = nn.Conv2d(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     stride=stride,
                     pad_mode=pad_mode,
                     padding=padding,
                     group=group,
                     has_bias=has_bias,
                     weight_init=weight
                     )
    return conv


class ConvLayer(nn.Cell):
    '''Convolution layer (conv + bn + relu).'''

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            pad_mode='pad',
            padding=3,
            group=1,
            has_bias=False
    ):
        super(ConvLayer, self).__init__()
        self.conv = _conv2d(in_channels, out_channels, kernel_size,
                            stride, pad_mode, padding, group, has_bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1(nn.Cell):
    """1x1 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, group=1):
        super(Conv1x1, self).__init__()
        self.conv = _conv2d(in_channels, out_channels, 1, stride=stride, pad_mode='valid', padding=0, group=group)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1Linear(nn.Cell):
    '''1x1 convolution + bn (w/o non-linearity).'''

    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv1x1Linear, self).__init__()
        self.conv = _conv2d(in_channels, out_channels, 1, stride, pad_mode='valid', padding=0)
        self.bn = nn.BatchNorm2d(out_channels)

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Conv3x3(nn.Cell):
    """3x3 convolution + bn + relu."""
    def __init__(self, in_channels, out_channels, stride=1, group=1):
        super(Conv3x3, self).__init__()
        self.conv = _conv2d(in_channels, out_channels, 3, stride, pad_mode='pad', padding=1, group=group)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LightConv3x3(nn.Cell):
    """Lightweight 3x3 convolution.
    1x1 (linear) + dw 3x3 (nonlinear).
    """
    def __init__(self, in_channels, out_channels):
        super(LightConv3x3, self).__init__()
        self.conv1 = _conv2d(in_channels, out_channels, 1, stride=1, pad_mode='valid', padding=0)
        self.conv2 = _conv2d(out_channels, out_channels, 3, stride=1, pad_mode='pad', padding=1, group=out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ChannelGate(nn.Cell):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(
            self,
            in_channels,
            num_gates=None,
            return_gates=False,
            gate_activation='sigmoid',
            reduction=16,
    ):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = ops.ReduceMean(keep_dims=True)
        self.fc1 = _conv2d(in_channels, in_channels//reduction, kernel_size=1, stride=1,
                           pad_mode='valid', padding=0, has_bias=True)
        self.relu = nn.ReLU()
        self.fc2 = _conv2d(in_channels//reduction, num_gates, kernel_size=1, stride=1,
                           pad_mode='valid', padding=0, has_bias=True)
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU()
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError(
                "Unknown gate activation: {}".format(gate_activation)
            )

    def construct(self, x):
        '''constuct function'''
        inputs = x
        x = self.global_avgpool(x, (2, 3))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return inputs * x


class OSBlock(nn.Cell):
    """Omni-scale feature learning block."""

    def __init__(
            self,
            in_channels,
            out_channels,
            bottleneck_reduction=4,
            **kwargs
    ):
        super(OSBlock, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.SequentialCell(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.SequentialCell(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2d = nn.SequentialCell(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        self.relu = nn.ReLU()
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)

    def construct(self, x):
        '''construct layer'''
        identity = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        add = x3 + identity
        out = self.relu(add)
        return out


class OSNet(nn.Cell):
    """Omni-Scale Network."""
    def __init__(
            self,
            num_classes,
            blocks,
            layers,
            channels,
            feature_dim=512,
            **kwargs
    ):
        super(OSNet, self).__init__()
        num_blocks = len(blocks)
        assert num_blocks == len(layers)
        assert num_blocks == len(channels) - 1
        self.feature_dim = feature_dim
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3)
        self.pad = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT")
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv2 = self._make_layer(
            blocks[0],
            layers[0],
            channels[0],
            channels[1],
            reduce_spatial_size=True,
        )
        self.conv3 = self._make_layer(
            blocks[1],
            layers[1],
            channels[1],
            channels[2],
            reduce_spatial_size=True
        )
        self.conv4 = self._make_layer(
            blocks[2],
            layers[2],
            channels[2],
            channels[3],
            reduce_spatial_size=False
        )
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = ops.ReduceMean(keep_dims=True)
        self.fc = self._construct_fc_layer(
            self.feature_dim, channels[3], dropout_p=None
        )
        self.classifier = nn.Dense(self.feature_dim, num_classes)
        self.stop_layer = ops.Identity()

    def _make_layer(
            self,
            block,
            layer,
            in_channels,
            out_channels,
            reduce_spatial_size,
    ):
        '''make block layers.'''
        layers = []
        layers.append(block(in_channels, out_channels))
        for _ in range(1, layer):
            layers.append(block(out_channels, out_channels))
        if reduce_spatial_size:
            layers.append(
                nn.SequentialCell(
                    Conv1x1(out_channels, out_channels),
                    nn.AvgPool2d(2, stride=2)
                )
            )

        return nn.SequentialCell(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        '''constuct full-connection layer.'''
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return None
        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]

        layers = []
        for dim in fc_dims:
            layers.append(nn.Dense(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.SequentialCell(*layers)

    def construct(self, x):
        '''construct'''
        x = self.conv1(x)
        x = self.pad(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        v = self.global_avgpool(x, (2, 3))
        v = v.view(v.shape[0], -1)
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            return v
        y = self.stop_layer(v)
        y = self.classifier(v)
        return y


def init_pretrained_weights(model, pretrained_param_dir):
    """
    Initializes model with pretrained weights.
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """

    filename = 'init_osnet.ckpt'
    file = os.path.join(pretrained_param_dir, filename)
    print(file)
    if not os.path.exists(file):
        raise ValueError(
            'The file:{} does not exist.'.format(file)
        )
    param_dict = load_checkpoint(file)
    model_dict = model.parameters_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in param_dict.items():
        if k in model_dict and model_dict[k].data.shape == v.shape:
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    load_param_into_net(model, model_dict)

    if matched_layers:
        warnings.warn(
            'The pretrained weights from "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(file)
        )
    else:
        print(
            'Successfully loaded imagenet pretrained weights from "{}"'.
            format(file)
        )
        if discarded_layers:
            print(
                '** The following layers are discarded '
                'due to unmatched keys or layer size: {}'.
                format(discarded_layers)
            )


def create_osnet(num_classes=1500, pretrained=False, pretrained_dir='./model_utils', **kwargs):
    '''create osnet.'''
    model = OSNet(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[64, 256, 384, 512],
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, pretrained_dir)
    return model
