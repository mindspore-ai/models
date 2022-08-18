# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""StNet"""
import math
import numpy as np
import mindspore.nn as nn
from mindspore.common.initializer import HeNormal, HeUniform, Uniform
from mindspore.ops import operations as ops
from mindspore import Tensor

from src.config import config as cfg


class Bottleneck(nn.Cell):
    """resnet block"""
    expansion = 4

    def __init__(self, inplanes, planes, cardinality, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, has_bias=False, pad_mode='pad'
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, has_bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()
        # Downsample
        self.down_sample_layer = downsample
        self.stride = stride

    def construct(self, x):
        """construct"""
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

        res = out + residual
        res = self.relu(res)
        return res


class TemporalXception(nn.Cell):
    '''
        model=TemporalXception(2048,2048)
    '''

    def __init__(self, in_channels, out_channels):
        super(TemporalXception, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.att_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), stride=(1, 1), padding=(1, 1, 0, 0),
                                  # group=2048,
                                  weight_init=HeUniform(), pad_mode='pad', has_bias=True)
        self.att_2 = nn.Conv2d(out_channels, 1024, kernel_size=(1, 1), stride=(1, 1), weight_init=HeUniform()
                               , has_bias=True)
        self.bn2 = nn.BatchNorm2d(1024)
        self.att_1 = nn.Conv2d(1024, 1024, kernel_size=(3, 1), stride=(1, 1), padding=(1, 1, 0, 0),
                               # group=1024,
                               weight_init=HeUniform(), pad_mode='pad', has_bias=True)
        self.att1_2 = nn.Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), weight_init=HeUniform(), has_bias=True)
        self.dw = nn.Conv2d(in_channels, 1024, kernel_size=(1, 1), stride=(1, 1), weight_init=HeUniform(),
                            has_bias=True)
        self.relu = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(1024)

    def construct(self, x):
        """construct"""
        x = self.bn1(x)
        x1 = self.att_conv(x)
        x1 = self.att_2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu(x1)
        x1 = self.att_1(x1)
        x1 = self.att1_2(x1)
        x2 = self.dw(x)
        add_to = x1 + x2

        return self.relu(self.bn3(add_to))


class TemporalBlock(nn.Cell):
    """temp model"""
    def __init__(self, channels):
        super(TemporalBlock, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv3d(
            channels,
            channels,
            kernel_size=(3, 1, 1),
            stride=1,
            pad_mode="pad",
            padding=(1, 1, 0, 0, 0, 0),
            weight_init=HeUniform(),
            has_bias=True
        )
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU()
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()

    def construct(self, x):
        """construct"""
        B, T, C, H, W = x.shape
        out = self.transpose(x, (0, 2, 1, 3, 4))

        x = self.conv1(out)
        x = self.relu(x)
        x = self.bn1(x)
        x = x + out
        x = self.transpose(x, (0, 2, 1, 3, 4))
        x = self.reshape(x, (B * T, C, H, W))
        return x


class Stnet_Res_model(nn.Cell):
    """main model"""
    def __init__(
            self, block, layers, cardinality=32, num_classes=101, T=7, N=5, input_channels=3,
    ):
        super(Stnet_Res_model, self).__init__()
        self.inplanes = 64
        self.cardinality = cardinality
        self.T = T
        self.N = N
        self.conv1 = nn.Conv2d(
            input_channels * self.N, 64, kernel_size=7, stride=2, padding=3, has_bias=False, pad_mode='pad'
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.temp1 = TemporalBlock(512)
        self.temp2 = TemporalBlock(1024)
        self.op_avg = nn.AvgPool2d(kernel_size=cfg.avgpool_kernel_size, pad_mode="valid")
        self.xception = TemporalXception(2048, 2048)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(self.T, 1))
        self.reshape = ops.Reshape()
        self.sqrt = ops.Sqrt()
        stdv = 1.0/math.sqrt(1024*1.0)
        self.fc = nn.Dense(1024, num_classes, weight_init=Uniform(stdv))
        self.transpose = ops.Transpose()

    def _initialize_weights(self):
        self.init_parameters_data()
        for _, m in self.cells_and_names():
            if isinstance(m, nn.Conv2d):
                kaiming_norml = HeNormal(negative_slope=0, mode="fan_out", nonlinearity="relu")
                m.weight_init = kaiming_norml
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma.set_data(
                    Tensor(np.ones(m.gamma.data.shape, dtype="float32")))
                m.beta.set_data(
                    Tensor(np.zeros(m.beta.data.shape, dtype="float32")))

    def _make_layer(self, block, planes, blocks, stride=1):
        """_make_layer"""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    has_bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, self.cardinality, stride, downsample)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.cardinality))

        return nn.SequentialCell(*layers)

    def construct(self, x):
        """construct"""
        # size (batch_size, T, video_length = channels* N, height, width)
        B, C, _, H, W = x.shape
        x = self.reshape(x, (B * self.T, self.N * C, H, W))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        size = x.shape
        x = self.reshape(x, (B, self.T, size[1], size[2], size[3]))
        x = self.temp1(x)
        x = self.layer3(x)
        size = x.shape
        x = self.reshape(x, (B, self.T, size[1], size[2], size[3]))
        x = self.temp2(x)
        x = self.layer4(x)
        pool = self.op_avg(x)
        x = self.reshape(pool, (-1, self.T, pool.shape[1], 1))
        x = self.transpose(x, (0, 2, 1, 3))
        x = self.xception(x)
        x = self.maxpool1(x)
        x = self.reshape(x, (-1, 1024))
        x = self.fc(x)
        return x


def stnet50(**kwargs):
    """
    Construct stnet with a Resnet 50 backbone.
    """

    model = Stnet_Res_model(
        Bottleneck,
        [3, 4, 6, 3],
        **kwargs,
    )
    return model
