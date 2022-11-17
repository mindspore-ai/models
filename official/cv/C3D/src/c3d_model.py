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

import math
import mindspore.nn as nn
import mindspore.ops as P
from mindspore.common import initializer as init
from src.utils import default_recurisive_init, KaimingNormal


class C3D(nn.Cell):
    """
    C3D network definition.

    Args:
        num_classes (int): Class numbers. Default: 1000.
    Returns:
        Tensor, infer output tensor.

    Examples:
        >>> C3D(num_classes=1000)
    """

    def __init__(self, num_classes=1000):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(3, 3, 3),
                               padding=(1, 1, 1, 1, 1, 1), pad_mode='pad', has_bias=True)
        self.pool1 = P.MaxPool3D(kernel_size=(1, 2, 2), strides=(1, 2, 2), pad_mode='same')

        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3),
                               padding=(1, 1, 1, 1, 1, 1), pad_mode='pad', has_bias=True)
        self.pool2 = P.MaxPool3D(kernel_size=(2, 2, 2), strides=(2, 2, 2), pad_mode='same')

        self.conv3a = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3, 3, 3),
                                padding=(1, 1, 1, 1, 1, 1), pad_mode='pad', has_bias=True)
        self.conv3b = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3, 3, 3),
                                padding=(1, 1, 1, 1, 1, 1), pad_mode='pad', has_bias=True)
        self.pool3 = P.MaxPool3D(kernel_size=(2, 2, 2), strides=(2, 2, 2), pad_mode='same')

        self.conv4a = nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(3, 3, 3),
                                padding=(1, 1, 1, 1, 1, 1), pad_mode='pad', has_bias=True)
        self.conv4b = nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3, 3, 3),
                                padding=(1, 1, 1, 1, 1, 1), pad_mode='pad', has_bias=True)
        self.pool4 = P.MaxPool3D(kernel_size=(2, 2, 2), strides=(2, 2, 2), pad_mode='same')

        self.conv5a = nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3, 3, 3),
                                padding=(1, 1, 1, 1, 1, 1), pad_mode='pad', has_bias=True)
        self.conv5b = nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3, 3, 3),
                                padding=(1, 1, 1, 1, 1, 1), pad_mode='pad', has_bias=True)
        self.pool5 = P.MaxPool3D(kernel_size=(2, 2, 2), strides=(2, 2, 2), pad_mode='same')

        self.fc6 = nn.Dense(in_channels=8192, out_channels=4096)
        self.fc7 = nn.Dense(in_channels=4096, out_channels=4096)
        self.fc8 = nn.Dense(in_channels=4096, out_channels=num_classes, bias_init=init.Normal(0.02))

        self.dropout = nn.Dropout(keep_prob=0.5)
        self.relu = nn.ReLU()
        self.pad = nn.Pad(paddings=((0, 0), (0, 0), (1, 0), (1, 0)), mode="CONSTANT")

        self.__init_weight()

    def __init_weight(self):
        default_recurisive_init(self)
        self.custom_init_weight()

    def construct(self, x):

        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = x.view(-1, 512 * 2, 7, 7)
        x = self.pad(x)
        x = x.view(-1, 512, 2, 8, 8)
        x = self.pool5(x)

        x = x.view(-1, 8192)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)

        logits = self.fc8(x)

        return logits

    def custom_init_weight(self):
        """
        Init the weight of Conv3d and Dense in the net.
        """
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv3d):
                cell.weight.set_data(init.initializer(
                    KaimingNormal(a=math.sqrt(5), mode='fan_out', nonlinearity='relu'),
                    cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(
                        'zeros', cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(init.initializer(
                    init.Normal(0.01), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(
                        'zeros', cell.bias.shape, cell.bias.dtype))
