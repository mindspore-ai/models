"""
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


import mindspore as ms
import mindspore.nn as nn
import numpy as np

class Basic_Block(nn.Cell):
    """
     Components constituting resnet50
    """
    expansion = 4
    def __init__(self, in_c, out_c, stride=1, downsample=None):
        super(Basic_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(1, 1), stride=1)
        self.bn1 = nn.BatchNorm2d(out_c, moving_mean_init=0, moving_var_init=1)
        self.conv2 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3, 3), stride=stride, \
                               pad_mode='pad', padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv3 = nn.Conv2d(in_channels=out_c, out_channels=out_c*4, kernel_size=(1, 1), stride=1)
        self.bn3 = nn.BatchNorm2d(out_c*4)
        self.relu = nn.ReLU()
        self.down_sample_layer = downsample

    def construct(self, x):
        """

        Args:
            x: tensor

        Returns: tensor

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

        if self.down_sample_layer is not None:
            residual = self.down_sample_layer(residual)
        out = out + residual
        out = self.relu(out)
        return out


class ResNet50(nn.Cell):
    """
    A BoneBack Net of RAS
    """
    def __init__(self):
        super(ResNet50, self).__init__()
        self.in_c = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=2, pad_mode='pad', padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=2, pad_mode='same')

        self.layer1 = self._build_layer(Basic_Block, 64, 3, 1)
        self.layer2 = self._build_layer(Basic_Block, 128, 4, 2)
        self.layer3 = self._build_layer(Basic_Block, 256, 6, 2)
        self.layer4 = self._build_layer(Basic_Block, 512, 3, 2)

        for _, m in self.cells_and_names():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.set_data(ms.Tensor(np.random.normal(0, np.sqrt(2./n), m.weight.data.shape).astype(np.float32)))
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma.set_data(ms.Tensor(np.ones(m.gamma.data.shape, dtype=np.float32)))
                m.beta.set_data(ms.Tensor(np.zeros(m.beta.data.shape, dtype=np.float32)))

    def _build_layer(self, block, out_c, blocks, stride):
        layers = []
        downsample = nn.SequentialCell(nn.Conv2d(self.in_c, out_c*block.expansion, kernel_size=(1, 1), stride=stride),
                                       nn.BatchNorm2d(out_c*4))
        layers.append(block(self.in_c, out_c, stride=stride, downsample=downsample))
        self.in_c = out_c * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_c, out_c))
        return nn.SequentialCell(layers)


    def construct(self, x):
        """

        Args:
            x:

        Returns:
            5 outputs
        """
        out = self.conv1(x)
        out = self.bn1(out)
        x1 = self.relu(out)
        x2 = self.pool(x1)

        x2 = self.layer1(x2)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x1, x2, x3, x4, x5
