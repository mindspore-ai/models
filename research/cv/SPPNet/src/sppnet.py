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
ZFNet
Original paper: 'Visualizing and Understanding Convolutional Networks,' https://arxiv.org/abs/1311.2901.
input size should be : (3 x 224 x 224)
"""
import mindspore.nn as nn
from mindspore.ops import operations as P
import mindspore.ops as ops
from src.spatial_pyramid_pooling import SpatialPyramidPool


class SppNet(nn.Cell):
    """
    SppNet
    base on zfnet
    """
    def __init__(self, num_classes=10, channel=3, phase='train', include_top=True, train_model="sppnet_single"):
        '''
        :param num_classes: picture classes
        :param channel: obvious is 3
        :param phase: train or test
        :param include_top: True
        '''
        super(SppNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=96, kernel_size=7, stride=2,
                               pad_mode="same", has_bias=True)
        self.conv2 = nn.Conv2d(96, 256, 5, stride=2, pad_mode="same", has_bias=True)
        self.conv3 = nn.Conv2d(256, 384, 3, padding=1, pad_mode="pad", has_bias=True)
        self.conv4 = nn.Conv2d(384, 384, 3, padding=1, pad_mode="pad", has_bias=True)
        self.conv5 = nn.Conv2d(384, 256, 3, padding=1, pad_mode="pad", has_bias=True)

        self.LRN = P.LRN()
        self.relu = P.ReLU()

        self.max_pool2d = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')

        self.train_model = train_model
        self.include_top = include_top

        self.spp_pool_224 = SpatialPyramidPool(13, (6, 3, 2, 1))
        self.spp_pool_180 = SpatialPyramidPool(10, (6, 3, 2, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(in_channels=(6 * 6 + 3 * 3 + 2 * 2 + 1 * 1) * 256, out_channels=4096)
        self.fc2 = nn.Dense(in_channels=4096, out_channels=4096)
        self.fc3 = nn.Dense(in_channels=4096, out_channels=num_classes)

        if self.train_model == "zfnet":
            dropout_ratio = 0.65
            self.fc1 = nn.Dense(in_channels=6 * 6 * 256, out_channels=4096)
        elif self.train_model == "sppnet_single":
            dropout_ratio = 0.59
        else:
            dropout_ratio = 0.58
        if phase == 'test':
            dropout_ratio = 1.0

        self.dropout = nn.Dropout(dropout_ratio)

    def construct(self, x):
        """
        input: x (3 * 224 * 224 or 3 * 180 * 180)

        output: x (1000)
        """
        _, _, H, _ = ops.Shape()(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.LRN(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.LRN(x)
        x = self.max_pool2d(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)

        if self.train_model == "zfnet":
            x = self.max_pool2d(x)
            x = self.flatten(x)
        else:
            if H == 224:
                x = self.spp_pool_224(x)
            elif H == 180:
                x = self.spp_pool_180(x)

        if not self.include_top:
            return x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
