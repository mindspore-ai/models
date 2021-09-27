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
The finetune network,svm network and bbox regression network of rcnn.
"""
from mindspore import nn
from mindspore.ops import operations as P

from src.common.logger import Logger


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, pad_mode="valid", has_bias=True):
    """

    :param in_channels: in_channels
    :param out_channels: out_channels
    :param kernel_size: kernel_size
    :param stride: stride
    :param padding: padding
    :param pad_mode: pad_mode
    :param has_bias: has_bias
    :return: conv
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                     has_bias=has_bias, pad_mode=pad_mode)


def fc_with_initialize(input_channels, out_channels, has_bias=True):
    """

    :param input_channels: input_channels
    :param out_channels: out_channels
    :param has_bias: has_bias
    :return: fc_with_initialize
    """
    return nn.Dense(input_channels, out_channels, has_bias=has_bias)


class AlexNet_backbone(nn.Cell):
    """
    AlexNet_backbone
    """

    def __init__(self, channel=3, phase='train', include_top=True):
        super(AlexNet_backbone, self).__init__()
        self.conv1 = conv(channel, 64, 11, stride=4, pad_mode="same", has_bias=True)
        self.conv2 = conv(64, 128, 5, pad_mode="same", has_bias=True)
        self.conv3 = conv(128, 192, 3, pad_mode="same", has_bias=True)
        self.conv4 = conv(192, 256, 3, pad_mode="same", has_bias=True)
        self.conv5 = conv(256, 256, 3, pad_mode="same", has_bias=True)
        self.relu = P.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')
        self.include_top = include_top
        if self.include_top:
            dropout_ratio = 0.65
            if phase == 'test':
                dropout_ratio = 1.0
            self.flatten = nn.Flatten()
            self.fc1 = fc_with_initialize(6 * 6 * 256, 4096)
            self.fc2 = fc_with_initialize(4096, 4096)
            self.dropout = nn.Dropout(dropout_ratio)

    def construct(self, x):
        """

        :param x: input
        :return: output
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        if not self.include_top:
            return x
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class AlexNetCombine(nn.Cell):
    """
    AlexNetCombine
    """

    def __init__(self, class_num=21, phase='train'):
        super().__init__(auto_prefix=False)
        self.backbone = AlexNet_backbone(phase=phase)
        self.head = fc_with_initialize(4096, class_num)
        Logger().info("The network has initialized completely,the number of categories:%s" % class_num)
        Logger().debug(self)

    def construct(self, x):
        """

        :param x: input
        :return: output
        """
        x = self.backbone(x)
        c1 = self.head(x)
        return c1


class BBoxNet(nn.Cell):
    """
    BBoxNet
    """

    def __init__(self):
        super().__init__(auto_prefix=False)
        self.backbone = AlexNet_backbone()
        self.head = fc_with_initialize(4096, 80)
        self.matmul = nn.MatMul()

    def construct(self, x, y):
        """

        :param x: input x
        :param y: input y
        :return: output
        """
        x = self.backbone(x)
        x = self.head(x)
        x = x.view((x.shape[0], 4, 20))
        y = y.reshape(y.shape[0], 20, 1)
        x = self.matmul(x, y)
        return x.squeeze(2)
