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

"""Discriminator network"""

from mindspore import nn

from src.model.base_network import ConvBlock, DenseBlock
from src.util.utils import init_weights


class Discriminator(nn.Cell):
    """Structure of Discriminator network"""

    def __init__(self, num_channels, base_filter, image_size):
        super(Discriminator, self).__init__()
        self.image_size = image_size

        self.conv_blocks = nn.SequentialCell(
            # 3 64
            ConvBlock(num_channels, base_filter, 3, 1, 1, activation='lrelu'),
            # 64 64
            ConvBlock(base_filter, base_filter, 3, 2, 1, activation='lrelu', norm='batch'),
            # 64 128
            ConvBlock(base_filter, base_filter * 2, 3, 1, 1, activation='lrelu', norm='batch'),
            # 128 128
            ConvBlock(base_filter * 2, base_filter * 2, 3, 2, 1, activation='lrelu', norm='batch'),
            # 128 256
            ConvBlock(base_filter * 2, base_filter * 4, 3, 1, 1, activation='lrelu', norm='batch'),
            # 256 256
            ConvBlock(base_filter * 4, base_filter * 4, 3, 2, 1, activation='lrelu', norm='batch'),
            # 256 512
            ConvBlock(base_filter * 4, base_filter * 8, 3, 1, 1, activation='lrelu', norm='batch'),
            # 512 512
            ConvBlock(base_filter * 8, base_filter * 8, 3, 2, 1, activation='lrelu', norm='batch'),
        )
        self.flatten = nn.Flatten()
        self.dense_layers = nn.SequentialCell(
            DenseBlock(base_filter * 8 * image_size // 16 * image_size // 16, base_filter * 16, activation='lrelu',
                       norm=None),
            DenseBlock(base_filter * 16, 1, activation='sigmoid', norm=None)
        )

    def construct(self, x):
        """discriminator compute graph
        Args:
            x(Tensor): low resolution image
        Outputs:
            Tensor: [0,1]
        """
        out = self.conv_blocks(x)
        out = self.flatten(out)
        out = self.dense_layers(out)
        return out


def get_discriminator(num_channels, base_filter, image_size, init_gain):
    """Return discriminator by args."""
    net = Discriminator(num_channels, base_filter, image_size)
    init_weights(net, 'KaimingNormal', init_gain)
    return net
