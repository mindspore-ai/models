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
"""model define"""
from mindspore import nn
import mindspore.ops.operations as P
from mindspore.common import initializer as init


def init_weights(net, init_type='he', init_gain=0.02):
    """
    Initialize network weights.

    Args:
        net (Cell): Network to be initialized
        init_type (str): The name of an initialization method: normal | xavier.
        init_gain (float): Gain factor for normal and xavier.

    Returns:
        init_type(str): init_type, 'normal', 'xavier', 'he', 'constant'
    """
    for _, cell in net.cells_and_names():
        if isinstance(cell, (nn.Conv2d, nn.Conv2dTranspose)):
            if init_type == 'normal':
                cell.weight.set_data(init.initializer(init.Normal(init_gain), cell.weight.shape))
            elif init_type == 'xavier':
                cell.weight.set_data(init.initializer(init.XavierUniform(init_gain), cell.weight.shape))
            elif init_type == 'he':
                cell.weight.set_data(init.initializer(init.HeUniform(init_gain), cell.weight.shape))
            elif init_type == 'constant':
                cell.weight.set_data(init.initializer(0.001, cell.weight.shape))
            else:
                # print(init_type)
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif isinstance(cell, nn.BatchNorm2d):
            cell.gamma.set_data(init.initializer('ones', cell.gamma.shape))
            cell.beta.set_data(init.initializer('zeros', cell.beta.shape))
        elif isinstance(cell, nn.Dense):
            if init_type == 'normal':
                cell.weight.set_data(init.initializer(init.Normal(init_gain), cell.weight.shape))
            elif init_type == 'xavier':
                cell.weight.set_data(init.initializer(init.XavierUniform(init_gain), cell.weight.shape))
            elif init_type == 'he':
                cell.weight.set_data(init.initializer(init.HeUniform(init_gain), cell.weight.shape))
            elif init_type == 'constant':
                cell.weight.set_data(init.initializer(0.001, cell.weight.shape))
            cell.bias.set_data(init.initializer(0.001, cell.bias.shape))
    return init_type

class Generator(nn.Cell):
    """
    Generator network
    """

    def __init__(self, noise_channel=100, img_channels=1, classes_num=10, embed_size=100, features_d=64,
                 norm_mode='group', auto_prefix=True):
        super(Generator, self).__init__(auto_prefix=auto_prefix)
        self.concat = P.Concat(axis=1)
        self.embed = nn.Embedding(classes_num, embed_size)
        self.unsqueeze = P.ExpandDims()
        self.has_bias = True
        self.norm_mode = norm_mode
        if norm_mode is not None:
            self.has_bias = False
        self.net = nn.SequentialCell([
            self._block(noise_channel + embed_size, features_d * 4, 4, 1, pad_mode='valid', padding=0),
            self._block(features_d * 4, features_d * 2, 4, 2, pad_mode='pad', padding=1),
            self._block(features_d * 2, features_d, 4, 2, pad_mode='pad', padding=1),
            nn.Conv2dTranspose(features_d, img_channels, 4, 2, pad_mode='pad', padding=1),
            nn.Sigmoid()
        ])

    def _get_norm(self, out_channels):  # 'instance', 'group'
        if self.norm_mode == 'instance':
            norm = nn.InstanceNorm2d(out_channels)
        elif self.norm_mode == 'batch':
            norm = nn.BatchNorm2d(out_channels)
        elif self.norm_mode == 'group':
            norm = nn.GroupNorm(1, out_channels)
        else:
            raise Exception("Invalid norm_mode!", self.norm_mode)
        return norm

    def _block(self, in_channels, out_channels, kernel_size, stride, pad_mode, padding):
        return nn.SequentialCell([
            nn.Conv2dTranspose(in_channels, out_channels, kernel_size, stride, pad_mode, padding,
                               has_bias=self.has_bias),
            self._get_norm(out_channels),
            nn.ReLU()
        ])

    def construct(self, x, labels):
        """
        construct

        Args:
            x(Tensor): noise
            labels(Tensor): labels

        Returns:
            x: net output
        """
        embedding = self.embed(labels)
        x = self.concat((x, embedding))
        x = self.unsqueeze(self.unsqueeze(x, 2), 2)
        x = self.net(x)

        return x



class Discriminator(nn.Cell):
    """Discriminator network"""
    def __init__(self, img_channels=1, classes_num=10, embed_size=32 * 32, features_d=64, img_size=32,
                 norm_mode='group', auto_prefix=True):
        super(Discriminator, self).__init__(auto_prefix=auto_prefix)
        self.img_size = img_size
        self.has_bias = True
        self.norm_mode = norm_mode
        if norm_mode is not None:
            self.has_bias = False
        self.net = nn.SequentialCell([
            self._block(img_channels + 1, features_d, 4, 2, pad_mode='pad', padding=1),
            self._block(features_d, features_d * 2, 4, 2, pad_mode='pad', padding=1),
            self._block(features_d * 2, features_d * 4, 4, 2, pad_mode='pad', padding=1),
            nn.Conv2d(features_d * 4, 1, 4, 2, pad_mode='valid'),
            nn.Sigmoid(),
        ])
        self.concat = P.Concat(axis=1)
        self.embed = nn.Embedding(classes_num, embed_size)

    def _get_norm(self, out_channels):  # 'instance', 'group'
        if self.norm_mode == 'instance':
            norm = nn.InstanceNorm2d(out_channels)
        elif self.norm_mode == 'batch':
            norm = nn.BatchNorm2d(out_channels)
        elif self.norm_mode == 'group':
            norm = nn.GroupNorm(1, out_channels)
        else:
            raise Exception("Invalid norm_mode!", self.norm_mode)
        return norm

    def _block(self, in_channels, out_channels, kernel_size, stride, pad_mode, padding):
        return nn.SequentialCell([
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_mode, padding, has_bias=self.has_bias),
            self._get_norm(out_channels),
            nn.LeakyReLU()
        ])

    def construct(self, x, label):
        """
        construct

        Args:
            x(Tensor): noise
            labels(Tensor): labels

        Returns:
            x: net output
        """
        embedding = self.embed(label).reshape(label.shape[0], 1, self.img_size, self.img_size)
        x = self.concat((x, embedding))
        x = self.net(x)
        return x
