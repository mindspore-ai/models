# Copyright 2022 Huawei Technologies Co., Ltd
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
"""HiFaceGAN generator"""
import mindspore.nn as nn
import mindspore.ops as ops

from src.model.architecture import SPADEResnetBlock


class SimplifiedLIP(nn.Cell):
    """Local Importance-Based Pooling"""

    def __init__(self, channels):
        super().__init__()

        self.logit = nn.SequentialCell([
            nn.Conv2d(channels, channels, kernel_size=3, pad_mode='same'),
            nn.InstanceNorm2d(channels, affine=True),
            nn.Sigmoid()
        ])
        self.coeff = 12.0

        self.pad_w = ops.Pad(((0, 0), (0, 0), (1, 1), (1, 1)))
        self.pad_xw = ops.Pad(((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avg_pool_w = nn.AvgPool2d(3, 2)
        self.avg_pool_xw = nn.AvgPool2d(3, 2)

    def lip2d(self, x, logit):
        """Lip2d"""
        weight = ops.Exp()(logit)
        avg_with_weights = self.avg_pool_xw(self.pad_xw(x * weight))
        norm = self.avg_pool_w(self.pad_w(weight))
        return avg_with_weights / norm

    def construct(self, x):
        """Feed forward"""
        return self.lip2d(x, self.coeff * self.logit(x))


class Upsample(nn.Cell):
    """Upsample class"""

    def __init__(self, scale_factor=2):
        super().__init__()
        self._scale_factor = int(scale_factor)

    def construct(self, x):
        """Feed forward"""
        _, _, h, w = x.shape
        return ops.ResizeNearestNeighbor((self._scale_factor * h, self._scale_factor * w))(x)


class ContentAdaptiveSuppressor(nn.Cell):
    """Content adaptive suppressor"""

    def __init__(self, n2xdown, ngf, input_nc):
        super().__init__()
        self.max_ratio = 16
        kw = 3

        self.head = nn.SequentialCell([
            nn.Conv2d(input_nc, ngf, kw, pad_mode='same'),
            nn.InstanceNorm2d(ngf, affine=False),
            nn.ReLU(),
        ])

        self.encoder = nn.CellList([])
        cur_ratio = 1
        for i in range(n2xdown):
            next_ratio = min(cur_ratio * 2, self.max_ratio)
            model = [
                SimplifiedLIP(ngf * cur_ratio),
                nn.Conv2d(ngf * cur_ratio, ngf * next_ratio, kw, pad_mode='same', has_bias=True),
                nn.InstanceNorm2d(ngf * next_ratio, affine=False),
            ]
            cur_ratio = next_ratio
            if i < n2xdown - 1:
                model += [nn.ReLU()]
            self.encoder.append(nn.SequentialCell(*model))

    def construct(self, x):
        """Feed forward"""
        result = [self.head(x)]

        for cell in self.encoder:
            result = [cell(result[0])] + result

        return result


class HiFaceGANGenerator(nn.Cell):
    """HiFaceGAN generator"""

    def __init__(self, ngf, input_nc):
        super().__init__()
        self.scale_ratio = 5
        self.fc = nn.Conv2d(input_nc, 16 * ngf, kernel_size=3, pad_mode='same',
                            has_bias=True)

        self.head_0 = SPADEResnetBlock(16 * ngf, 16 * ngf, 16 * ngf)
        self.G_middle_0 = SPADEResnetBlock(16 * ngf, 16 * ngf, 16 * ngf)
        self.G_middle_1 = SPADEResnetBlock(16 * ngf, 16 * ngf, 16 * ngf)

        self.spade_resnet_block = nn.CellList([
            SPADEResnetBlock(16 * ngf, 8 * ngf, 8 * ngf),
            SPADEResnetBlock(8 * ngf, 4 * ngf, 4 * ngf),
            SPADEResnetBlock(4 * ngf, 2 * ngf, 2 * ngf),
            SPADEResnetBlock(2 * ngf, 1 * ngf, 1 * ngf)
        ])
        self.to_rgbs = nn.Conv2d(1 * ngf, 3, kernel_size=3, pad_mode='same', has_bias=True)

        self.nested_encode = ContentAdaptiveSuppressor(self.scale_ratio, ngf, input_nc)
        self.relu = nn.LeakyReLU(0.2)
        self.tanh = ops.Tanh()

    def encode(self, inp):
        """Resize input"""
        _, _, h, w = inp.shape
        sh, sw = h // 2 ** self.scale_ratio, w // 2 ** self.scale_ratio
        interpolate = ops.ResizeNearestNeighbor(size=(sh, sw))
        x = interpolate(inp)
        return self.fc(x)

    def construct(self, inp):
        """Feed forward"""
        xs = self.nested_encode(inp)
        x = self.encode(inp)
        x = self.head_0(x, xs[0])
        x = Upsample()(x)

        x = self.G_middle_0(x, xs[1])
        x = self.G_middle_1(x, xs[1])

        for i in range(self.scale_ratio - 1):
            x = self.spade_resnet_block[i](Upsample()(x), xs[i + 2])

        x = self.to_rgbs(self.relu(x))
        return self.tanh(x)
