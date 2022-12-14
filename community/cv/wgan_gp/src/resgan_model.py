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
""" Dcgan model """
import mindspore.nn as nn
from mindspore import ops

def get_norm(out_channels, isDisc, mode='wgan-gp'):
    """get norm layer"""
    if isDisc and (mode == 'wgan-gp'):
        return nn.GroupNorm(1, out_channels) # replace layer norm
    return nn.BatchNorm2d(out_channels)

class UpsampleConv(nn.Cell):
    """up sample Conv"""
    def __init__(self, input_dim, output_dim, filter_size, padding, has_bias):
        super(UpsampleConv, self).__init__()

        self.conv = nn.Conv2d(input_dim, output_dim, filter_size, 1, 'pad', padding, has_bias=has_bias)
        self.up = nn.ResizeBilinear()
    def construct(self, x):
        out = self.up(x, scale_factor=2)
        out = self.conv(out)
        return out

class MeanPoolConv(nn.Cell):
    """down sample Conv"""
    def __init__(self, input_dim, output_dim, has_bias):
        super(MeanPoolConv, self).__init__()
        self.conv_down = nn.Conv2d(input_dim, output_dim, 4, 2, 'pad', 0, has_bias=has_bias)

    def construct(self, x):
        out = self.conv_down(x)
        return out

class BottleneckResidualBlock(nn.Cell):
    """BottleneckResidualBlock"""
    def __init__(self, input_dim, output_dim, filter_size, resample='none', isDisc=False): # filter_size=3
        super(BottleneckResidualBlock, self).__init__()
        if resample == 'down':
            self.shortcut = MeanPoolConv(input_dim, output_dim, has_bias=True)
            self.conv_1 = nn.Conv2d(input_dim, input_dim // 2, 1, 1, 'pad', 0, has_bias=False)
            self.conv_2 = MeanPoolConv(input_dim // 2, output_dim // 2, has_bias=False)
            self.conv_3 = nn.Conv2d(output_dim // 2, output_dim, 1, 1, 'pad', 0, has_bias=False)
        elif resample == 'up':
            self.shortcut = UpsampleConv(input_dim, output_dim, filter_size=1, padding=0, has_bias=True)
            self.conv_1 = nn.Conv2d(input_dim, input_dim // 2, 1, 1, 'pad', 0, has_bias=False)
            self.conv_2 = UpsampleConv(input_dim // 2, output_dim // 2, filter_size, padding=1, has_bias=False)
            self.conv_3 = nn.Conv2d(output_dim // 2, output_dim, 1, 1, 'pad', 0, has_bias=False)
        elif resample == 'none':
            self.shortcut = nn.Identity() if output_dim == input_dim else \
                nn.Conv2d(input_dim, output_dim, 1, 1, 'pad', 0, has_bias=True)
            self.conv_1 = nn.Conv2d(input_dim, input_dim // 2, 1, 1, 'pad', 0, has_bias=False)
            self.conv_2 = nn.Conv2d(input_dim // 2, output_dim // 2, filter_size, 1, 'pad', 1, has_bias=False)
            self.conv_3 = nn.Conv2d(output_dim // 2, output_dim, 1, 1, 'pad', 0, has_bias=False)
        else:
            raise Exception('invalid resample value')


        self.main = nn.SequentialCell([nn.ReLU(),
                                       self.conv_1,
                                       nn.ReLU(),
                                       self.conv_2,
                                       nn.ReLU(),
                                       self.conv_3,
                                       get_norm(output_dim, isDisc=isDisc),
                                       ])

    def construct(self, x):
        return 0.3*self.main(x) + self.shortcut(x)

class ResidualBlock(nn.Cell):
    """ResidualBlock"""
    def __init__(self, input_dim, output_dim, filter_size, resample='none', isDisc=False):
        super(ResidualBlock, self).__init__()
        if resample == 'down':
            self.shortcut = MeanPoolConv(input_dim, output_dim, has_bias=True)
            self.conv_1 = nn.Conv2d(input_dim, output_dim, filter_size, 1, 'pad', 1, has_bias=False)
            self.conv_2 = MeanPoolConv(output_dim, output_dim, has_bias=False)
        elif resample == 'up':
            self.shortcut = UpsampleConv(input_dim, output_dim, filter_size=1, padding=0, has_bias=True)
            self.conv_1 = UpsampleConv(input_dim, output_dim, filter_size, padding=1, has_bias=False)
            self.conv_2 = nn.Conv2d(output_dim, output_dim, filter_size, 1, 'pad', 1, has_bias=False)
        elif resample == 'none':
            self.shortcut = nn.Identity() if output_dim == input_dim else \
                nn.Conv2d(input_dim, output_dim, 1, 1, 'pad', 0, has_bias=True)
            self.conv_1 = nn.Conv2d(input_dim, output_dim, filter_size, 1, 'pad', 1, has_bias=False)
            self.conv_2 = nn.Conv2d(output_dim, output_dim, filter_size, 1, 'pad', 1, has_bias=False)
        else:
            raise Exception('invalid resample value')


        self.main = nn.SequentialCell([get_norm(input_dim, isDisc=isDisc),
                                       nn.ReLU(),
                                       self.conv_1,
                                       get_norm(output_dim, isDisc=isDisc),
                                       nn.ReLU(),
                                       self.conv_2,
                                       ])

    def construct(self, x):
        return self.main(x) + self.shortcut(x)


class GoodGenerator(nn.Cell):
    """Generator based on Resnet"""
    def __init__(self, isize, nz, nc, ngf):
        super(GoodGenerator, self).__init__()
        assert isize == 64, "isize has to be 64"

        main = nn.SequentialCell()
        # input is Z, going into a convolution
        main.append(nn.Conv2dTranspose(nz, 8*ngf, 4, 1, 'pad', 0, has_bias=False))

        main.append(ResidualBlock(8 * ngf, 8 * ngf, 3, resample='up'))
        main.append(ResidualBlock(8 * ngf, 4 * ngf, 3, resample='up'))
        main.append(ResidualBlock(4 * ngf, 2 * ngf, 3, resample='up'))
        main.append(ResidualBlock(2 * ngf, 1 * ngf, 3, resample='up'))

        main.append(get_norm(ngf, isDisc=False))
        main.append(nn.ReLU())
        main.append(nn.Conv2d(ngf, nc, 3))
        main.append(nn.Tanh())
        self.main = main

    def construct(self, x):
        """construct"""
        output = self.main(x)
        return output

class GoodDiscriminator(nn.Cell):
    """Discriminator based on Resnet"""
    def __init__(self, isize, nc, ngf):
        super(GoodDiscriminator, self).__init__()
        assert isize == 64, "isize has to be 64"

        main = nn.SequentialCell()
        # input is Z, going into a convolution
        main.append(nn.Conv2d(nc, ngf, 3, 1, 'pad', 1, has_bias=False))

        main.append(ResidualBlock(ngf, 2 * ngf, 3, resample='down', isDisc=True))
        main.append(ResidualBlock(2 * ngf, 4 * ngf, 3, resample='down', isDisc=True))
        main.append(ResidualBlock(4 * ngf, 8 * ngf, 3, resample='down', isDisc=True))
        main.append(ResidualBlock(8 * ngf, 8 * ngf, 3, resample='down', isDisc=True))


        main.append(nn.Conv2d(8 * ngf, 1, 4))

        self.main = main
        self.reduce_mode = ops.ReduceMean()

    def construct(self, x):
        """construct"""
        output = self.main(x)
        output = self.reduce_mode(output)
        return output


class ResnetGenerator(nn.Cell):
    """Generator based on Resnet101"""
    def __init__(self, isize, nz, nc, ngf):
        super(ResnetGenerator, self).__init__()
        assert isize == 64, "isize has to be 64"

        main = nn.SequentialCell()
        # input is Z, going into a convolution
        main.append(nn.Conv2dTranspose(nz, 8 * ngf, 4, 1, 'pad', 0, has_bias=False))

        for _ in range(6):
            main.append(BottleneckResidualBlock(8 * ngf, 8 * ngf, 3, resample='none'))
        main.append(BottleneckResidualBlock(8 * ngf, 4 * ngf, 3, resample='up'))
        for _ in range(6):
            main.append(BottleneckResidualBlock(4 * ngf, 4 * ngf, 3, resample='none'))
        main.append(BottleneckResidualBlock(4 * ngf, 2 * ngf, 3, resample='up'))
        for _ in range(6):
            main.append(BottleneckResidualBlock(2 * ngf, 2 * ngf, 3, resample='none'))
        main.append(BottleneckResidualBlock(2 * ngf, 1 * ngf, 3, resample='up'))
        for _ in range(6):
            main.append(BottleneckResidualBlock(1 * ngf, 1 * ngf, 3, resample='none'))
        main.append(BottleneckResidualBlock(1 * ngf, ngf//2, 3, resample='up'))
        for _ in range(5):
            main.append(BottleneckResidualBlock(ngf//2, ngf//2, 3, resample='none'))

        main.append(nn.Conv2d(ngf//2, nc, 1))
        self.tan = nn.Tanh()
        self.main = main

    def construct(self, x):
        """construct"""
        output = self.main(x)
        output = self.tan(output / 5.)
        return output

class ResnetDiscriminator(nn.Cell):
    """Discriminator based on Resnet101"""
    def __init__(self, isize, nc, ngf):
        super(ResnetDiscriminator, self).__init__()
        assert isize == 64, "isize has to be 64"

        main = nn.SequentialCell()
        # input is Z, going into a convolution
        main.append(nn.Conv2d(nc, ngf // 2, 1, 1, has_bias=True))

        for _ in range(5):
            main.append(BottleneckResidualBlock(ngf // 2, ngf // 2, 3, resample='none'))
        main.append(BottleneckResidualBlock(ngf // 2, ngf, 3, resample='down'))
        for _ in range(6):
            main.append(BottleneckResidualBlock(ngf, ngf, 3, resample='none'))
        main.append(BottleneckResidualBlock(ngf, 2 * ngf, 3, resample='down'))
        for _ in range(6):
            main.append(BottleneckResidualBlock(2 * ngf, 2 * ngf, 3, resample='none'))
        main.append(BottleneckResidualBlock(2 * ngf, 4 * ngf, 3, resample='down'))
        for _ in range(6):
            main.append(BottleneckResidualBlock(4 * ngf, 4 * ngf, 3, resample='none'))
        main.append(BottleneckResidualBlock(4 * ngf, 8 * ngf, 3, resample='down'))
        for _ in range(6):
            main.append(BottleneckResidualBlock(8 * ngf, 8 * ngf, 3, resample='none'))

        main.append(nn.Conv2d(8 * ngf, 1, 4, 1, 'pad', 0, has_bias=True))
        self.main = main
        self.reduce_mode = ops.ReduceMean()

    def construct(self, x):
        """construct"""
        output = self.main(x)
        output = self.reduce_mode(output)
        return output
