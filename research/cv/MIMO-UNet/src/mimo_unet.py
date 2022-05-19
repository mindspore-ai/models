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
"""
MIMO-UNet architecture
"""

import mindspore
from mindspore import nn
from mindspore import ops

from src.layers import BasicConv, ResBlock


class EBlock(nn.Cell):
    """EBlock"""
    def __init__(self, out_channel, num_res=8):
        super().__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.SequentialCell(*layers)

    def construct(self, x):
        """construct EBlock"""
        return self.layers(x)


class DBlock(nn.Cell):
    """DBlock"""
    def __init__(self, channel, num_res=8):
        super().__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.SequentialCell(*layers)

    def construct(self, x):
        """construct DBlock"""
        return self.layers(x)


class AFF(nn.Cell):
    """AFF"""
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.SequentialCell(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

        self.cat = ops.Stack(axis=1)

    def construct(self, x1, x2, x4):
        """construct AFF"""
        x = ops.Concat(1)([x1, x2, x4])
        return self.conv(x)


class SCM(nn.Cell):
    """SCM"""
    def __init__(self, out_plane):
        super().__init__()
        self.main = nn.SequentialCell(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane-3, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

        self.cat = ops.Stack(axis=1)

    def construct(self, x):
        """construct SCM"""
        y = self.main(x)
        x = ops.Concat(1)([x, y])
        return self.conv(x)


class FAM(nn.Cell):
    """FAM"""
    def __init__(self, channel):
        super().__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def construct(self, x1, x2):
        """construct FAM"""
        x = x1 * x2
        out = x1 + self.merge(x)
        return out


class MIMOUNet(nn.Cell):
    """MIMOUnet"""
    def __init__(self, num_res=8):
        super().__init__()

        base_channel = 32

        self.Encoder = nn.CellList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.CellList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.CellList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.CellList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.CellList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.CellList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2)
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

        self.cat = ops.Stack(axis=1)
        self.nn_interpolate = nn.ResizeBilinear()

    def interpolate(self, x, scale_factor):
        """interpolate"""
        _, _, h, w = x.shape
        h = ops.Cast()(h * scale_factor, mindspore.int32)[0]
        w = ops.Cast()(w * scale_factor, mindspore.int32)[0]
        return self.nn_interpolate(x, size=(h, w))

    def interpolate_upscale(self, x, scale_factor):
        """upscale"""
        _, _, h, w = x.shape
        h = h * scale_factor
        w = w * scale_factor
        return self.nn_interpolate(x, size=(h, w))

    def interpolate_downscale(self, x, scale_factor):
        """downscale"""
        _, _, h, w = x.shape
        h = h // scale_factor
        w = w // scale_factor
        return self.nn_interpolate(x, size=(h, w))

    def construct(self, x):
        """construct MIMOUnet"""
        x_2 = self.interpolate_downscale(x, scale_factor=2)
        x_4 = self.interpolate_downscale(x_2, scale_factor=2)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = []

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = self.interpolate_downscale(res1, scale_factor=2)
        z21 = self.interpolate_upscale(res2, scale_factor=2)
        z42 = self.interpolate_upscale(z, scale_factor=2)
        z41 = self.interpolate_upscale(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_+x_4)

        z = ops.Concat(1)([z, res2])
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = ops.Concat(1)([z, res1])
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z+x)

        return outputs


class MIMOUNetPlus(nn.Cell):
    """MIMOUNetPlus"""
    def __init__(self, num_res=20):
        super().__init__()
        base_channel = 32
        self.Encoder = nn.CellList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.CellList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.CellList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.CellList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.CellList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.CellList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2)
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.1)

        self.cat = ops.Stack(axis=1)
        self.interpolate = nn.ResizeBilinear()

    def construct(self, x):
        """construct MIMOUNetPlus"""
        x_2 = self.interpolate(x, scale_factor=0.5)
        x_4 = self.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = []

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = self.interpolate(res1, scale_factor=0.5)
        z21 = self.interpolate(res2, scale_factor=2)
        z42 = self.interpolate(z, scale_factor=2)
        z41 = self.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        res2 = self.drop2(res2)
        res1 = self.drop1(res1)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_+x_4)

        z = ops.Concat(1)([z, res2])
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = ops.Concat(1)([z, res1])
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z+x)

        return outputs


def build_net(model_name):
    """build network"""
    if model_name == "MIMO-UNetPlus":
        return MIMOUNetPlus()
    if model_name == "MIMO-UNet":
        return MIMOUNet()
    raise ValueError('Wrong Model!\nYou should choose MIMO-UNetPlus or MIMO-UNet.')
