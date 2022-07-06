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
"""Discriminator for CTSDG"""

from mindspore import nn
from mindspore import ops

from src.discriminator.spectral_conv import Conv2dNormalized
from src.initializer import KaimingUniform
from src.initializer import UniformBias


class Discriminator(nn.Cell):
    """Discriminator cell"""
    def __init__(self, image_in_channels, edge_in_channels):
        super().__init__()

        self.texture_branch = TextureBranch(in_channels=image_in_channels)
        self.structure_branch = StructureBranch(in_channels=edge_in_channels)
        self.edge_detector = EdgeDetector()

    def construct(self, output, gray_image, real_edge, is_real):
        """construct"""
        texture_pred = self.texture_branch(output)
        fake_edge = self.edge_detector(output)

        if is_real:
            to_concat = real_edge
        else:
            to_concat = fake_edge

        pred = ops.Concat(axis=1)((to_concat, gray_image))
        structure_pred = self.structure_branch(pred)

        return ops.Concat(axis=1)((texture_pred, structure_pred)), fake_edge


class StructureBranch(nn.Cell):
    """Structure branch cell for discriminator"""
    def __init__(self, in_channels=4, use_sigmoid=True):
        super().__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = nn.SequentialCell(
            Conv2dNormalized(
                in_channel=in_channels,
                out_channel=64,
                kernel_size=4,
                stride=2,
                pad_mode='pad',
                pad=1,
                has_bias=False,
            ),
            nn.LeakyReLU(0.2),
        )

        self.conv2 = nn.SequentialCell(
            Conv2dNormalized(
                in_channel=64,
                out_channel=128,
                kernel_size=4,
                stride=2,
                pad_mode='pad',
                pad=1,
                has_bias=False,
            ),
            nn.LeakyReLU(0.2),
        )

        self.conv3 = nn.SequentialCell(
            Conv2dNormalized(
                in_channel=128,
                out_channel=256,
                kernel_size=4,
                stride=2,
                pad_mode='pad',
                pad=1,
                has_bias=False,
            ),
            nn.LeakyReLU(0.2),
        )

        self.conv4 = nn.SequentialCell(
            Conv2dNormalized(
                in_channel=256,
                out_channel=512,
                kernel_size=4,
                stride=1,
                pad_mode='pad',
                pad=1,
                has_bias=False,
            ),
            nn.LeakyReLU(0.2),
        )

        self.conv5 = Conv2dNormalized(
            in_channel=512,
            out_channel=1,
            kernel_size=4,
            stride=1,
            pad_mode='pad',
            pad=1,
            has_bias=False,
        )

    def construct(self, edge):
        """construct"""
        edge_pred = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(edge)))))

        if self.use_sigmoid:
            edge_pred = ops.Sigmoid()(edge_pred)

        return edge_pred


class EdgeDetector(nn.Cell):
    """Edge detector cell for discriminator"""
    def __init__(self, in_channels=3, mid_channels=16, out_channels=1):
        super().__init__()
        self.projection = nn.SequentialCell(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_mode='pad',
                has_bias=True,
                weight_init=KaimingUniform(),
                bias_init=UniformBias([out_channels, in_channels])
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
        )
        self.res_layer = nn.SequentialCell(
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_mode='pad',
                has_bias=True,
                weight_init=KaimingUniform(),
                bias_init=UniformBias([mid_channels, mid_channels])
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_mode='pad',
                has_bias=True,
                weight_init=KaimingUniform(),
                bias_init=UniformBias([mid_channels, mid_channels])
            ),
            nn.BatchNorm2d(mid_channels)
        )
        self.relu = nn.ReLU()
        self.out_layer = nn.SequentialCell(
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=1,
                has_bias=True,
                weight_init=KaimingUniform(),
                bias_init=UniformBias([out_channels, mid_channels])
            ),
            nn.Sigmoid(),
        )

    def construct(self, image):
        """construct"""
        image = self.projection(image)
        edge = self.res_layer(image)
        edge = self.relu(edge + image)
        edge = self.out_layer(edge)

        return edge


class TextureBranch(nn.Cell):
    """Texture branch cell for discriminator"""
    def __init__(
            self,
            in_channels=3,
            use_sigmoid=True,
    ):
        super().__init__()

        self.use_sigmoid = use_sigmoid

        self.conv1 = nn.SequentialCell(
            Conv2dNormalized(
                in_channel=in_channels,
                out_channel=64,
                kernel_size=4,
                stride=2,
                pad_mode='pad',
                pad=1,
                has_bias=False,
            ),
            nn.LeakyReLU(0.2),
        )

        self.conv2 = nn.SequentialCell(
            Conv2dNormalized(
                in_channel=64,
                out_channel=128,
                kernel_size=4,
                stride=2,
                pad_mode='pad',
                pad=1,
                has_bias=False,
            ),
            nn.LeakyReLU(0.2),
        )

        self.conv3 = nn.SequentialCell(
            Conv2dNormalized(
                in_channel=128,
                out_channel=256,
                kernel_size=4,
                stride=2,
                pad_mode='pad',
                pad=1,
                has_bias=False,
            ),
            nn.LeakyReLU(0.2),
        )

        self.conv4 = nn.SequentialCell(
            Conv2dNormalized(
                in_channel=256,
                out_channel=512,
                kernel_size=4,
                stride=1,
                pad_mode='pad',
                pad=1,
                has_bias=False,
            ),
            nn.LeakyReLU(0.2),
        )

        self.conv5 = Conv2dNormalized(
            in_channel=512,
            out_channel=1,
            kernel_size=4,
            stride=1,
            pad_mode='pad',
            pad=1,
            has_bias=False,
        )

    def construct(self, image):
        """construct"""
        image_pred = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(image)))))

        if self.use_sigmoid:
            image_pred = ops.Sigmoid()(image_pred)

        return image_pred
