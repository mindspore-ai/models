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
import mindspore.nn as nn
import mindspore.ops as ops
from src.conv2d_ops import _conv1x1_valid
from model.basic_layers import ResidualBlock

class AttentionModule_pre(nn.Cell):

    def __init__(self, in_channels, out_channels, size1, size2, size3):
        super(AttentionModule_pre, self).__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)
        self.trunk_branches = nn.SequentialCell([
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)])
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.softmax1_blocks = ResidualBlock(in_channels, out_channels)
        self.skip1_connection_residual_block = ResidualBlock(in_channels, out_channels)
        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.softmax2_blocks = ResidualBlock(in_channels, out_channels)
        self.skip2_connection_residual_block = ResidualBlock(in_channels, out_channels)
        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.softmax3_blocks = nn.SequentialCell([
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)])
        self.interpolation3 = nn.UpsamplingBilinear2d(size=size3)
        self.softmax4_blocks = ResidualBlock(in_channels, out_channels)
        self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)
        self.softmax5_blocks = ResidualBlock(in_channels, out_channels)
        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)
        self.softmax6_blocks = nn.SequentialCell([
            nn.BatchNorm2d(num_features=out_channels, momentum=0.9),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, pad_mode='pad', has_bias=False),
            nn.BatchNorm2d(num_features=out_channels, momentum=0.9),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, pad_mode='pad', has_bias=False),
            nn.Sigmoid()])
        self.last_blocks = ResidualBlock(in_channels, out_channels)

    def construct(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
        out_mpool2 = self.mpool2(out_softmax1)
        out_softmax2 = self.softmax2_blocks(out_mpool2)
        out_skip2_connection = self.skip2_connection_residual_block(out_softmax2)
        out_mpool3 = self.mpool3(out_softmax2)
        out_softmax3 = self.softmax3_blocks(out_mpool3)
        out_interp3 = self.interpolation3(out_softmax3)
        out = out_interp3 + out_skip2_connection
        out_softmax4 = self.softmax4_blocks(out)
        out_interp2 = self.interpolation2(out_softmax4)
        out = out_interp2 + out_skip1_connection
        out_softmax5 = self.softmax5_blocks(out)
        out_interp1 = self.interpolation1(out_softmax5)
        out_softmax6 = self.softmax6_blocks(out_interp1)
        out = (1 + out_softmax6) * out_trunk
        out_last = self.last_blocks(out)
        return out_last

class AttentionModule_stage0(nn.Cell):
    # input size is 112*112
    def __init__(self, in_channels, out_channels, size1=(112, 112),
                 size2=(56, 56), size3=(28, 28), size4=(14, 14)):
        super(AttentionModule_stage0, self).__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)
        self.trunk_branches = nn.SequentialCell([
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)])
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        # 56*56
        self.softmax1_blocks = ResidualBlock(in_channels, out_channels)
        self.skip1_connection_residual_block = ResidualBlock(in_channels, out_channels)
        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        # 28*28
        self.softmax2_blocks = ResidualBlock(in_channels, out_channels)
        self.skip2_connection_residual_block = ResidualBlock(in_channels, out_channels)
        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        # 14*14
        self.softmax3_blocks = ResidualBlock(in_channels, out_channels)
        self.skip3_connection_residual_block = ResidualBlock(in_channels, out_channels)
        self.mpool4 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        # 7*7
        self.softmax4_blocks = nn.SequentialCell([
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)])
        self.interpolation4 = nn.UpsamplingBilinear2d(size=size4)
        self.softmax5_blocks = ResidualBlock(in_channels, out_channels)
        self.interpolation3 = nn.UpsamplingBilinear2d(size=size3)
        self.softmax6_blocks = ResidualBlock(in_channels, out_channels)
        self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)
        self.softmax7_blocks = ResidualBlock(in_channels, out_channels)
        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)
        self.softmax8_blocks = nn.SequentialCell([
            nn.BatchNorm2d(num_features=out_channels, momentum=0.9),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, pad_mode='pad', has_bias=False),
            nn.BatchNorm2d(num_features=out_channels, momentum=0.9),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, pad_mode='pad', has_bias=False),
            nn.Sigmoid()])
        self.last_blocks = ResidualBlock(in_channels, out_channels)

    def construct(self, x):
        # 112*112
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        # 56*56
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
        out_mpool2 = self.mpool2(out_softmax1)
        # 28*28
        out_softmax2 = self.softmax2_blocks(out_mpool2)
        out_skip2_connection = self.skip2_connection_residual_block(out_softmax2)
        out_mpool3 = self.mpool3(out_softmax2)
        # 14*14
        out_softmax3 = self.softmax3_blocks(out_mpool3)
        out_skip3_connection = self.skip3_connection_residual_block(out_softmax3)
        out_mpool4 = self.mpool4(out_softmax3)
        # 7*7
        out_softmax4 = self.softmax4_blocks(out_mpool4)
        out_interp4 = self.interpolation4(out_softmax4) + out_softmax3
        out = out_interp4 + out_skip3_connection
        out_softmax5 = self.softmax5_blocks(out)
        out_interp3 = self.interpolation3(out_softmax5) + out_softmax2
        out = out_interp3 + out_skip2_connection
        out_softmax6 = self.softmax6_blocks(out)
        out_interp2 = self.interpolation2(out_softmax6) + out_softmax1
        out = out_interp2 + out_skip1_connection
        out_softmax7 = self.softmax7_blocks(out)
        out_interp1 = self.interpolation1(out_softmax7) + out_trunk
        out_softmax8 = self.softmax8_blocks(out_interp1)
        out = (1 + out_softmax8) * out_trunk
        out_last = self.last_blocks(out)
        return out_last


class AttentionModule_stage1(nn.Cell):
    # input size is 56*56
    def __init__(self, in_channels, out_channels, size1=(56, 56), size2=(28, 28), size3=(14, 14)):
        super(AttentionModule_stage1, self).__init__()
        self.size1 = size1
        self.size2 = size2
        self.size3 = size3
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)
        self.trunk_branches = nn.SequentialCell([
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)])
        self.pad = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.softmax1_blocks = ResidualBlock(in_channels, out_channels)
        self.skip1_connection_residual_block = ResidualBlock(in_channels, out_channels)
        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.softmax2_blocks = ResidualBlock(in_channels, out_channels)
        self.skip2_connection_residual_block = ResidualBlock(in_channels, out_channels)
        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.softmax3_blocks = nn.SequentialCell([
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)])
        self.interpolation3 = ops.ResizeBilinear(size=size3, align_corners=True)
        self.softmax4_blocks = ResidualBlock(in_channels, out_channels)
        self.interpolation2 = ops.ResizeBilinear(size=size2, align_corners=True)
        self.softmax5_blocks = ResidualBlock(in_channels, out_channels)
        self.interpolation1 = ops.ResizeBilinear(size=size1, align_corners=True)
        self.softmax6_blocks = nn.SequentialCell([
            nn.BatchNorm2d(num_features=out_channels, momentum=0.9),
            nn.ReLU(),
            _conv1x1_valid(in_channels=out_channels, out_channels=out_channels, stride=1),
            nn.BatchNorm2d(num_features=out_channels, momentum=0.9),
            nn.ReLU(),
            _conv1x1_valid(in_channels=out_channels, out_channels=out_channels, stride=1),
            nn.Sigmoid()])
        self.last_blocks = ResidualBlock(in_channels, out_channels)

    def construct(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        opt_pad1 = self.pad(x)
        out_mpool1 = self.mpool1(opt_pad1)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
        opt_pad2 = self.pad(out_softmax1)
        out_mpool2 = self.mpool2(opt_pad2)
        out_softmax2 = self.softmax2_blocks(out_mpool2)
        out_skip2_connection = self.skip2_connection_residual_block(out_softmax2)
        opt_pad2 = self.pad(out_softmax2)
        out_mpool3 = self.mpool3(opt_pad2)
        out_softmax3 = self.softmax3_blocks(out_mpool3)
        out_interp3 = self.interpolation3(out_softmax3) + out_softmax2
        out = out_interp3 + out_skip2_connection
        out_softmax4 = self.softmax4_blocks(out)
        out_interp2 = self.interpolation2(out_softmax4) + out_softmax1
        out = out_interp2 + out_skip1_connection
        out_softmax5 = self.softmax5_blocks(out)
        out_interp1 = self.interpolation1(out_softmax5) + out_trunk
        out_softmax6 = self.softmax6_blocks(out_interp1)
        out = (1 + out_softmax6) * out_trunk
        out_last = self.last_blocks(out)
        return out_last

class AttentionModule_stage2(nn.Cell):
    # input image size is 28*28
    def __init__(self, in_channels, out_channels, size1=(28, 28), size2=(14, 14)):
        super(AttentionModule_stage2, self).__init__()
        self.size1 = size1
        self.size2 = size2
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)
        self.trunk_branches = nn.SequentialCell([
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)])
        self.pad = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.softmax1_blocks = ResidualBlock(in_channels, out_channels)
        self.skip1_connection_residual_block = ResidualBlock(in_channels, out_channels)
        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.softmax2_blocks = nn.SequentialCell([
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)])
        self.interpolation2 = ops.ResizeBilinear(size=size2, align_corners=True)
        self.softmax3_blocks = ResidualBlock(in_channels, out_channels)
        self.interpolation1 = ops.ResizeBilinear(size=size1, align_corners=True)
        self.softmax4_blocks = nn.SequentialCell([
            nn.BatchNorm2d(num_features=out_channels, momentum=0.9),
            nn.ReLU(),
            _conv1x1_valid(in_channels=out_channels, out_channels=out_channels, stride=1),
            nn.BatchNorm2d(num_features=out_channels, momentum=0.9),
            nn.ReLU(),
            _conv1x1_valid(in_channels=out_channels, out_channels=out_channels, stride=1),
            nn.Sigmoid()])
        self.last_blocks = ResidualBlock(in_channels, out_channels)

    def construct(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        opt_pad1 = self.pad(x)
        out_mpool1 = self.mpool1(opt_pad1)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
        opt_pad2 = self.pad(out_softmax1)
        out_mpool2 = self.mpool2(opt_pad2)
        out_softmax2 = self.softmax2_blocks(out_mpool2)
        out_interp2 = self.interpolation2(out_softmax2) + out_softmax1
        out = out_interp2 + out_skip1_connection
        out_softmax3 = self.softmax3_blocks(out)
        out_interp1 = self.interpolation1(out_softmax3) + out_trunk
        out_softmax4 = self.softmax4_blocks(out_interp1)
        out = (1 + out_softmax4) * out_trunk
        out_last = self.last_blocks(out)
        return out_last

class AttentionModule_stage3(nn.Cell):
    # input image size is 14*14
    def __init__(self, in_channels, out_channels, size1=(14, 14)):
        super(AttentionModule_stage3, self).__init__()
        self.size1 = size1
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)
        self.pad = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT")
        self.trunk_branches = nn.SequentialCell([
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)])
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.softmax1_blocks = nn.SequentialCell([
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)])
        self.interpolation1 = ops.ResizeBilinear(size=size1, align_corners=True)
        self.softmax2_blocks = nn.SequentialCell([
            nn.BatchNorm2d(num_features=out_channels, momentum=0.9),
            nn.ReLU(),
            _conv1x1_valid(in_channels=out_channels, out_channels=out_channels, stride=1),
            nn.BatchNorm2d(num_features=out_channels, momentum=0.9),
            nn.ReLU(),
            _conv1x1_valid(in_channels=out_channels, out_channels=out_channels, stride=1),
            nn.Sigmoid()])
        self.last_blocks = ResidualBlock(in_channels, out_channels)

    def construct(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        opt_pad = self.pad(x)
        out_mpool1 = self.mpool1(opt_pad)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_interp1 = self.interpolation1(out_softmax1) + out_trunk
        out_softmax2 = self.softmax2_blocks(out_interp1)
        out = (1 + out_softmax2) * out_trunk
        out_last = self.last_blocks(out)
        return out_last

class AttentionModule_stage1_cifar(nn.Cell):
    # input size is 16*16
    def __init__(self, in_channels, out_channels, size1=(16, 16), size2=(8, 8)):
        super(AttentionModule_stage1_cifar, self).__init__()
        self.size1 = size1
        self.size2 = size2
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)
        self.trunk_branches = nn.SequentialCell([
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)])
        self.pad = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT")
        self.mpool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))  # 8*8
        self.down_residual_blocks1 = ResidualBlock(in_channels, out_channels)
        self.skip1_connection_residual_block = ResidualBlock(in_channels, out_channels)
        self.mpool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))  # 4*4
        self.middle_2r_blocks = nn.SequentialCell([
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)])
        self.interpolation1 = ops.ResizeBilinear(size=size2, align_corners=True)
        self.up_residual_blocks1 = ResidualBlock(in_channels, out_channels)
        self.interpolation2 = ops.ResizeBilinear(size=size1, align_corners=True)
        self.conv1_1_blocks = nn.SequentialCell([
            nn.BatchNorm2d(num_features=out_channels, momentum=0.9),
            nn.ReLU(),
            _conv1x1_valid(in_channels=out_channels, out_channels=out_channels, stride=1),
            nn.BatchNorm2d(num_features=out_channels, momentum=0.9),
            nn.ReLU(),
            _conv1x1_valid(in_channels=out_channels, out_channels=out_channels, stride=1),
            nn.Sigmoid()])
        self.last_blocks = ResidualBlock(in_channels, out_channels)

    def construct(self, x):
        x = self.first_residual_blocks(x)  #x:(64, 128, 32, 32)
        out_trunk = self.trunk_branches(x)
        opt_pad1 = self.pad(x)
        out_mpool1 = self.mpool1(opt_pad1)
        out_down_residual_blocks1 = self.down_residual_blocks1(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_down_residual_blocks1)
        opt_pad2 = self.pad(out_down_residual_blocks1)
        out_mpool2 = self.mpool2(opt_pad2)
        out_middle_2r_blocks = self.middle_2r_blocks(out_mpool2)
        out_interp = self.interpolation1(out_middle_2r_blocks) + out_down_residual_blocks1
        out = out_interp + out_skip1_connection
        out_up_residual_blocks1 = self.up_residual_blocks1(out)
        out_interp2 = self.interpolation2(out_up_residual_blocks1) + out_trunk
        out_conv1_1_blocks = self.conv1_1_blocks(out_interp2)
        out = (1 + out_conv1_1_blocks) * out_trunk
        out_last = self.last_blocks(out)
        return out_last

class AttentionModule_stage2_cifar(nn.Cell):
    # input size is 8*8
    def __init__(self, in_channels, out_channels, size=(8, 8)):
        super(AttentionModule_stage2_cifar, self).__init__()
        self.size = size
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)
        self.trunk_branches = nn.SequentialCell([
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)])
        self.pad = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.mpool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))  # 4*4
        self.middle_2r_blocks = nn.SequentialCell([
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)])
        self.interpolation1 = ops.ResizeBilinear(size=size, align_corners=True)
        self.conv1_1_blocks = nn.SequentialCell([
            nn.BatchNorm2d(num_features=out_channels, momentum=0.9),
            nn.ReLU(),
            _conv1x1_valid(in_channels=out_channels, out_channels=out_channels, stride=1),
            nn.BatchNorm2d(num_features=out_channels, momentum=0.9),
            nn.ReLU(),
            _conv1x1_valid(in_channels=out_channels, out_channels=out_channels, stride=1),
            nn.Sigmoid()])
        self.last_blocks = ResidualBlock(in_channels, out_channels)

    def construct(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        opt_pad1 = self.pad(x)
        out_mpool1 = self.mpool1(opt_pad1)
        out_middle_2r_blocks = self.middle_2r_blocks(out_mpool1)
        out_interp = self.interpolation1(out_middle_2r_blocks) + out_trunk
        out_conv1_1_blocks = self.conv1_1_blocks(out_interp)
        out = (1 + out_conv1_1_blocks) * out_trunk
        out_last = self.last_blocks(out)
        return out_last

class AttentionModule_stage3_cifar(nn.Cell):
    # input size is 4*4
    def __init__(self, in_channels, out_channels, size=(8, 8)):
        super(AttentionModule_stage3_cifar, self).__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)
        self.trunk_branches = nn.SequentialCell([
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)])
        self.middle_2r_blocks = nn.SequentialCell([
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)])
        self.conv1_1_blocks = nn.SequentialCell([
            nn.BatchNorm2d(num_features=out_channels, momentum=0.9),
            nn.ReLU(),
            _conv1x1_valid(in_channels=out_channels, out_channels=out_channels, stride=1),
            nn.BatchNorm2d(num_features=out_channels, momentum=0.9),
            nn.ReLU(),
            _conv1x1_valid(in_channels=out_channels, out_channels=out_channels, stride=1),
            nn.Sigmoid()])
        self.last_blocks = ResidualBlock(in_channels, out_channels)

    def construct(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_middle_2r_blocks = self.middle_2r_blocks(x)
        out_conv1_1_blocks = self.conv1_1_blocks(out_middle_2r_blocks)
        out = (1 + out_conv1_1_blocks) * out_trunk
        out_last = self.last_blocks(out)
        return out_last
