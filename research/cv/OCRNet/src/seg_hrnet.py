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
"""HRNet definition."""
import logging
import math

import numpy as np
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import ops
from mindspore.common import initializer
from mindspore.nn import BatchNorm2d

BN_MOMENTUM = 0.99
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, pad_mode='pad',
                     padding=1, has_bias=False)


class NoneCell(nn.Cell):
    """Cell doing nothing."""

    def __init__(self):
        super(NoneCell, self).__init__()
        self.name = "NoneCell"

    def construct(self, x):
        """NoneCell construction."""
        return x


class BasicBlock(nn.Cell):
    """BasicBlock definition."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride
        self.add = ops.Add()

    def construct(self, x):
        """BasicBlock construction."""
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.add(out, residual)
        out = self.relu2(out)

        return out


class Bottleneck(nn.Cell):
    """Bottleneck definition."""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, has_bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, pad_mode='pad',
                               padding=1, has_bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               has_bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.add = ops.Add()

    def construct(self, x):
        """Bottleneck construction."""
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.add(out, residual)
        out = self.relu3(out)

        return out


class HighResolutionModule(nn.Cell):
    """HRModule definition."""

    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU()
        self.add = ops.Add()
        self.resize_bilinear = nn.ResizeBilinear()

    def _check_branches(self, num_branches, num_blocks, num_inchannels, num_channels):
        """Check branches."""
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        """Make one branch for parallel layer."""
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.SequentialCell([
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, has_bias=False),
                BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM)
            ])

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        i = 1
        while i < num_blocks[branch_index]:
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))
            i += 1

        return nn.SequentialCell(layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        """Make branches in parallel layer."""
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.CellList(branches)

    def _make_fuse_layers(self):
        """Make fusion layer."""
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.SequentialCell([
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  padding=0,
                                  has_bias=False),
                        BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)]))
                elif j == i:
                    fuse_layer.append(NoneCell())
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.SequentialCell([
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, pad_mode='pad', padding=1, has_bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM)]))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.SequentialCell([
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, pad_mode='pad', padding=1, has_bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU()]))
                    fuse_layer.append(nn.SequentialCell(conv3x3s))
            fuse_layers.append(nn.CellList(fuse_layer))

        return nn.CellList(fuse_layers)

    def get_num_inchannels(self):
        """Get the number of in_channels."""
        return self.num_inchannels

    def construct(self, x):
        """HRModule construction."""
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i, flayer in enumerate(self.fuse_layers):
            y = x[0] if i == 0 else flayer[0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = self.add(y, x[j])
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    t = self.flayer[j](x[j])
                    # t = ops.ResizeNearestNeighbor((height_output, width_output))(t)
                    y = self.add(y, self.resize_bilinear(t, size=(height_output, width_output)))
                else:
                    y = self.add(y, flayer[j](x[j]))
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Cell):
    """HRNet definition."""

    def __init__(self, config, num_classes):
        extra = config.extra
        super(HighResolutionNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, pad_mode='pad', padding=1,
                               has_bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, pad_mode='pad', padding=1,
                               has_bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.stage1_cfg = extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1, self.flag1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2, self.flag2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3, self.flag3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        last_inp_channels = np.int(np.sum(pre_stage_channels))

        self.resize_bilinear = nn.ResizeBilinear()

        self.last_layer = nn.SequentialCell([
            nn.Conv2d(in_channels=last_inp_channels,
                      out_channels=last_inp_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      has_bias=True),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(),
            nn.Conv2d(in_channels=last_inp_channels,
                      out_channels=num_classes,
                      kernel_size=extra.FINAL_CONV_KERNEL,
                      stride=1, pad_mode='pad',
                      padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0,
                      has_bias=True)
        ])

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        """Make a transition layer between different stages."""
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        flag = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.SequentialCell([
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  pad_mode='pad',
                                  padding=1,
                                  has_bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU()]))
                    flag.append("ops")
                else:
                    transition_layers.append(NoneCell())
                    flag.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.SequentialCell([
                        nn.Conv2d(inchannels, outchannels, 3, 2, pad_mode='pad', padding=1, has_bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU()]))
                transition_layers.append(nn.SequentialCell(conv3x3s))
                flag.append("ops")

        return nn.CellList(transition_layers), flag

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        """Make the first stage."""
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.SequentialCell([
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, has_bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            ])

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        i = 1
        while i < blocks:
            layers.append(block(inplanes, planes))
            i += 1

        return nn.SequentialCell(layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        """Make a stage."""
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()
            self.concat = ops.Concat(axis=1)

        return nn.SequentialCell(modules), num_inchannels

    def construct(self, x):
        """HRNet construction."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.flag1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.flag2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.flag3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        out1, out2, out3, out4 = x
        h, w = ops.Shape()(out1)[2:]
        x1 = ops.Cast()(out1, mstype.float32)
        x2 = self.resize_bilinear(out2, size=(h, w))
        x3 = self.resize_bilinear(out3, size=(h, w))
        x4 = self.resize_bilinear(out4, size=(h, w))

        x = self.concat((x1, x2, x3, x4))

        x = self.last_layer(x)

        return x


def get_seg_model(cfg, num_classes):
    """Create HRNet object, and initialize it by initializer or checkpoint."""
    net = HighResolutionNet(cfg, num_classes)

    for _, cell in net.cells_and_names():
        if isinstance(cell, nn.Conv2d):
            cell.weight.set_data(initializer.initializer(initializer.HeUniform(),
                                                         cell.weight.shape,
                                                         cell.weight.dtype))
            if cell.has_bias:
                cell.bias.set_data(get_conv_bias(cell))
        elif isinstance(cell, BatchNorm2d):
            cell.gamma.set_data(initializer.initializer(1,
                                                        cell.gamma.shape,
                                                        cell.gamma.dtype))
            cell.beta.set_data(initializer.initializer(0,
                                                       cell.beta.shape,
                                                       cell.beta.dtype))

    return net


def calculate_fan_in_and_fan_out(shape):
    """
    calculate fan_in and fan_out

    Args:
        shape (tuple): input shape.

    Returns:
        Tuple, a tuple with two elements, the first element is `n_in` and the second element is `n_out`.
    """
    dimensions = len(shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    if dimensions == 2:
        fan_in = shape[1]
        fan_out = shape[0]
    else:
        num_input_fmaps = shape[1]
        num_output_fmaps = shape[0]
        receptive_field_size = 1
        if dimensions > 2:
            receptive_field_size = shape[2] * shape[3]
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def get_conv_bias(cell):
    """Bias initializer for conv."""
    weight = initializer.initializer(initializer.HeUniform(negative_slope=math.sqrt(5)),
                                     cell.weight.shape, cell.weight.dtype).to_tensor()
    fan_in, _ = calculate_fan_in_and_fan_out(weight.shape)
    bound = 1 / math.sqrt(fan_in)
    return initializer.initializer(initializer.Uniform(scale=bound),
                                   cell.bias.shape, cell.bias.dtype)
