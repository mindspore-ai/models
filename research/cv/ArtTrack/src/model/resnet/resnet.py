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

from mindspore import nn
from mindspore import ops


class Subsample(nn.Cell):
    """Subsamples the input along the spatial dimensions.
    Args:
        factor: The subsampling factor.
    Returns:
        output: A `Tensor` of size [batch, channels, height_out, width_out] with the
        input, either intact (if factor == 1) or subsampled (if factor > 1).
    """

    def __init__(self, factor):
        super(Subsample, self).__init__()
        self.factor = factor
        self.max_pool2d = nn.MaxPool2d(1, self.factor)

    def construct(self, inputs):
        if self.factor == 1:
            return inputs

        return self.max_pool2d(inputs)


class Conv2dSame(nn.Cell):
    """
    stride 2-D convolution with 'SAME' padding.
    When stride > 1, then we do explicit zero-padding, followed by conv2d with
    'VALID' padding.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, rate=1):
        """
        Args:
            in_channels: in channels
            out_channels: out channels
            kernel_size: An int with the kernel_size of the filters.
            stride: An integer, the output stride.
            rate: An integer, rate for atrous convolution.
        """
        super(Conv2dSame, self).__init__()
        self.in_channels = in_channels
        self.num_outputs = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.rate = rate

        self.kernel_size_effective = self.kernel_size + (self.kernel_size - 1) * (self.rate - 1)
        self.pad_total = self.kernel_size_effective - 1
        self.pad_beg = self.pad_total // 2
        self.pad_end = self.pad_total - self.pad_beg

        if self.stride == 1:
            self.conv2d = nn.Conv2dBnAct(in_channels, self.num_outputs, self.kernel_size,
                                         stride=self.stride, dilation=self.rate, pad_mode='same', has_bn=True,
                                         activation="relu")
        else:
            self.pad = nn.Pad(((0, 0), (0, 0), (self.pad_beg, self.pad_end), (self.pad_beg, self.pad_end)))
            self.conv2d = nn.SequentialCell([self.pad, nn.Conv2dBnAct(in_channels, self.num_outputs, self.kernel_size,
                                                                      stride=self.stride, dilation=self.rate,
                                                                      pad_mode='valid', has_bn=True,
                                                                      activation="relu")])

    def construct(self, inputs):
        """
        Args:
            inputs: A 4-D tensor of size [batch, channels, height_out, width_out].
        Returns:
            output: A 4-D tensor of size [batch, channels, height_out, width_out] with
              the convolution output.
        """
        return self.conv2d(inputs)


class Bottleneck(nn.Cell):
    """
    Bottleneck residual unit variant with BN after convolutions.
    """
    expansion = 4

    def __init__(self,
                 in_channel,
                 out_channel,
                 stride,
                 rate=1):
        """
        Args:
            in_channel: in channel.
            out_channel: out channel.
            stride: The ResNet unit's stride. Determines the amount of downsampling of
              the units output compared to its input.
            rate: An integer, rate for atrous convolution.
        """
        super(Bottleneck, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.depth = out_channel // 4
        self.stride = stride
        self.rate = rate
        if self.out_channel == self.in_channel:
            self.conv2d_shortcut = Subsample(self.stride)
        else:
            self.conv2d_shortcut = nn.Conv2dBnAct(self.in_channel, self.out_channel, 1, stride=self.stride, has_bn=True)

        self.relu = nn.ReLU()

        self.conv2d1 = nn.Conv2dBnAct(self.in_channel, self.depth, 1, stride=1, activation="relu", has_bn=True)
        self.conv2d2 = Conv2dSame(self.depth, self.depth, 3, self.stride, self.rate)
        self.conv2d3 = nn.Conv2dBnAct(self.depth, self.out_channel, 1, stride=1, has_bn=True)

    def construct(self, inputs):
        """
        Args:
            inputs: A tensor of size [batch, channels, height, width].
        Returns:
            The ResNet unit's output.
        """
        shortcut = self.conv2d_shortcut(inputs)

        residual = self.conv2d1(inputs)
        residual = self.conv2d2(residual)
        residual = self.conv2d3(residual)

        output = self.relu(shortcut + residual)
        return output


class Block(nn.Cell):
    """
    Resnet block consisted of Bottlenecks
    """

    def __init__(self, units, intermediate=None):
        """
        Args:
            units: Bottlenecks
            intermediate: if is not None, the result of  corresponding Bottlenecks  will be returned
            as a intermediate value
        """
        super(Block, self).__init__()
        self.units = units
        self.units_cells = nn.CellList(units)
        self.intermediate = intermediate

    def construct(self, feature, intermediate=None):
        """
        Args:
            feature: feature
            intermediate: intermediate

        Return:
            (feature, ) or (feature, intermediate)
        """
        out = feature
        out_intermediate = intermediate
        for i, cell in enumerate(self.units_cells, 1):
            out = cell(out)
            if self.intermediate is not None and self.intermediate == i:
                out_intermediate = out
        if out_intermediate is None:
            r = (out,)
        else:
            r = (out, out_intermediate)
        return r


class Layer(nn.Cell):
    """
    Resnet Layer consisted of Block.
    """
    current_stride = 1
    rate = 1

    def __init__(self, in_channel, blocks, output_stride=None, unit_class=Bottleneck, block_class=Block):
        """
        Args:
            in_channel: in channel
            blocks: blocks config. should be generated by _make_block
            output_stride: If None, then the output will be computed at the nominal
                network stride. If output_stride is not None, it specifies the requested
                ratio of input to output spatial resolution.
            unit_class: class of unit. options: Bottleneck
            block_class: class of block. options: Block
        """
        super(Layer, self).__init__()
        self.unit_class = unit_class
        self.block_class = block_class
        self.blocks = blocks
        self.output_stride = output_stride
        self.in_channel = in_channel
        self.blocks_cell = []
        self.last_out_channel = self.in_channel
        self.intermediate_block = 3
        self.intermediate_unit = 12
        for i, block in enumerate(blocks, 1):
            units = []
            for _, unit in enumerate(block, 1):
                if self.output_stride is not None and self.current_stride > self.output_stride:
                    raise ValueError('The target output_stride cannot be reached.')

                if self.output_stride is not None and self.current_stride == self.output_stride:
                    net = self.unit_class(rate=self.rate, in_channel=self.last_out_channel, **dict(unit, stride=1))
                    self.last_out_channel = unit['out_channel']
                    self.rate *= unit.get('stride', 1)

                else:
                    net = self.unit_class(rate=1, in_channel=self.last_out_channel, **unit)
                    self.last_out_channel = unit['out_channel']
                    self.current_stride *= unit.get('stride', 1)
                units.append(net)
            self.blocks_cell.append(
                self.block_class(units, None if i != self.intermediate_block else self.intermediate_unit))
        if self.output_stride is not None and self.current_stride != self.output_stride:
            raise ValueError('The target output_stride cannot be reached.')
        self.blocks_cell = nn.CellList(self.blocks_cell)
        self.out_channel = self.last_out_channel

    def construct(self, inputs):
        """
        Args:
            inputs: feature
        Return:
            block result
        """
        out = (inputs, None)
        for cell in self.blocks_cell:
            out = cell(*out)
        return out


class ResNet(nn.Cell):
    """
    Resnet
    """

    def __init__(self,
                 blocks,
                 in_channels,
                 num_classes=None,
                 global_pool=True,
                 output_stride=None,
                 include_root_block=True,
                 ):
        """
        Args:
            blocks: blocks config. should be generated by _make_block
            in_channels: in channels
            num_classes: the number of classes
            global_pool: If True, we perform global average pooling before computing the
                logits. Set to True for image classification, False for dense prediction.
            output_stride: If None, then the output will be computed at the nominal
                network stride. If output_stride is not None, it specifies the requested
                ratio of input to output spatial resolution.
            include_root_block: include_root_block: If True, include the initial convolution followed by
                max-pooling, if False excludes it.
        """
        super(ResNet, self).__init__()
        self.blocks = blocks
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.output_stride = output_stride
        self.include_root_block = include_root_block

        self.conv2d_same = Conv2dSame(self.in_channels, 64, 7, 2)
        self.max_pool2d = nn.MaxPool2d(3, 2, pad_mode='same')

        self.reduce_mean = ops.ReduceMean(True)

        if self.include_root_block:
            if self.output_stride is not None:
                if self.output_stride % 4 != 0:
                    raise ValueError('The output_stride needs to be a multiple of 4.')
                self.output_stride /= 4
            self.layer = Layer(64, self.blocks, self.output_stride)
        else:
            self.layer = Layer(self.in_channels, self.blocks, self.output_stride)
        if self.num_classes is not None:
            self.conv2d1 = nn.Conv2d(self.blocks[-1][-1]['out_channel'], self.num_classes, 1)

    def construct(self, inputs):
        """
        Args:
            inputs: inputs
        Return:
            (result, intermediate)
        """
        net = inputs
        if self.include_root_block:
            net = self.conv2d_same(net)
            net = self.max_pool2d(net)
        net, intermediate = self.layer(net)
        if self.global_pool:
            # Global average pooling.
            net = self.reduce_mean(net, [2, 3])
        if self.num_classes is not None:
            net = self.conv2d1(net)
        return net, intermediate
