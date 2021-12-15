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

"""Generate feature maps for ssd."""

import mindspore.nn as nn
from mindspore.ops import Pad
from src import inception_v2


def get_depth_fn(depth_multipler, min_depth):
    """Builds a callable to compute depth (output channels) of conv filters.

    Args:
        depth_multiplier: a multiplier for the nominal depth.
        min_depth: a lower bound on the depth of filters.

    Returns:
        A callable that takes in a nominal depth and returns the depth to use.
    """
    def multiply_depth(depth):
        """Limit minimum of depth."""
        new_depth = int(depth * depth_multipler)
        return max(new_depth, min_depth)
    return multiply_depth


class NoneCell(nn.Cell):
    """
    Define noneCell
    """

    def construct(self, inputs):
        """Construct a forward graph"""
        return inputs


class FixedPadding(nn.Cell):
    """
    Fix pad
    """
    def __init__(self, kernel_size, rate=1):
        """TensorFlow fixed_pad.

        Args:
            kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                        Should be a positive integer.
            rate: An integer, rate for atrous convolution.

        Returns:
            padded_inputs: A tensor of size [batch, height_out, width_out, channels] with the
                        input, either intact (if kernel_size == 1) or padded (if kernel_size > 1).
        """
        super(FixedPadding, self).__init__()
        self.kernel_size = kernel_size
        self.rate = rate
        kernel_size_effective = self.kernel_size + (self.kernel_size - 1) * (self.rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        self.pad = Pad(((0, 0), (0, 0), (pad_beg, pad_end), (pad_beg, pad_end)))

    def construct(self, inputs):
        padded_inputs = self.pad(inputs)
        return padded_inputs


class AdditionalFeatureMapGenerator(nn.Cell):
    """
    Generate feature layers outside the backbone network
    """
    def __init__(self, insert_1x1_conv, use_explicit_padding,
                 use_depthwise, pool_residual, pre_channels, layer_depth,
                 depth_multiplier, min_depth, conv_kernel_size):
        """Create operation cell sequence when `from_layer` is an empty str.

        Args:
            insert_1x1_conv (bool):
            use_explicit_padding (bool):
            use_depthwise (bool):
            pool_residual (bool):
            pre_channels (int): Depth of the previous feature map in feature map list.
            layer_depth (int): Layer depth in `feature_map_layout`.
            depth_multiplier ():
            min_depth (int): Minimum depth of outputs.
            conv_kernel_size (int): Kernel size of conv2d and fixed_padding.

        Inputs:
            pre_layer (tensor): Previous feature map.

        Outputs:
            feature_map (tensor): A new feature map.
        """
        super(AdditionalFeatureMapGenerator, self).__init__()
        depth_fn = get_depth_fn(depth_multiplier, min_depth)

        self.channels = depth_fn(layer_depth)
        self.insert_1x1_conv = insert_1x1_conv
        self.use_explicit_padding = use_explicit_padding
        self.use_depthwise = use_depthwise

        self.pre_channels = pre_channels
        intermediate_layer_channels = pre_channels
        self.relu = nn.ReLU6()
        if insert_1x1_conv:
            intermediate_layer_channels = depth_fn(layer_depth // 2)
            self.conv1x1 = nn.Conv2d(pre_channels, intermediate_layer_channels,
                                     kernel_size=1, stride=1, pad_mode='same')
            self.bn_1x1 = nn.BatchNorm2d(intermediate_layer_channels, eps=0.001, momentum=0.9997)
        stride = 2
        pad_mode = 'same'
        if self.use_explicit_padding:
            pad_mode = 'valid'
            self.fix_padding = FixedPadding(kernel_size=conv_kernel_size)
        if self.use_depthwise:
            # in_channels intermedia_layer depth
            out_channels = intermediate_layer_channels * depth_multiplier
            self.depthwise_Conv2d = inception_v2.depthwise_separable_conv(intermediate_layer_channels,
                                                                          None,
                                                                          depth_multiplier, kernel_size=3,
                                                                          pad_mode=pad_mode, stride=stride)
            self.Conv2d_1x1 = nn.Conv2d(out_channels,
                                        depth_fn(layer_depth), [1, 1],
                                        padding='SAME',
                                        stride=1)
            if pool_residual and pre_channels == depth_fn(layer_depth):
                if use_explicit_padding:
                    self.fix_padding = FixedPadding(kernel_size=conv_kernel_size)
                self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, pad_mode='same')
        else:
            self.conv = nn.Conv2d(intermediate_layer_channels, depth_fn(layer_depth),
                                  kernel_size=conv_kernel_size, stride=stride, pad_mode=pad_mode)
        self.bn = nn.BatchNorm2d(depth_fn(layer_depth), eps=0.001, momentum=0.9997)

    def construct(self, pre_layer):
        """Construct a forward graph"""
        intermediate_layer = pre_layer
        if self.insert_1x1_conv:
            intermediate_layer = self.conv1x1(pre_layer)
            intermediate_layer = self.bn_1x1(intermediate_layer)
            intermediate_layer = self.relu(intermediate_layer)
        if self.use_explicit_padding:
            intermediate_layer = self.fix_padding(intermediate_layer)
        if self.use_depthwise:
            feature_map = self.depthwise_Conv2d(intermediate_layer)
            feature_map = self.Conv2d_1x1(feature_map)
            if self.pool_residual and self.pre_layer_depth == self.channels:
                if self.use_explicit_padding:
                    pre_layer = self.fix_padding(pre_layer)
                feature_map += self.avg_pool(pre_layer)
        else:
            feature_map = self.conv(intermediate_layer)
        feature_map = self.bn(feature_map)
        feature_map = self.relu(feature_map)
        return feature_map


class MultiResolutionFeatureMaps(nn.Cell):
    """Define the method of eval"""
    layer_depth: list

    def __init__(self, feature_map_layout, depth_multiplier, feature_map_channels,
                 min_depth=16, insert_1x1_conv=True, pool_residual=False):
        """
        Args:
            feature_map_layout: A dict describing feature maps.
            feature_map_channels: A dict of out_channels of InceptionV2. Keys are the name of each layer.
                Values are out_channels of each layer.
            min_depth:
            insert_1x1_conv (bool):
            pool_residual (bool):
        """
        super(MultiResolutionFeatureMaps, self).__init__()
        self.depth_fn = get_depth_fn(depth_multiplier, min_depth)

        # self.image_features = image_features
        self.feature_map_layout = feature_map_layout
        self.feature_map_channels = feature_map_channels    # get it from inception_v2
        self.from_layer = feature_map_layout['from_layer']
        self.layer_depth = feature_map_layout['layer_depth']
        self.num_features = len(feature_map_layout['from_layer'])
        feature_map_keys = []
        self.use_explicit_padding = False
        if 'use_explicit_padding' in feature_map_layout:
            self.use_explicit_padding = feature_map_layout["use_explicit_padding"]
        self.use_depthwise = False
        if 'use_depthwise' in feature_map_layout:
            self.use_depthwise = feature_map_layout["use_depthwise"]
        feature_generators = []
        feature_channels = []
        # A list of operation sequences for each feature map generation.
        for index, from_layer in enumerate(self.from_layer):
            layer_depth = self.layer_depth[index]
            conv_kernel_size = 3
            if 'conv_kernel_size' in feature_map_layout:
                conv_kernel_size = feature_map_layout['conv_kernel_size'][index]
            if from_layer:
                feature_generators.append(NoneCell())
                feature_channels.append(self.feature_map_channels[from_layer])
                feature_map_keys.append(from_layer)
            else:
                pre_channels = feature_channels[-1]
                operation = AdditionalFeatureMapGenerator(insert_1x1_conv, self.use_explicit_padding,
                                                          self.use_depthwise, pool_residual, pre_channels,
                                                          layer_depth, depth_multiplier, min_depth,
                                                          conv_kernel_size)
                feature_generators.append(operation)
                feature_channels.append(operation.channels)
        self.feature_generators = nn.CellList(feature_generators)
        self.num_features = len(self.feature_generators)

    def construct(self, image_features):
        """
        Inputs:
            image_features (dict): Feature map of each layer.
        """
        features = []
        for k in range(self.num_features):
            if k < 2:
                feature_map = image_features[self.from_layer[k]]
            else:
                pre_layer = features[-1]
                feature_map = self.feature_generators[k](pre_layer)
            features.append(feature_map)
        return features
