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
"""Multi Box layers"""

import mindspore.nn as nn
from mindspore.ops import functional as F
from mindspore.ops import operations as P


def _conv2d(
        in_channel,
        out_channel,
        kernel_size=3,
        stride=1,
        pad_mod='same',
) -> nn.Conv2d:
    """Construct a convolution layer

    Args:
        in_channel (int): A number of the input channels.
        out_channel (int): A number of the output channels
        kernel_size (int): The size of the convolution kernel.
        stride (int): A number that represents the height and width of the kernel movement.
        pad_mod (str): Padding mode of the convolution.

    Returns:
        (nn.Conv2d): A 2d Convolution layer.
    """
    return nn.Conv2d(
        in_channel,
        out_channel,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,
        pad_mode=pad_mod,
        has_bias=True,
    )


def _bn(channel) -> nn.BatchNorm2d:
    """Construct two dimensional batch-normalization layer.

    Args:
        channel (int): A number of channels.

    Returns:
        (nn.BatchNorm2d): A batch-normalization layer.
    """
    return nn.BatchNorm2d(
        channel,
        eps=1e-3,
        momentum=0.97,
        gamma_init=1,
        beta_init=0,
        moving_mean_init=0,
        moving_var_init=1,
    )


def _last_conv2d(
        in_channel: int,
        out_channel: int,
        kernel_size: int = 3,
        stride: int = 1,
        pad_mode: str = 'same',
        pad: int = 0,
) -> nn.SequentialCell:
    """Construct multibox convolution block.

    Args:
        in_channel: Input channels number.
        out_channel: Output channels number
        kernel_size: A size of the depthwise convolution kernel.
        stride: A stride of the depthwise convolution layer.
        pad_mode: A pad_mode of the depthwise convolution layer.
        pad: A padding value of the depthwise convolution layer.

    Returns:
        (nn.SequentialCell): A multibox block.
    """
    in_channels = in_channel
    out_channels = in_channel

    depthwise_conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        pad_mode=pad_mode,
        padding=pad,
        group=in_channels,
    )
    conv = _conv2d(
        in_channel,
        out_channel,
        kernel_size=1,
    )
    return nn.SequentialCell([
        depthwise_conv,
        _bn(in_channel),
        nn.ReLU6(),
        conv,
    ])


class FlattenConcat(nn.Cell):
    """
    Concatenate predictions into a single tensor.

    Args:
        config (dict): The default config of SSD.

    Returns:
        Tensor, flatten predictions.
    """

    def __init__(self, config):
        super(FlattenConcat, self).__init__()
        self.num_ssd_boxes = config.num_ssd_boxes
        self.concat = P.Concat(axis=1)
        self.transpose = P.Transpose()

    def construct(self, inputs):
        """Construct FlattenConcat"""
        output = ()
        batch_size = F.shape(inputs[0])[0]
        for x in inputs:
            x = self.transpose(x, (0, 2, 3, 1))
            output += (F.reshape(x, (batch_size, -1)),)
        res = self.concat(output)
        return F.reshape(res, (batch_size, self.num_ssd_boxes, -1))


class MultiBox(nn.Cell):
    """Multibox convolution layers.
    Each multibox layer contains class confidence scores and localization predictions.

    Args:
        config (dict): The default config of SSD.

    Returns:
        Tensor, localization predictions.
        Tensor, class conf scores.
    """

    def __init__(self, config):
        super(MultiBox, self).__init__()
        num_classes = config.num_classes
        out_channels = config.extras_out_channels
        num_default = config.num_default

        loc_layers = []
        cls_layers = []
        for k, out_channel in enumerate(out_channels):
            loc_layers += [_last_conv2d(out_channel, 4 * num_default[k],
                                        kernel_size=3, stride=1, pad_mode='same', pad=0)]
            cls_layers += [_last_conv2d(out_channel, num_classes * num_default[k],
                                        kernel_size=3, stride=1, pad_mode='same', pad=0)]

        self.multi_loc_layers = nn.layer.CellList(loc_layers)
        self.multi_cls_layers = nn.layer.CellList(cls_layers)
        self.flatten_concat = FlattenConcat(config)

    def construct(self, inputs):
        """Calculate localization predictions and confidence scores by given features.

        Args:
            inputs: An input features.

        Returns:
            (Tensor): localization predictions.
            (Tensor): class confidence scores.
        """
        loc_outputs = ()
        cls_outputs = ()
        for i in range(len(self.multi_loc_layers)):
            loc_outputs += (self.multi_loc_layers[i](inputs[i]),)

            cls_outputs += (self.multi_cls_layers[i](inputs[i]),)

        return self.flatten_concat(loc_outputs), self.flatten_concat(cls_outputs)
