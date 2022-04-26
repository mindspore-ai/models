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
"""Model modules."""
from mindspore import nn
from mindspore import ops

from src.layers import depth_sep_dilated_conv_3x3_bn
from src.layers import hlconv


class DepthwiseM2OIndexBlock(nn.Cell):
    """
    Depthwise many-to-one IndexNet block.

    Args:
        inp (int): Input channels.
        use_nonlinear (bool): Use nonlinear in index blocks.
        use_context (bool): Use context in index blocks.
    """
    def __init__(self, inp, use_nonlinear, use_context):
        super().__init__()

        self.indexnet1 = self._build_index_block(inp, use_nonlinear, use_context)
        self.indexnet2 = self._build_index_block(inp, use_nonlinear, use_context)
        self.indexnet3 = self._build_index_block(inp, use_nonlinear, use_context)
        self.indexnet4 = self._build_index_block(inp, use_nonlinear, use_context)

        self.sigmoid = ops.Sigmoid()
        self.reshape = ops.Reshape()
        self.concat = ops.Concat(axis=2)
        self.transpose = ops.Transpose()
        self.unsqueeze = ops.ExpandDims()
        self.softmax = ops.Softmax(axis=2)

    @staticmethod
    def _build_index_block(inp, use_nonlinear, use_context):
        """
        Build IndexNet block.

        Args:
            inp (int): Input channels.
            use_nonlinear (bool): Use nonlinear in index blocks.
            use_context (bool): Use context in index blocks.

        Returns:
            block: Inited index block.
        """
        if use_context:
            k_s, pad = 4, 1
        else:
            k_s, pad = 2, 0

        if use_nonlinear:
            return nn.SequentialCell(
                [
                    nn.Conv2d(inp, inp, kernel_size=k_s, stride=2, pad_mode='pad', padding=pad, has_bias=False),
                    nn.BatchNorm2d(inp),
                    nn.ReLU6(),
                    nn.Conv2d(inp, inp, kernel_size=1, stride=1, pad_mode='pad', padding=0, has_bias=False)
                ]
            )

        return nn.SequentialCell(
            [
                nn.Conv2d(inp, inp, kernel_size=k_s, stride=2, pad_mode='pad', padding=pad, has_bias=False)
            ]
        )

    def depth_to_space(self, input_x, kh=2, kw=2):
        """
        Change depth of the tensor.
        BS x C x H x W -> BS x C/kh/kw x H*kh x W*kw.

        Args:
            input_x: Input tensor. Shape BS x C x H x W.
            kh: Scaling for height dim.
            kw: Scaling for width dim.

        Returns:
            output_x: Output tensor. Shape BS x C/kh/kw x H*kh x W*kw.
        """
        _, c, h, w = input_x.shape
        nc = c // kh // kw
        output_x = self.reshape(input_x, (-1, nc, kh, kw, h, w))
        output_x = self.transpose(output_x, (0, 1, 4, 2, 5, 3))
        output_x = self.reshape(output_x, (-1, nc, h * kh, w * kw))
        return output_x

    def construct(self, x):
        """
        Block feed forward.

        Args:
            x: Input feature map.

        Returns:
            idx_en: Predicted indices to encoder stage.
            idx_de: Predicted indices to decoder stage.
        """
        bs, c, h, w = x.shape

        x1 = self.unsqueeze(self.indexnet1(x), 2)
        x2 = self.unsqueeze(self.indexnet2(x), 2)
        x3 = self.unsqueeze(self.indexnet3(x), 2)
        x4 = self.unsqueeze(self.indexnet4(x), 2)

        x = self.concat((x1, x2, x3, x4))

        # normalization
        y = self.sigmoid(x)
        z = self.softmax(y)
        # pixel shuffling
        y = self.reshape(y, (bs, c * 4, h // 2, w // 2))
        z = self.reshape(z, (bs, c * 4, h // 2, w // 2))
        idx_en = self.depth_to_space(z)
        idx_de = self.depth_to_space(y)

        return idx_en, idx_de


class InvertedResidual(nn.Cell):
    """
    Inverted residual block.

    Args:
        inp (int): Block input channels.
        oup (int): Block output channels.
        stride (int): Depthwise conv stride.
        dilation (int): Depthwise conv dilation.
        expand_ratio (int): Hidden channels ratio.
    """
    def __init__(self, inp, oup, stride, dilation, expand_ratio):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        self.kernel_size = 3
        self.dilation = dilation

        if expand_ratio == 1:
            self.conv = nn.SequentialCell(
                [
                    # dw
                    nn.Conv2d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=3,
                        stride=stride,
                        pad_mode='pad',
                        padding=0,
                        dilation=dilation,
                        group=hidden_dim,
                        has_bias=False
                    ),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(),
                    # pw-linear
                    nn.Conv2d(
                        hidden_dim,
                        oup,
                        kernel_size=1,
                        stride=1,
                        pad_mode='pad',
                        padding=0,
                        has_bias=False
                    ),
                    nn.BatchNorm2d(oup),
                ]
            )
        else:
            self.conv = nn.SequentialCell(
                [
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, pad_mode='pad', padding=0, has_bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(),
                    # dw
                    nn.Conv2d(
                        hidden_dim,
                        hidden_dim,
                        3,
                        stride,
                        pad_mode='pad',
                        padding=0,
                        dilation=dilation,
                        group=hidden_dim,
                        has_bias=False,
                    ),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(),
                    # pw-linear
                    nn.Conv2d(
                        hidden_dim,
                        oup,
                        1,
                        1,
                        pad_mode='pad',
                        padding=0,
                        has_bias=False,
                    ),
                    nn.BatchNorm2d(oup),
                ]
            )

    @staticmethod
    def fixed_padding(inputs, kernel_size, dilation):
        kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        padded_inputs = nn.Pad(paddings=((0, 0), (0, 0), (pad_beg, pad_end), (pad_beg, pad_end)))(inputs)
        return padded_inputs

    def construct(self, x):
        x_pad = self.fixed_padding(x, self.kernel_size, self.dilation)
        if self.use_res_connect:
            return x + self.conv(x_pad)

        return self.conv(x_pad)


class _ASPPModule(nn.Cell):
    """ASPP module."""
    def __init__(self, inp, planes, kernel_size, padding, dilation):
        super().__init__()
        if kernel_size == 1:
            self.atrous_conv = nn.SequentialCell(
                [
                    nn.Conv2d(inp, planes, 1, 1, pad_mode='pad', padding=padding, dilation=dilation, has_bias=False),
                    nn.BatchNorm2d(planes),
                    nn.ReLU6()
                ]
            )
        elif kernel_size == 3:
            # depth-wise separable convolution to save the number of parameters
            self.atrous_conv = depth_sep_dilated_conv_3x3_bn(inp, planes, padding, dilation)

    def construct(self, x):
        x = self.atrous_conv(x)
        return x


class ASPP(nn.Cell):
    """
    ASPP block.

    Args:
        inp (int): Block input channels.
        oup (int): Block output channels.
        output_stride (int): Output image stride.
        width_mult (float): Hidden layers ratio.
    """
    def __init__(self, inp, oup, output_stride, width_mult):
        super().__init__()

        if output_stride == 32:
            dilations = [1, 2, 4, 8]
        elif output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inp, int(256 * width_mult), 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inp, int(256 * width_mult), 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inp, int(256 * width_mult), 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inp, int(256 * width_mult), 3, padding=dilations[3], dilation=dilations[3])

        self.adaptive_pooling = ops.AdaptiveAvgPool2D((1, 1))

        self.global_avg_pool = nn.SequentialCell(
            [
                nn.Conv2d(inp, int(256 * width_mult), 1, stride=1, pad_mode='pad', padding=0, has_bias=False),
                nn.BatchNorm2d(int(256 * width_mult)),
                nn.ReLU6()
            ]
        )

        self.bottleneck_conv = nn.SequentialCell(
            [
                nn.Conv2d(int(256 * width_mult) * 5, oup, 1, stride=1, pad_mode='pad', padding=0, has_bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6()
                ]
        )

        self.dropout = nn.Dropout(0.5)
        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        """
        ASPP block feed forward.

        Args:
            x: Input feature map.

        Returns:
            x: Output feature map of ASPP block.
        """
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x_5 = self.adaptive_pooling(x)
        x5 = self.global_avg_pool(x_5)
        x5 = ops.ResizeNearestNeighbor(size=x4.shape[2:])(x5)
        x = self.concat((x1, x2, x3, x4, x5))

        x = self.bottleneck_conv(x)

        return self.dropout(x)


class IndexedUpsamlping(nn.Cell):
    """
    Upsampling by index block.

    Args:
        inp (int): Block input channels.
        oup (int): Block output channels.
        conv_operator (str): Name of block conv operator.
        kernel_size (int): Kernel size of block convs.
    """
    def __init__(self, inp, oup, conv_operator, kernel_size):
        super().__init__()
        self.oup = oup

        hlconv2d = hlconv[conv_operator]

        # inp, oup, kernel_size, stride, batch_norm
        self.dconv = hlconv2d(inp, oup, kernel_size, 1)
        self.concat = ops.Concat(axis=1)

    def construct(self, l_encode, l_low, indices=None):
        if indices is not None:
            l_encode = indices * ops.ResizeNearestNeighbor(size=l_low.shape[2:])(l_encode)
        l_cat = self.concat((l_encode, l_low))
        return self.dconv(l_cat)
