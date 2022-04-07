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
# =======================================================================================
"""Base network blocks for yolox"""
from mindspore import nn
import mindspore.ops as ops


class SiLU(nn.Cell):
    def __init__(self):
        super(SiLU, self).__init__()
        self.silu = nn.Sigmoid()

    def construct(self, x):
        return x * self.silu(x)


def get_activation(name="silu"):
    """ get the activation function """
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU()
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1)
    else:
        raise AttributeError("Unsupported activate type: {}".format(name))

    return module


class BaseConv(nn.Cell):
    """
    A conv2d  -> BatchNorm  -> silu/leaky relu block

    """

    def __init__(
            self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super(BaseConv, self).__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            pad_mode="pad",
            group=groups,
            has_bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act)

    def construct(self, x):
        x = self.act(self.bn(self.conv(x)))
        return x


def use_syc_bn(network):
    """Use synchronized batchnorm layer"""
    for _, cell in network.cells_and_names():
        if isinstance(cell, BaseConv):
            out_channels = cell.bn.num_features
            cell.bn = nn.SyncBatchNorm(out_channels)


class DWConv(nn.Cell):
    """Depthwise Conv + Point Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super(DWConv, self).__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act,
        )

    def construct(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Cell):
    """ Standard bottleneck """

    def __init__(
            self,
            in_channels,
            out_channels,
            shortcut=True,
            expansion=0.5,
            depthwise=False,
            act="silu"
    ):
        super(Bottleneck, self).__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def construct(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class ResLayer(nn.Cell):
    "Residual layer with `in_channels` inputs."

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, act="lrelu"
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, act="lrelu"
        )

    def construct(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class SPPBottleneck(nn.Cell):
    """Spatial pyramid pooling layer used in YOLOv3-SPP """

    def __init__(
            self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super(SPPBottleneck, self).__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.CellList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1)
                for ks in kernel_sizes
            ]
        )
        self.pad0 = ops.Pad(((0, 0), (0, 0), (kernel_sizes[0] // 2, kernel_sizes[0] // 2),
                             (kernel_sizes[0] // 2, kernel_sizes[0] // 2)))
        self.pad1 = ops.Pad(((0, 0), (0, 0), (kernel_sizes[1] // 2, kernel_sizes[1] // 2),
                             (kernel_sizes[1] // 2, kernel_sizes[1] // 2)))
        self.pad2 = ops.Pad(((0, 0), (0, 0), (kernel_sizes[2] // 2, kernel_sizes[2] // 2),
                             (kernel_sizes[2] // 2, kernel_sizes[2] // 2)))
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def construct(self, x):
        x = self.conv1(x)
        op = ops.Concat(axis=1)
        x1 = self.m[0](self.pad0(x))
        x2 = self.m[1](self.pad1(x))
        x3 = self.m[2](self.pad2(x))
        x = op((x, x1, x2, x3))
        x = self.conv2(x)
        return x


class CSPLayer(nn.Cell):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
            self,
            in_channels,
            out_channels,
            n=1,
            shortcut=True,
            expansion=0.5,
            depthwise=False,
            act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.SequentialCell(module_list)

    def construct(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        op = ops.Concat(axis=1)
        x = op((x_1, x_2))
        return self.conv3(x)


class Focus(nn.Cell):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def construct(self, x):
        """ Focus forward """
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        op = ops.Concat(axis=1)
        x = op(
            (patch_top_left,
             patch_bot_left,
             patch_top_right,
             patch_bot_right)
        )
        return self.conv(x)
