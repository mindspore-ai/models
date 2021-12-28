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
"""merge cell in nasfpn"""

from abc import abstractmethod

import mindspore.nn as nn
from mindspore.ops import operations as P

class Relu(nn.Cell):
    def __init__(self):
        super(Relu, self).__init__()
        self.max = P.Maximum()

    def construct(self, x):
        x = self.max(0, x)
        return x

class GlobalAvgPooling(nn.Cell):
    """
        global average pooling feature map.

    Args:
         mean (tuple): means for each channel.
    """
    def __init__(self):
        super(GlobalAvgPooling, self).__init__()
        self.mean = P.ReduceMean(keep_dims=True)

    def construct(self, x):
        x = self.mean(x, (2, 3))
        return x

class ActCon2dBn(nn.Cell):
    """
        Relu Conv2d BatchNorm2d block definition.
    """
    def __init__(self, in_planes, out_planes, kernel_size=3, padding=0, pad_mode='pad', stride=1, groups=1):
        super(ActCon2dBn, self).__init__()
        relu = nn.ReLU()
        con = nn.Conv2d(in_planes, out_planes, kernel_size, stride, pad_mode=pad_mode,
                        padding=padding)
        bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.997,
                            gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)
        layers = [relu, con, bn]
        self.features = nn.SequentialCell(layers)

    def construct(self, x):
        output = self.features(x)
        return output

class BaseMergeCell(nn.Cell):
    """The basic class for cells used in NAS-FPN.

    BaseMergeCell takes 2 inputs. After applying convolution
    on them, they are resized to the target size. Then,
    they go through binary_op, which depends on the type of cell.
    If with_out_conv is True, the result of output will go through
    another convolution layer.

    Args:
        in_channels (int): number of input channels in out_conv layer.
        out_channels (int): number of output channels in out_conv layer.
        with_out_conv (bool): Whether to use out_conv layer
    """

    def __init__(self,
                 fused_channels=256,
                 out_channels=256,
                 with_out_conv=True,
                 groups=1,
                 kernel_size=3,
                 padding=1
                ):
        super(BaseMergeCell, self).__init__()

        self.with_out_conv = with_out_conv

        if self.with_out_conv:
            self.out_conv = ActCon2dBn(
                fused_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=groups
                )

    @abstractmethod
    def _binary_op(self, x1, x2):
        pass

    def _resize(self, x, size):
        if x.shape[-2] == size[0]:
            return x
        if x.shape[-2] < size[0]:
            return P.ResizeNearestNeighbor(size)(x)
        kernel_size = x.shape[-1] // size[-1]
        x = P.MaxPool(kernel_size=kernel_size, strides=kernel_size, pad_mode="SAME")(x)
        return x

    def construct(self, x1, x2, out_size=None):
        """Forward function."""
        if out_size is None:  # resize to larger one
            if x1.size()[2] < x2.size()[2]:
                out_size = x2.size()[2:]
            else:
                out_size = x1.size()[2:]

        x1 = self._resize(x1, out_size)
        x2 = self._resize(x2, out_size)

        x = self._binary_op(x1, x2)
        if self.with_out_conv:
            x = self.out_conv(x)
        return x

class SumCell(BaseMergeCell):
    """sum two cell."""
    def __init__(self, in_channels, out_channels, groups=1, kernel_size=3, padding=1, with_out_conv=True):
        super(SumCell, self).__init__(in_channels, out_channels,
                                      groups=groups, kernel_size=kernel_size, padding=padding,
                                      with_out_conv=with_out_conv)

    def _binary_op(self, x1, x2):
        return x1 + x2

class ConcatCell(BaseMergeCell):
    """concat two cell."""
    def __init__(self, in_channels, out_channels, groups=1, kernel_size=3, padding=1, with_out_conv=True):
        super(ConcatCell, self).__init__(in_channels * 2, out_channels,
                                         groups=groups, kernel_size=kernel_size, padding=padding,
                                         with_out_conv=with_out_conv)

    def _binary_op(self, x1, x2):
        ret = P.Concat(1)([x1, x2])
        return ret

class GlobalPoolingCell(BaseMergeCell):
    """global pooling cell."""
    def __init__(self, in_channels=None, out_channels=None, groups=1, kernel_size=3, padding=1, with_out_conv=True):
        super(GlobalPoolingCell, self).__init__(in_channels, out_channels, groups=groups,
                                                kernel_size=kernel_size, padding=padding,
                                                with_out_conv=with_out_conv)
        self.global_pool = GlobalAvgPooling()
        self.sigmoid = P.Sigmoid()

    def _binary_op(self, x1, x2):
        x2_att = self.global_pool(x2)
        x2_att = self.sigmoid(x2_att)
        return x2 + x2_att * x1
