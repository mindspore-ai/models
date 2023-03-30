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

import mindspore.ops as ops
from mindspore import nn


def pixel_shuffle(x, upscale_factor):
    r"""
    pixel_shuffle operatrion.

    Applies a pixel_shuffle operation over an input signal composed of several input planes. This is useful for
    implementiong efficient sub-pixel convolution with a stride of :math:`1/r`. For more details, refer to
    `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
    <https://arxiv.org/abs/1609.05158>`_ .

    Typically, the `x` is of shape :math:`(*, C \times r^2, H, W)` , and the output is of shape
    :math:`(*, C, H \times r, W \times r)`, where `r` is an upscale factor and `*` is zero or more batch dimensions.

    Args:
        x (Tensor): Tensor of shape :math:`(*, C \times r^2, H, W)` . The dimension of `x` is larger than 2, and the
            length of third to last dimension can be divisible by `upscale_factor` squared.
        upscale_factor (int):  factor to increase spatial resolution by, and is a positive integer.

    Returns:
        - **output** (Tensor) - Tensor of shape :math:`(*, C, H \times r, W \times r)` .

    Raises:
        ValueError: If `upscale_factor` is not a positive integer.
        ValueError: If the length of third to last dimension is not divisible by `upscale_factor` squared.
        TypeError: If the dimension of `x` is less than 3.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = np.arange(3 * 2 * 9 * 4 * 4).reshape((3, 2, 9, 4, 4))
        >>> input_x = mindspore.Tensor(input_x, mindspore.dtype.int32)
        >>> output = ops.pixel_shuffle(input_x, 3)
        >>> print(output.shape)
        (3, 2, 1, 12, 12)
    """
    assert isinstance(upscale_factor, int) and upscale_factor > 0, "upscale_factor should be bigger than 0"
    idx = x.shape
    length = len(idx)
    if length < 3:
        raise TypeError(
            f"For pixel_shuffle, the dimension of `x` should be larger than 2, but got {length}.")
    pre = idx[:-3]
    c, h, w = idx[-3:]
    if c % upscale_factor ** 2 != 0:
        raise ValueError(
            "For 'pixel_shuffle', the length of third to last dimension is not divisible"
            "by `upscale_factor` squared.")
    c = c // upscale_factor ** 2
    input_perm = (pre + (c, upscale_factor, upscale_factor, h, w))
    reshape = ops.Reshape()
    x = reshape(x, input_perm)
    input_perm = [i for i in range(length - 2)]
    input_perm = input_perm + [length, length - 2, length + 1, length - 1]
    input_perm = tuple(input_perm)
    transpose = ops.Transpose()
    x = transpose(x, input_perm)
    x = reshape(x, (pre + (c, upscale_factor * h, upscale_factor * w)))
    return x


class PixelShuffle(nn.Cell):
    r"""
    PixelShuffle operatrion.

    Applies a pixelshuffle operation over an input signal composed of several input planes. This is useful for
    implementiong efficient sub-pixel convolution with a stride of :math:`1/r`. For more details, refer to
    `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
    <https://arxiv.org/abs/1609.05158>`_ .

    Typically, the input is of shape :math:`(*, C \times r^2, H, W)` , and the output is of shape
    :math:`(*, C, H \times r, W \times r)`, where r is an upscale factor and * is zero or more batch dimensions.

    Args:
        upscale_factor (int):  factor to increase spatial resolution by, and is a positive integer.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(*, C \times r^2, H, W)` . The dimension of `x` is larger than 2, and
          the length of third to last dimension can be divisible by `upscale_factor` squared.

    Outputs:
        - **output** (Tensor) - Tensor of shape :math:`(*, C, H \times r, W \times r)` .

    Raises:
        ValueError: If `upscale_factor` is not a positive integer.
        ValueError: If the length of third to last dimension of `x` is not divisible by `upscale_factor` squared.
        TypeError: If the dimension of `x` is less than 3.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = np.arange(3 * 2 * 9 * 4 * 4).reshape((3, 2, 9, 4, 4))
        >>> input_x = mindspore.Tensor(input_x, mindspore.dtype.int32)
        >>> pixel_shuffle = nn.PixelShuffle(3)
        >>> output = pixel_shuffle(input_x)
        >>> print(output.shape)
        (3, 2, 1, 12, 12)
    """

    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def construct(self, x):
        return pixel_shuffle(x, self.upscale_factor)
