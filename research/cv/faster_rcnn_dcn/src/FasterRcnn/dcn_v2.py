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
"""Deformable Convolution operator V2"""

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np

np.random.seed(0)
ms.common.set_seed(0)


@ops.constexpr
def _get_offset_base(offset_shape, stride):
    """
    get base position index from deformable shift of each kernel element.
    """
    # (n, 2*k*k, h, w)
    k2, h, w = offset_shape[1] // 2, offset_shape[2], offset_shape[3]
    k = int(k2**0.5)
    # (k)
    range_pn = np.arange(-(k - 1) // 2, (k - 1) // 2 + 1)
    # (k, k), (k, k)
    p_n_x, p_n_y = np.meshgrid(range_pn, range_pn, indexing='ij')
    # (k*k), (k*k) -> (2*k, k)
    p_n = np.concatenate((p_n_x, p_n_y), axis=0)
    # (2*k, k) -> (1, 2*k*k, 1, 1)
    _shape = (1, 2 * k2, 1, 1)
    p_n = p_n.reshape(_shape)

    range_h = np.arange(k // 2, h * stride + 1, stride)
    range_w = np.arange(k // 2, w * stride + 1, stride)
    # (h, w), (h, w)
    p_0_x, p_0_y = np.meshgrid(range_h, range_w, indexing='ij')

    # (h, w) -> (1, 1, h, w)
    p_0_x = p_0_x.reshape(1, 1, h, w)
    # (1, 1, h, w) -> (1, k*k, h, w)
    p_0_x = np.tile(p_0_x, (1, k2, 1, 1))

    # (h, w) -> (1, 1, h, w)
    p_0_y = p_0_y.reshape(1, 1, h, w)
    # (1, 1, h, w) -> (1, k*k, h, w)
    p_0_y = np.tile(p_0_y, (1, k2, 1, 1))

    # (1, k*k, h, w), (1, k*k, h, w) -> (1, 2*k*k, h, w)
    p_0 = np.concatenate((p_0_x, p_0_y), axis=1)
    # (1, 2*k*k, h, w) + (1, 2*k*k, 1, 1) -> (1, 2*k*k, h, w)
    p = p_0 + p_n
    return ms.Tensor(p.astype(np.float32))


def _get_feature_by_index(x, p_h, p_w):
    """gather feature by specified index"""
    # x (n, c, h_in, w_in)
    # p_h (n, h, w, k*k)
    # p_w (n, h, w, k*k)
    n, c, h_in, w_in = x.shape
    _, h, w, k2 = p_h.shape
    # (n, c, h_in, w_in) -> (n, h_in, w_in, c)
    x = x.transpose(0, 2, 3, 1)

    # the following is the opt for:
    # input(n, h_in, w_in, c), index_x/index_y(n, h, w, k*k) -> output(n, h, w, k*k, c)

    # (n, h_in, w_in, c) -> (n*h_in*w_in, c)
    x = x.reshape(-1, c)

    # (n)
    idx_0_n = ops.range(ms.Tensor(0, mstype.int32), ms.Tensor(n, mstype.int32), ms.Tensor(1, mstype.int32))
    # (n, h, w, k*k) + (n, h, w, k*k) + (n, 1, 1, 1) -> (n, h, w, k*k)
    index = p_w + p_h * w_in + idx_0_n.reshape(n, 1, 1, 1) * w_in * h_in

    # (n*h_in*w_in, c), (n, h, w, k*k) -> (n, h, w, k*k, c)
    x_offset = ops.Gather()(x, index, 0)
    # (n, h*w*k*k, c) -> (n, h*w, k*k, c)
    x_offset = x_offset.reshape(n, h * w, k2, c)
    # (n, h*w, k*k, c) -> (n, c, h*w, k*k)
    x_offset = x_offset.transpose(0, 3, 1, 2)
    # (n, c, h*w, k*k) -> (n, c, h, w, k*k)
    x_offset = x_offset.reshape(n, c, h, w, k2)
    return x_offset


def _regenerate_feature_map(x_offset):
    """ get rescaled feature map which was enlarged by ks**2 times."""
    # offset (n, c, h, w, k*k)
    n, c, h, w, k2 = x_offset.shape
    k = ops.ScalarCast()(k2 ** 0.5, mstype.int32)
    # (n, c, h, w, k*k) -> k * (n, c, h, w, k)
    splits = ops.Split(axis=-1, output_num=k)(x_offset)
    # k * (n, c, h, w, k) -> (n, c, h, k*w, k)
    x_offset = ops.Concat(axis=3)(splits)
    # (n, c, h, k*w, k) -> (n, c, h*k, w*k)
    x_offset = x_offset.reshape(n, c, h * k, w * k)
    return x_offset


class DeformConv2d(nn.Cell):
    """
    Deformable convolution opertor

    Args:
        inc(int): Input channel.
        outc(int): Output channel.
        kernel_size (int): Convolution window. Default: 3.
        stride (int): The distance of kernel moving. Default: 1.
        padding (int): Implicit paddings size on both sides of the input. Default: 1.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: False.
        modulation (bool): If True, modulated defomable convolution (Deformable ConvNets v2). Default: True.
    Returns:
        Tensor, detection of images(bboxes, score, keypoints and category id of each objects)
    """
    def __init__(self, inc, outc, kernel_size=3, stride=1, pad_mode='same', padding=0, has_bias=False, modulation=True):
        super().__init__()
        self.stride = stride
        self.modulation = modulation
        self.zero_padding = nn.Pad(((0, 0), (0, 0), (padding, padding), (padding, padding)))
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, pad_mode='valid', padding=0,
                              stride=kernel_size, has_bias=has_bias)

        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=kernel_size,
                                pad_mode=pad_mode, padding=padding, stride=stride, has_bias=False)

        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=kernel_size,
                                    pad_mode=pad_mode, padding=padding, stride=stride, has_bias=False)
        if kernel_size % 2 == 0:
            raise ValueError("Only odd number is supported, but current kernel sizeis {}".format(kernel_size))

    def construct(self, x):
        """deformed sampling locations with augmented offsets"""
        # 0 ── h ──x
        # |
        # w
        # |
        # y

        # (n, c, h_in, w_in)
        x_shape = x.shape
        # get learned shift for each pixels(the shift is relative to current pixel)
        # (n, c, h_in, w_in) -> (n, 2*k*k, h, w)
        offset = self.p_conv(x)

        # get absolute position of each pixel w.r.s to input feature map without offset
        # -> (1, 2*k*k, h, w)
        p_base = _get_offset_base(offset.shape, self.stride)
        # (1, 2*k*k, h, w) + (n, 2*k*k, h, w) -> (n, 2*k*k, h, w)
        p = p_base + offset

        # (n, 2*k*k, h, w) -> (n, h, w, 2*k*k)
        p = p.transpose(0, 2, 3, 1)
        p_lt = ops.Floor()(p).astype(mstype.int32)
        p_rb = p_lt + 1

        # (n, h, w, 2*k*k) -> (n, h, w, k*k), (n, h, w, k*k)
        k2 = p.shape[-1] // 2
        p_h = p[:, :, :, :k2].clip(0, x_shape[2] - 1)
        p_w = p[:, :, :, k2:].clip(0, x_shape[3] - 1)

        # (n, h, w, 2*k*k) -> (n, h, w, k*k), (n, h, w, k*k)
        p_lt_h = p_lt[:, :, :, :k2].clip(0, x_shape[2] - 1)
        p_lt_w = p_lt[:, :, :, k2:].clip(0, x_shape[3] - 1)

        # (n, h, w, 2*k*k) -> (n, h, w, k*k), (n, h, w, k*k)
        p_rb_h = p_rb[:, :, :, :k2].clip(0, x_shape[2] - 1)
        p_rb_w = p_rb[:, :, :, k2:].clip(0, x_shape[3] - 1)

        # perform bilinear interpolation
        # (n, h, w, k*k) -> (n, h, w, k*k)
        weight_lt = (1 - (p_h - p_lt_h)) * (1 - (p_w - p_lt_w))
        weight_rb = (p_h - p_lt_h) * (p_w - p_lt_w)
        weight_rt = (1 - (p_h - p_lt_h)) * (p_w - p_lt_w)
        weight_lb = (p_h - p_lt_h) * (1 - (p_w - p_lt_w))

        # (n, c, h_in, w_in), (n, h, w, k*k), (n, h, w, k*k) -> (n, c, h, w, k*k)
        x_p_lt = _get_feature_by_index(x, p_lt_h, p_lt_w)
        x_p_rb = _get_feature_by_index(x, p_rb_h, p_rb_w)
        x_p_lb = _get_feature_by_index(x, p_rb_h, p_lt_w)
        x_p_rt = _get_feature_by_index(x, p_lt_h, p_rb_w)

        # (n, h, w, k*k) -> (n, 1, h, w, k*k) * (n, c, h, w, k*k) -> (n, c, h, w, k*k)
        x_offset = (ops.ExpandDims()(weight_lt, 1) * x_p_lt +
                    ops.ExpandDims()(weight_rb, 1) * x_p_rb +
                    ops.ExpandDims()(weight_lb, 1) * x_p_lb +
                    ops.ExpandDims()(weight_rt, 1) * x_p_rt)

        if self.modulation:
            # modulation (b, 1, h, w, N)
            # (n, c, h, w) -> (n, k*k, h, w)
            m = ops.Sigmoid()(self.m_conv(x))
            # (n, k*k, h, w) -> (n, h, w, k*k)
            m = m.transpose(0, 2, 3, 1)
            # (n, h, w, k*k) -> (n, 1, h, w, k*k)
            m = ops.ExpandDims()(m, 1)
            # (n, 1, h, w, k*k) * (n, c, h, w, k*k) -> (n, c, h, w, k*k)
            x_offset = x_offset * m
        # (n, c, h, w, k*k) -> (n, c, h*k, w*k)
        x_offset = _regenerate_feature_map(x_offset)
        # (n, c, h*k, w*k) -> (n, c, h, w)
        out = self.conv(x_offset)
        return out
