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

"""The T-Net of Semantic Human Matting Network"""
import numpy as np
from mindspore import Tensor, ops, nn
from mindspore.ops import operations as P
from mindspore.ops.operations import Add


@ops.constexpr
def get_offset(h_in, w_in, h_out, w_out, align_corners=False):
    range_dst_h = np.arange(0, h_out)
    range_dst_w = np.arange(0, w_out)
    dst_w, dst_h = np.meshgrid(range_dst_w, range_dst_h)
    if align_corners:
        src_h = np.clip(dst_h * (h_in - 1) / (h_out - 1), 0, h_in - 1)
        src_w = np.clip(dst_w * (w_in - 1) / (w_out - 1), 0, w_in - 1)
    else:
        src_h = np.clip((dst_h + 0.5) * h_in / h_out - 0.5, 0, h_in - 1)
        src_w = np.clip((dst_w + 0.5) * w_in / w_out - 0.5, 0, w_in - 1)
    src_t_h = np.clip(np.floor(src_h), 0, h_in - 1).astype(np.int32)
    src_l_w = np.clip(np.floor(src_w), 0, w_in - 1).astype(np.int32)
    src_b_h = np.clip(src_t_h + 1, 0, h_in - 1).astype(np.int32)
    src_r_w = np.clip(src_l_w + 1, 0, w_in - 1).astype(np.int32)

    weight_l = src_r_w - src_w
    weight_r = 1 - weight_l
    weight_t = src_b_h - src_h
    weight_b = 1 - weight_t
    weight_lt = weight_l * weight_t
    weight_rb = weight_r * weight_b
    weight_lb = weight_l * weight_b
    weight_rt = weight_r * weight_t

    src_lt = Tensor(src_t_h), Tensor(src_l_w)
    src_rb = Tensor(src_b_h), Tensor(src_r_w)
    weight_lrtb = (
        Tensor(weight_lt.reshape(1, 1, h_out, w_out).astype(np.float32)),
        Tensor(weight_rb.reshape(1, 1, h_out, w_out).astype(np.float32)),
        Tensor(weight_lb.reshape(1, 1, h_out, w_out).astype(np.float32)),
        Tensor(weight_rt.reshape(1, 1, h_out, w_out).astype(np.float32)),
    )
    return src_lt, src_rb, weight_lrtb


class Upsample(nn.Cell):
    """Upsamples a given multi-channel data"""

    def __init__(self, scale_factor, mode="bilinear", align_corners=False):
        super().__init__()
        self.scale_factor = scale_factor
        assert mode == "bilinear", "only bilinear method is supported currently"
        self.align_corners = align_corners

    @staticmethod
    def _get_pixel_by_index(x, src_h, src_w):
        """gather pixel by specified index"""
        # x (n, c, h_in, w_in)
        # src_h (h_out, w_out)
        # src_w (h_out, w_out)
        n, c, h_in, w_in = x.shape
        h_out, w_out = src_h.shape
        # (n, c, h_in, w_in) ->(n*c, h_in*w_in) -> (h_in*w_in, n*c)
        x = x.reshape(n * c, h_in * w_in).transpose(1, 0)
        # (h_out, w_out) -> (h_out*w_out)
        index = src_w.reshape(-1) + src_h.reshape(-1) * w_in
        # (h_in*w_in, n*c), (h_out*w_out) -> (h_out*w_out, n*c)
        dst_x = ops.Gather()(x, index, 0)
        # (h_out*w_out, n*c) -> (n*c, h_out*w_out) -> (n, c, h_out, w_out)
        dst_x = dst_x.transpose(1, 0).reshape(n, c, h_out, w_out)
        return dst_x

    def construct(self, x):
        _, _, h_in, w_in = x.shape
        h_out = h_in * self.scale_factor
        w_out = w_in * self.scale_factor

        # src_lt: (n, c, h_out, w_out) * 2
        # src_rb: (n, c, h_out, w_out) * 2
        # weight_lrtb = (n, c, h_out, w_out) * 4
        src_lt, src_rb, weight_lrtb = get_offset(h_in, w_in, h_out, w_out, self.align_corners)
        src_t_h, src_l_w = src_lt
        src_b_h, src_r_w = src_rb
        weight_lt, weight_rb, weight_lb, weight_rt = weight_lrtb

        # (n, c, h_in, w_in), (n, c, h_out, w_out), (n, c, h_out, w_out) -> (n, c, h_out, w_out)
        dst_lt = self._get_pixel_by_index(x, src_t_h, src_l_w)
        dst_rb = self._get_pixel_by_index(x, src_b_h, src_r_w)
        dst_lb = self._get_pixel_by_index(x, src_b_h, src_l_w)
        dst_rt = self._get_pixel_by_index(x, src_t_h, src_r_w)

        # (h_out, w_out) -> (1, 1, h_out, w_out) * (n, c, h_out, w_out) -> (n, c, h_out, w_out)
        out = weight_lt * dst_lt + weight_rb * dst_rb + weight_lb * dst_lb + weight_rt * dst_rt
        return out


class InvertedResidual(nn.Cell):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.SequentialCell(
            [
                # pw
                nn.Conv2d(
                    inp, inp * expand_ratio, 1, 1, pad_mode="same", padding=0, has_bias=False, weight_init="Uniform"
                ),
                nn.BatchNorm2d(inp * expand_ratio),
                nn.ReLU6(),
                # dw
                nn.Conv2d(
                    inp * expand_ratio,
                    inp * expand_ratio,
                    3,
                    stride,
                    pad_mode="pad",
                    padding=1,
                    group=inp * expand_ratio,
                    has_bias=False,
                    weight_init="Uniform",
                ),
                nn.BatchNorm2d(inp * expand_ratio),
                nn.ReLU6(),
                # pw-linear
                nn.Conv2d(
                    inp * expand_ratio, oup, 1, 1, pad_mode="same", padding=0, has_bias=False, weight_init="Uniform"
                ),
                nn.BatchNorm2d(oup),
            ]
        )

        self.add = Add()
        self.cast = P.Cast()

    def construct(self, x):
        identity = x
        x = self.conv(x)
        if self.use_res_connect:
            return self.add(identity, x)
        return x


class mobilenet_v2(nn.Cell):
    """Mobilenet v2 network"""

    def __init__(self, nInputChannels=3):
        super(mobilenet_v2, self).__init__()

        # 1/2 ——> shape scale: (n, c, h, w) ——> (n, c', h * 1/2, w * 1/2)
        self.head_conv = nn.SequentialCell(
            [
                nn.Conv2d(
                    in_channels=nInputChannels,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    pad_mode="pad",
                    padding=1,
                    has_bias=False,
                    weight_init="Uniform",
                ),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            ]
        )

        # 1/2
        self.block_1 = InvertedResidual(32, 16, 1, 1)
        # 1/4
        self.block_2 = nn.SequentialCell([InvertedResidual(16, 24, 2, 6), InvertedResidual(24, 24, 1, 6)])
        # 1/8
        self.block_3 = nn.SequentialCell(
            [InvertedResidual(24, 32, 2, 6), InvertedResidual(32, 32, 1, 6), InvertedResidual(32, 32, 1, 6)]
        )
        # 1/16
        self.block_4 = nn.SequentialCell(
            [
                InvertedResidual(32, 64, 2, 6),
                InvertedResidual(64, 64, 1, 6),
                InvertedResidual(64, 64, 1, 6),
                InvertedResidual(64, 64, 1, 6),
            ]
        )
        # 1/16
        self.block_5 = nn.SequentialCell(
            [InvertedResidual(64, 96, 1, 6), InvertedResidual(96, 96, 1, 6), InvertedResidual(96, 96, 1, 6)]
        )
        # 1/32
        self.block_6 = nn.SequentialCell(
            [InvertedResidual(96, 160, 2, 6), InvertedResidual(160, 160, 1, 6), InvertedResidual(160, 160, 1, 6)]
        )
        # 1/32
        self.block_7 = InvertedResidual(160, 320, 1, 6)

    def construct(self, x):
        x = self.head_conv(x)
        # 1/2
        s1 = self.block_1(x)
        # 1/4
        s2 = self.block_2(s1)
        # 1/8
        s3 = self.block_3(s2)
        # 1/16
        s4 = self.block_4(s3)
        s4 = self.block_5(s4)
        # 1/32
        s5 = self.block_6(s4)
        s5 = self.block_7(s5)

        return s1, s2, s3, s4, s5


class T_mv2_unet(nn.Cell):
    """
    T-Net architecture:
        mmobilenet v2 + unet
    """

    def __init__(self, classes=3):
        super(T_mv2_unet, self).__init__()
        # -----------------------------------------------------------------
        # encoder
        # ---------------------
        self.feature = mobilenet_v2()

        # -----------------------------------------------------------------
        # decoder
        # ---------------------
        self.s5_up_conv = nn.SequentialCell(
            [
                Upsample(scale_factor=2, align_corners=False),
                nn.Conv2d(
                    320, 96, 3, 1, padding=1, pad_mode="pad", has_bias=True, weight_init="Uniform", bias_init="Uniform"
                ),
                nn.BatchNorm2d(96),
                nn.ReLU(),
            ]
        )
        self.s4_fusion = nn.SequentialCell(
            [
                nn.Conv2d(
                    96, 96, 3, 1, padding=1, pad_mode="pad", has_bias=True, weight_init="Uniform", bias_init="Uniform"
                ),
                nn.BatchNorm2d(96),
            ]
        )

        self.s4_up_conv = nn.SequentialCell(
            [
                Upsample(scale_factor=2, align_corners=False),
                nn.Conv2d(
                    96, 32, 3, 1, padding=1, pad_mode="pad", has_bias=True, weight_init="Uniform", bias_init="Uniform"
                ),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            ]
        )
        self.s3_fusion = nn.SequentialCell(
            [
                nn.Conv2d(
                    32, 32, 3, 1, padding=1, pad_mode="pad", has_bias=True, weight_init="Uniform", bias_init="Uniform"
                ),
                nn.BatchNorm2d(32),
            ]
        )

        self.s3_up_conv = nn.SequentialCell(
            [
                Upsample(scale_factor=2, align_corners=False),
                nn.Conv2d(
                    32, 24, 3, 1, padding=1, pad_mode="pad", has_bias=True, weight_init="Uniform", bias_init="Uniform"
                ),
                nn.BatchNorm2d(24),
                nn.ReLU(),
            ]
        )
        self.s2_fusion = nn.SequentialCell(
            [
                nn.Conv2d(
                    24, 24, 3, 1, padding=1, pad_mode="pad", has_bias=True, weight_init="Uniform", bias_init="Uniform"
                ),
                nn.BatchNorm2d(24),
            ]
        )

        self.s2_up_conv = nn.SequentialCell(
            [
                Upsample(scale_factor=2, align_corners=False),
                nn.Conv2d(
                    24, 16, 3, 1, padding=1, pad_mode="pad", has_bias=True, weight_init="Uniform", bias_init="Uniform"
                ),
                nn.BatchNorm2d(16),
                nn.ReLU(),
            ]
        )
        self.s1_fusion = nn.SequentialCell(
            [
                nn.Conv2d(
                    16, 16, 3, 1, padding=1, pad_mode="pad", has_bias=True, weight_init="Uniform", bias_init="Uniform"
                ),
                nn.BatchNorm2d(16),
            ]
        )

        self.last_conv = nn.Conv2d(
            16, classes, 3, 1, padding=1, pad_mode="pad", has_bias=True, weight_init="Uniform", bias_init="Uniform"
        )

        self.add = Add()

    def construct(self, x):
        # -----------------------------------------------------------------
        # encoder
        # ---------------------
        s1, s2, s3, s4, s5 = self.feature(x)

        # -----------------------------------------------------------------
        # decoder
        # ---------------------
        s4_ = self.s5_up_conv(s5)
        s4_ = self.add(s4_, s4)
        s4 = self.s4_fusion(s4_)

        s3_ = self.s4_up_conv(s4)
        s3_ = self.add(s3_, s3)
        s3 = self.s3_fusion(s3_)

        s2_ = self.s3_up_conv(s3)
        s2_ = self.add(s2_, s2)
        s2 = self.s2_fusion(s2_)

        s1_ = self.s2_up_conv(s2)
        s1_ = self.add(s1_, s1)
        s1 = self.s1_fusion(s1_)

        out = self.last_conv(s1)

        return out
