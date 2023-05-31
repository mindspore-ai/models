# Copyright 2023 Huawei Technologies Co., Ltd
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
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as numpy


class DeMosaic(nn.Cell):
    # Class DeMosaic is modified from this url:
    # https://github.com/cheind/pytorch-debayer/blob/master/debayer/modules.py#L8

    def __init__(self):
        super(DeMosaic, self).__init__()
        self.kernels = ms.Tensor([
            [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0],
            [0.0, 0.25, 0.0], [0.25, 0.0, 0.25], [0.0, 0.25, 0.0],
            [0.25, 0.0, 0.25], [0.0, 0.0, 0.0], [0.25, 0.0, 0.25],
            [0.0, 0.0, 0.0], [0.5, 0.0, 0.5], [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0],
        ]).reshape(5, 1, 3, 3)

        self.index = ms.Tensor([
            [0, 3], [4, 2],
            [1, 0], [0, 1],
            [2, 4], [3, 0],
        ]).view(1, 3, 2, 2)

        self.pad = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode='SYMMETRIC')
        self.gather = ops.GatherD()

    def construct(self, raw):
        b, _, h, w = raw.shape
        rgb = ops.conv2d(self.pad(raw), self.kernels, pad_mode='pad', padding=0)
        rgb = self.gather(rgb, 1, numpy.tile(self.index, (b, 1, h//2, w//2)))
        return rgb


class Resize(nn.Cell):
    def __init__(self):
        super(Resize, self).__init__()
        self.size = (1280, 1280)

    def construct(self, rgb):
        return ops.interpolate(rgb, sizes=self.size, mode='bilinear', coordinate_transformation_mode='half_pixel')


class GrayWorldWB(nn.Cell):
    def __init__(self):
        super(GrayWorldWB, self).__init__()
        self.cast = ops.Cast()
        self.clip_min = 0
        self.clip_max = 2 ** 24 - 1

    def construct(self, rgb):
        rgb = self.cast(rgb, ms.float64)
        r, g, b = ops.split(rgb, axis=1, output_num=3)
        mean_r = r.mean(axis=(2, 3), keep_dims=True)
        mean_g = g.mean(axis=(2, 3), keep_dims=True)
        mean_b = b.mean(axis=(2, 3), keep_dims=True)
        r *= mean_g / mean_r
        b *= mean_g / mean_b
        rgb = ops.concat([r, g, b], axis=1)
        rgb = ops.clip_by_value(rgb, self.clip_min, self.clip_max)
        return self.cast(rgb, ms.float32)


class DataPreprocess(nn.Cell):
    def __init__(self, ops_):
        super(DataPreprocess, self).__init__()
        self.ops = [func() for func in ops_]
        self.norm = 2 ** 24 - 1

    def construct(self, raw):
        for op in self.ops:
            raw = op(raw)
        return raw / self.norm
