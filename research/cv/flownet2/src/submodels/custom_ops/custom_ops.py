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

import os
import math
from mindspore.common import dtype as mstype
from mindspore.nn import Cell
import mindspore.ops as ops



class TransposeAndPad(Cell):
    def __init__(self, pad_size):
        super(TransposeAndPad, self).__init__()
        self.tanspose = ops.Transpose()
        self.pad = ops.Pad(((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)))

    def construct(self, x):
        x_tanspose = self.tanspose(x, (0, 2, 3, 1))
        x_tanspose_pad = self.pad(x_tanspose)
        return x_tanspose_pad


class Correlation(Cell):
    def __init__(self, pad_size=0, kernel_size=0, max_displacement=0, stride1=1, stride2=2):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.max_displacement = max_displacement
        self.kernel_size = kernel_size
        self.stride1 = stride1
        self.stride2 = stride2
        self.transpose_pad = TransposeAndPad(pad_size)
        dir_path = os.path.dirname(os.path.abspath(__file__))
        self.func_path = dir_path + "/correlation.so"

    def construct(self, x1, x2):
        pad_x1 = self.transpose_pad(x1)
        pad_x2 = self.transpose_pad(x2)
        n_output_channels = (int(self.max_displacement / self.stride2) * 2 + 1) \
                            * (int(self.max_displacement / self.stride2) * 2 + 1)
        x1_shape = x1.shape
        kernel_radius = (self.kernel_size - 1) / 2
        border_radius = kernel_radius + self.max_displacement
        padded_height = x1_shape[2] + 2 * self.pad_size
        padded_width = x1_shape[3] + 2 * self.pad_size
        output_height = int(math.ceil((padded_height - 2 * border_radius) / self.stride1))
        output_width = int(math.ceil((padded_width - 2 * border_radius) / self.stride1))
        out_shape = (x1_shape[0], n_output_channels, output_height, output_width)
        correlation_forward = ops.Custom(self.func_path + ":correlation", out_shape, mstype.float32, "aot")
        output = correlation_forward(pad_x1, pad_x2)
        return output

    def bprop(self, x1, x2, out, dout):
        pad_x1 = self.transpose_pad(x1)
        pad_x2 = self.transpose_pad(x2)
        correlation_backward = ops.Custom(self.func_path + ":correlationGrad", (x1.shape, x2.shape),
                                          (mstype.float32, mstype.float32), "aot")
        dx1, dx2 = correlation_backward(pad_x1, pad_x2, dout)
        return dx1, dx2


class Resample2D(Cell):
    def __init__(self, kernel_size=1, bilinear=True):
        super(Resample2D, self).__init__()
        self.kernel_saize = kernel_size
        self.bilinear = bilinear
        dir_path = os.path.dirname(os.path.abspath(__file__))
        self.func_path = dir_path + "/resample2d.so"


    def construct(self, x1, x2):
        out_shape = (x2.shape[0], x1.shape[1], x2.shape[2], x2.shape[3])
        resample2d_forward = ops.Custom(self.func_path + ":Resample2d", out_shape, mstype.float32, "aot")
        output = resample2d_forward(x1, x2)
        return output

    def bprop(self, x1, x2, out, dout):
        Resample2d_backward = ops.Custom(self.func_path + ":Resample2dGrad", (x1.shape, x2.shape),
                                         (mstype.float32, mstype.float32), "aot")
        dx1, dx2 = Resample2d_backward(x1, x2, dout)
        return dx1, dx2
