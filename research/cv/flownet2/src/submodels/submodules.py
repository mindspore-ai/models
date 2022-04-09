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
import mindspore.nn as nn
import mindspore.ops as ops

class Norm(nn.Cell):
    def __init__(self, axis=1, keep_dims=False):
        super(Norm, self).__init__()
        self.axis = axis
        self.keep_dims = keep_dims
        self.reduce_sum = ops.ReduceSum(True)
        self.sqrt = ops.Sqrt()
        self.squeeze = ops.Squeeze(self.axis)

    def construct(self, x):
        x = self.sqrt(ops.maximum(self.reduce_sum(ops.square(x), self.axis), 1e-7))

        if not self.keep_dims:
            x = self.squeeze(x)
        return x


class ChannelNorm(nn.Cell):
    def __init__(self, axis=1):
        super(ChannelNorm, self).__init__()
        self.axis = axis
        self.add = ops.Add()
        self.norm = Norm(axis)

    def construct(self, x):
        output = self.norm(x)
        output = output.reshape(output.shape[0], 1, output.shape[1], output.shape[2])
        return output


class Upsample(nn.Cell):

    def __init__(self, scale_factor=4, mode='bilinear'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def construct(self, x):
        shape = x.shape
        new_height = shape[2] * self.scale_factor
        new_width = shape[3] * self.scale_factor
        if self.mode == 'nearest':
            upsample_op = ops.ResizeNearestNeighbor((new_height, new_width))
        else:
            upsample_op = ops.ResizeBilinear((new_height, new_width))
        return upsample_op(x)


def conv(batchnorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchnorm:
        conv2d = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, pad_mode='pad',
                           padding=(kernel_size - 1) // 2, has_bias=False)
        batchNorm2d = nn.BatchNorm2d(out_planes)
        leakyReLU = nn.LeakyReLU(0.1)
        return nn.SequentialCell([conv2d, batchNorm2d, leakyReLU])
    conv2d = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, pad_mode='pad',
                       padding=(kernel_size - 1) // 2, has_bias=True)
    leakyReLU = nn.LeakyReLU(0.1)
    return nn.SequentialCell([conv2d, leakyReLU])


def i_conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, bias=True):
    if batchNorm:
        conv2d = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, pad_mode='pad',
                           padding=(kernel_size - 1) // 2, has_bias=bias)
        batchNorm2d = nn.BatchNorm2d(out_planes)
        return nn.SequentialCell([conv2d, batchNorm2d])
    conv2d = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, pad_mode='pad',
                       padding=(kernel_size - 1) // 2, has_bias=bias)
    return nn.SequentialCell([conv2d])


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True)


def deconv(in_planes, out_planes):
    convTranspose2d = nn.Conv2dTranspose(in_planes, out_planes, kernel_size=4, stride=2, pad_mode='pad', padding=1,
                                         has_bias=True)
    leakyReLU = nn.LeakyReLU(0.1)
    return nn.SequentialCell([convTranspose2d, leakyReLU])
