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

import mindspore
import mindspore.nn as nn
import mindspore.ops as P


def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True):
    if isReLU:
        return nn.SequentialCell(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, has_bias=True, pad_mode="pad"),
            nn.LeakyReLU(0.1)
        )
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                     padding=((kernel_size - 1) * dilation) // 2, has_bias=True, pad_mode="pad")


def upsample2d_as(inputs, target_as):
    _, _, h1, w1 = P.Shape()(target_as)
    _, _, h2, _ = P.Shape()(inputs)
    resize = (h1 + 0.0) / (h2 + 0.0)
    return P.ResizeBilinear((h1, w1))(inputs) * resize

class FeatureExtractor(nn.Cell):
    '''Feature extract network'''
    def __init__(self, num_chs):
        super(FeatureExtractor, self).__init__()
        self.num_chs = num_chs

        self.convs = nn.CellList()

        for _, (ch_in, ch_out) in enumerate(zip(num_chs[:-1], num_chs[1:])):
            layer = nn.SequentialCell(
                conv(ch_in, ch_out, stride=2),
                conv(ch_out, ch_out)
            )
            self.convs.append(layer)

    def construct(self, x):
        feature_pyramid = []
        feature_pyramid_tmp = []
        for _conv in self.convs:
            x = _conv(x)
            feature_pyramid_tmp.append(x)
        feature_pyramid.append(feature_pyramid_tmp[5])
        feature_pyramid.append(feature_pyramid_tmp[4])
        feature_pyramid.append(feature_pyramid_tmp[3])
        feature_pyramid.append(feature_pyramid_tmp[2])
        feature_pyramid.append(feature_pyramid_tmp[1])
        feature_pyramid.append(feature_pyramid_tmp[0])
        return feature_pyramid


# Warping layer ---------------------------------
def get_grid(x):
    batch_size, height, width, _ = P.Shape()(x)
    tmp1 = nn.Range(batch_size)()
    tmp2 = nn.Range(height)()
    tmp3 = nn.Range(width)()
    inputs = (tmp1, tmp2, tmp3)
    Bg, Yg, Xg = P.Meshgrid(indexing='ij')(inputs)
    return Bg, Yg, Xg

def nearest_warp(x, flow):
    grid_b, grid_y, grid_x = get_grid(x)
    flow = flow.astype("Int32")

    warped_gy = P.Add()(grid_y, flow[:, :, :, 1])
    warped_gx = P.Add()(grid_x, flow[:, :, :, 0])
    _, h, w, _ = P.Shape()(x)
    warped_gy = mindspore.ops.clip_by_value(warped_gy, 0, h-1)
    warped_gx = mindspore.ops.clip_by_value(warped_gx, 0, w-1)
    warped_indices = P.Stack(3)([grid_b, warped_gy, warped_gx])

    warped_x = P.GatherNd()(x, warped_indices)
    return warped_x

def bilinear_warp(x, flow):
    _, h, w, _ = P.Shape()(x)
    grid_b, grid_y, grid_x = get_grid(x)
    grid_b = grid_b.astype("float32")
    grid_y = grid_y.astype("float32")
    grid_x = grid_x.astype("float32")

    temp1 = P.Unstack(-1)(flow)
    fx = temp1[0]
    fy = temp1[1]
    fx_0 = P.Floor()(fx)
    fx_1 = fx_0+1
    fy_0 = P.Floor()(fy)
    fy_1 = fy_0+1

    # warping indices
    h_lim = h-1
    w_lim = w-1

    gy_0 = mindspore.ops.clip_by_value(grid_y + fy_0, 0., h_lim)
    gy_1 = mindspore.ops.clip_by_value(grid_y + fy_1, 0., h_lim)
    gx_0 = mindspore.ops.clip_by_value(grid_x + fx_0, 0., w_lim)
    gx_1 = mindspore.ops.clip_by_value(grid_x + fx_1, 0., w_lim)

    g_00 = P.Stack(3)([grid_b, gy_0, gx_0]).astype("Int32")
    g_01 = P.Stack(3)([grid_b, gy_0, gx_1]).astype("Int32")
    g_10 = P.Stack(3)([grid_b, gy_1, gx_0]).astype("Int32")
    g_11 = P.Stack(3)([grid_b, gy_1, gx_1]).astype("Int32")

    # gather contents
    x_00 = P.GatherNd()(x, g_00)
    x_01 = P.GatherNd()(x, g_01)
    x_10 = P.GatherNd()(x, g_10)
    x_11 = P.GatherNd()(x, g_11)

    # coefficients
    c_00 = P.ExpandDims()((fy_1 - fy) * (fx_1 - fx), 3)
    c_01 = P.ExpandDims()((fy_1 - fy) * (fx - fx_0), 3)
    c_10 = P.ExpandDims()((fy - fy_0) * (fx_1 - fx), 3)
    c_11 = P.ExpandDims()((fy - fy_0) * (fx - fx_0), 3)

    return c_00 * x_00 + c_01 * x_01 + c_10 * x_10 + c_11 * x_11

class WarpingLayer(nn.Cell):
    '''define warplayer'''
    def __init__(self, warp_type='nearest'):
        super(WarpingLayer, self).__init__()
        self.warp = warp_type

    def construct(self, x, flow):
        x = mindspore.ops.Transpose()(x, (0, 2, 3, 1))
        flow = mindspore.ops.Transpose()(flow, (0, 2, 3, 1))
        if self.warp == 'nearest':
            x_warped = nearest_warp(x, flow)
        else:
            x_warped = bilinear_warp(x, flow)
        x_warped = mindspore.ops.Transpose()(x_warped, (0, 3, 1, 2))
        return x_warped


class OpticalFlowEstimator(nn.Cell):
    '''define OpticalFlowEstimator'''
    def __init__(self, ch_in):
        super(OpticalFlowEstimator, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in, 128),
            conv(128, 128),
            conv(128, 96),
            conv(96, 64),
            conv(64, 32)
        )
        self.conv_last = conv(32, 2, isReLU=False)

    def construct(self, x):
        x_intm = self.convs(x)
        return x_intm, self.conv_last(x_intm)


class FlowEstimatorDense(nn.Cell):
    '''define FlowEstimator network'''
    def __init__(self, ch_in):
        super(FlowEstimatorDense, self).__init__()
        self.conv1 = conv(ch_in, 128)
        self.conv2 = conv(ch_in + 128, 128)
        self.conv3 = conv(ch_in + 256, 96)
        self.conv4 = conv(ch_in + 352, 64)
        self.conv5 = conv(ch_in + 416, 32)
        self.conv_last = conv(ch_in + 448, 2, isReLU=False)
        self.concat = P.Concat(1)

    def construct(self, x):
        x1 = self.concat([self.conv1(x), x])
        x2 = self.concat([self.conv2(x1), x1])
        x3 = self.concat([self.conv3(x2), x2])
        x4 = self.concat([self.conv4(x3), x3])
        x5 = self.concat([self.conv5(x4), x4])
        x_out = self.conv_last(x5)
        return x5, x_out


class ContextNetwork(nn.Cell):
    '''context network'''
    def __init__(self, ch_in):
        super(ContextNetwork, self).__init__()

        self.convs = nn.SequentialCell(
            conv(ch_in, 128, 3, 1, 1),
            conv(128, 128, 3, 1, 2),
            conv(128, 128, 3, 1, 4),
            conv(128, 96, 3, 1, 8),
            conv(96, 64, 3, 1, 16),
            conv(64, 32, 3, 1, 1),
            conv(32, 2, isReLU=False)
        )

    def construct(self, x):
        return self.convs(x)
