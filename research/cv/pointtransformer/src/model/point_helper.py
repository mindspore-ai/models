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

import numpy as np

import mindspore
from mindspore import nn, ops, Tensor, dtype
from mindspore.ops.primitive import constexpr

class Dense16(nn.Cell):
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True,
                 activation=None):
        super().__init__()
        self.dense = nn.Dense(in_channels,
                              out_channels,
                              weight_init,
                              bias_init,
                              has_bias,
                              activation)
        self.cast = ops.Cast()
    def construct(self, x):
        x = self.cast(x, dtype.float16)
        x = self.dense(x)
        x = self.cast(x, dtype.float32)
        return x


@constexpr
def generate_tensor_fps(B, N):
    """generate tensor"""
    farthest = Tensor(np.random.randint(N, size=(B,)), mindspore.int32)
    return farthest


@constexpr
def generate_tensor_batch_indices(B):
    """generate tensor"""
    return Tensor(np.arange(B), mindspore.int32)


def batched_index_select(values, indices):
    org_shape = indices.shape
    indices = indices.reshape(org_shape[0], -1)

    indices = indices[..., None].repeat(values.shape[-1], axis=-1).astype(mindspore.int32)

    res = ops.GatherD()(values, 1, indices)
    return res.reshape(*org_shape, -1)



def square_distance(src, dst):
    dist = src[:, :, None] - dst[:, None]
    norm = nn.Norm(-1)
    return norm(dist)


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint(index)]
    """
    B, N, _ = xyz.shape

    centroids = ops.Zeros()((B, npoint), mindspore.float16)
    distance = ops.Fill()(mindspore.float32, (B, N), 1e10)
    farthest = generate_tensor_fps(B, N)
    one = Tensor(1, dtype=mindspore.int32)
    npoint = Tensor(npoint, dtype=mindspore.int32)
    batch_indices = generate_tensor_batch_indices(B)

    while npoint > 0:
        npoint = npoint - one
        centroids[:, npoint] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = nn.Norm(-1)(xyz - centroid)
        distance = ops.Minimum()(distance, dist)
        farthest = ops.ArgMaxWithValue(axis=-1)(distance)[0]
    return centroids.astype(mindspore.int32)


def knn_sample_and_group(npoint, num_neighbors, pos, feats):
    """
    Input:
        npoint: sampling center
        num_neighbors: number of neighbors
        pos: input points position data, [B, N, 3]
        points: input points data feature, [B, N, D]
    Return:
        new_pos: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, _, C = pos.shape

    fps_idx = farthest_point_sample(pos, npoint)

    centroids_pos = batched_index_select(pos, fps_idx)

    dists = square_distance(centroids_pos, pos)
    indices = ops.Sort(axis=-1)(dists)[1][:, :, :num_neighbors]

    grouped_pos = batched_index_select(pos, indices)

    grouped_pos_norm = grouped_pos - centroids_pos.view(B, npoint, 1, C)

    if feats is not None:
        grouped_points = batched_index_select(feats, indices)
        new_feats = ops.Concat(axis=-1)([grouped_pos_norm, grouped_points])
    else:
        new_feats = grouped_pos_norm

    return centroids_pos, new_feats


class TransitionDown(nn.Cell):
    def __init__(self, stride, nsample, in_channel, out_channel):
        super(TransitionDown, self).__init__()
        self.stride = stride
        self.nsample = nsample
        self.relu = nn.ReLU()

        if self.stride != 1:
            self.max_pooling = ops.ArgMaxWithValue(axis=2)
            self.conv = nn.Conv2d(in_channel + 3, out_channel, 1, has_bias=True, weight_init='he_uniform')
            self.bn1 = nn.BatchNorm2d(out_channel, momentum=0.1)
        else:
            self.linear = Dense16(in_channel, out_channel, weight_init='he_uniform')
            self.bn2 = nn.BatchNorm1d(out_channel, momentum=0.1)
    def construct(self, feats_pos):
        feats, pos = feats_pos
        B, N, _ = feats.shape

        if self.stride != 1:
            bn, conv = self.bn1, self.conv
            new_pos, new_feats = knn_sample_and_group(N // self.stride, self.nsample, pos, feats)
            new_feats = new_feats.transpose(0, 3, 2, 1)
            new_feats = self.relu(bn(conv(new_feats)))
            new_feats = self.max_pooling(new_feats)[1]
            new_feats = new_feats.transpose(0, 2, 1)
            pos = new_pos
        else:
            bn, linear = self.bn2, self.linear
            new_feats = bn(linear(feats).view(B*N, -1)).view(B, N, -1)
            new_feats = self.relu(new_feats)
        return [new_feats, pos]


class TransitionUp(nn.Cell):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        self.relu = nn.ReLU()
        if out_planes is None:
            self.linear = Dense16(in_planes, in_planes)
            self.bn = nn.BatchNorm1d(in_planes, momentum=0.1)
        else:
            self.linear1 = Dense16(out_planes, out_planes)
            self.bn1 = nn.BatchNorm1d(out_planes, momentum=0.1)
            self.linear2 = Dense16(in_planes, out_planes)
            self.bn2 = nn.BatchNorm1d(out_planes, momentum=0.1)

    def construct(self, feats_pos1, feats_pos2=None):
        if feats_pos2 is None:
            feats1, pos1 = feats_pos1
            B, N, _ = feats1.shape
            feats = self.linear(feats1)
            feats = self.relu(self.bn(feats1.view(B*N, -1)).view(B, N, -1))
        else:
            feats1, pos1 = feats_pos1
            feats2, pos2 = feats_pos2

            B2, N2, _ = feats2.shape
            feats2 = self.linear2(feats2)
            feats2 = self.relu(self.bn2(feats2.view(B2*N2, -1)).view(B2, N2, -1))

            B1, N1, _ = feats1.shape
            feats1 = self.linear1(feats1)
            feats1 = self.relu(self.bn1(feats1.view(B1*N1, -1)).view(B1, N1, -1))

            dists = square_distance(pos1, pos2)
            dists, idx = ops.Sort(axis=-1)(dists)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            dist_recip = 1.0 / (dists + 1e-8)
            norm = dist_recip.sum(axis=2, keepdims=True)
            weight = dist_recip / norm
            interpolated_points = batched_index_select(feats2, idx) * weight.view(B1, N1, 3, 1)
            feats = feats1 + interpolated_points.sum(axis=2)

        return feats
