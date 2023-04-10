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
"""model"""
import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops


class STN3d(nn.Cell):
    """STN3d"""
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = mindspore.nn.Conv1d(3, 64, 1, has_bias=True, bias_init='normal')  # in_channels, out_channels, kernel_size
        self.conv2 = mindspore.nn.Conv1d(64, 128, 1, has_bias=True, bias_init='normal')
        self.conv3 = mindspore.nn.Conv1d(128, 1024, 1, has_bias=True, bias_init='normal')
        self.fc1 = nn.Dense(1024, 512)  # in_channels, out_channels
        self.fc2 = nn.Dense(512, 256)
        self.fc3 = nn.Dense(256, 9)
        self.relu = ops.ReLU()
        self.reshape = ops.Reshape()
        self.tile = ops.Tile()
        self.bn1 = nn.BatchNorm2d(64)  # num_features
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.argmaxwithvalue = ops.ArgMaxWithValue(axis=2, keep_dims=True)
        self.s1 = Tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], mindspore.float32)

    def construct(self, x):
        """construct"""
        batchsize = x.shape[0]

        x = self.conv1(x)
        x = ops.ExpandDims()(x, -1)
        x = self.bn1(x)
        x = ops.Squeeze(-1)(x)
        x = self.relu(x)
        x = self.relu(ops.Squeeze(-1)(self.bn2(ops.ExpandDims()(self.conv2(x), -1))))
        x = self.relu(ops.Squeeze(-1)(self.bn3(ops.ExpandDims()(self.conv3(x), -1))))
        x = self.argmaxwithvalue(x)[1]

        x = self.reshape(x, (-1, 1024))

        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        multiples = (batchsize, 1)

        iden = self.tile(self.s1.view(1, 9), multiples)

        x = x + iden
        x = self.reshape(x, (-1, 3, 3))
        return x


class STNkd(nn.Cell):
    """STNkd"""
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = mindspore.nn.Conv1d(k, 64, 1, has_bias=True, bias_init='normal')
        self.conv2 = mindspore.nn.Conv1d(64, 128, 1, has_bias=True, bias_init='normal')
        self.conv3 = mindspore.nn.Conv1d(128, 1024, 1, has_bias=True, bias_init='normal')
        self.fc1 = nn.Dense(1024, 512)
        self.fc2 = nn.Dense(512, 256)
        self.fc3 = nn.Dense(256, k * k)
        self.relu = ops.ReLU()
        self.flatten = nn.Flatten()

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def construct(self, x):
        """construct"""
        batchsize = x.shape[0]
        x = self.relu(ops.Squeeze(-1)(self.bn1(ops.ExpandDims()(self.conv1(x), -1))))
        x = self.relu(ops.Squeeze(-1)(self.bn2(ops.ExpandDims()(self.conv2(x), -1))))
        x = self.relu(ops.Squeeze(-1)(self.bn3(ops.ExpandDims()(self.conv3(x), -1))))
        x = ops.ExpandDims()(Tensor(np.max(x.asnumpy(), axis=2)), -1)
        reshape = ops.Reshape()
        x = reshape(x, (-1, 1024))

        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        tile = ops.Tile()
        multiples = (batchsize, 1)

        iden = tile(Tensor(np.eye(self.k).flatten().astype(np.float32)).view(1, self.k * self.k), multiples)
        x = x + iden
        x = reshape(x, (-1, self.k, self.k))
        return x


class PointNetfeat(nn.Cell):
    """PointNetfeat"""
    def __init__(self, global_feat=True, feature_transform=False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = mindspore.nn.Conv1d(3, 64, 1, has_bias=True, bias_init='normal')
        self.conv2 = mindspore.nn.Conv1d(64, 128, 1, has_bias=True, bias_init='normal')
        self.conv3 = mindspore.nn.Conv1d(128, 1024, 1, has_bias=True, bias_init='normal')
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        self.relu = ops.ReLU()
        self.cat = ops.Concat(axis=1)
        self.argmaxwithvalue = ops.ArgMaxWithValue(axis=2, keep_dims=True)
        self.squeeze = ops.Squeeze(-1)
        self.expanddims = ops.ExpandDims()
        self.transpose = ops.Transpose()
        self.batmatmul = ops.BatchMatMul()
        self.reshape = ops.Reshape()
        self.tile = ops.Tile()
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def construct(self, x):
        """construct"""
        n_pts = x.shape[2]
        transf = self.stn(x)

        x = self.transpose(x, (0, 2, 1))

        x = self.batmatmul(x, transf)
        x = self.transpose(x, (0, 2, 1))
        x = self.relu(ops.Squeeze(-1)(self.bn1(ops.ExpandDims()(self.conv1(x), -1))))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = self.transpose(x, (0, 2, 1))
            x = self.batmatmul(x, transf)
            x = self.transpose(x, (0, 2, 1))
        else:
            trans_feat = None

        pointfeats = x
        x = self.relu(
            self.squeeze(self.bn2(self.expanddims(self.conv2(x), -1))))
        x = self.squeeze(self.bn3(self.expanddims(self.conv3(x), -1)))
        x = self.argmaxwithvalue(x)[1]


        x = self.reshape(x, (-1, 1024))
        multiples = (1, 1, n_pts)
        x = self.tile(self.reshape(x, (-1, 1024, 1)), multiples)

        return self.cat((x, pointfeats)), transf, trans_feat


class PointNetCls(nn.Cell):
    """PointNetCls"""
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Dense(1024, 512)
        self.fc2 = nn.Dense(512, 256)
        self.fc3 = nn.Dense(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = ops.ReLU()
        self.logsoftmax = nn.LogSoftmax(axis=1)

    def construct(self, x):
        """construct"""
        x, transf, trans_feat = self.feat(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return self.logsoftmax(x), transf, trans_feat


class PointNetDenseCls(nn.Cell):
    """PointNetDenseCls"""
    def __init__(self, k=2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = mindspore.nn.Conv1d(1088, 512, 1, has_bias=True, bias_init='normal')
        self.conv2 = mindspore.nn.Conv1d(512, 256, 1, has_bias=True, bias_init='normal')
        self.conv3 = mindspore.nn.Conv1d(256, 128, 1, has_bias=True, bias_init='normal')
        self.conv4 = mindspore.nn.Conv1d(128, self.k, 1, has_bias=True, bias_init='normal')
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.logsoftmax = nn.LogSoftmax(axis=-1)
        self.squeeze = ops.Squeeze(-1)
        self.expanddims = ops.ExpandDims()
        self.relu = ops.ReLU()
        self.train = True

    def construct(self, x):
        """construct"""
        batchsize = x.shape[0]
        n_pts = x.shape[2]
        x, _, _ = self.feat(x)
        x = self.relu(ops.Squeeze(-1)(self.bn1(ops.ExpandDims()(self.conv1(x), -1))))
        x = self.relu(ops.Squeeze(-1)(self.bn2(ops.ExpandDims()(self.conv2(x), -1))))
        x = self.relu(ops.Squeeze(-1)(self.bn3(ops.ExpandDims()(self.conv3(x), -1))))
        x = self.conv4(x)
        transpose = ops.Transpose()
        x = transpose(x, (0, 2, 1))
        x = self.logsoftmax(x.view(-1, self.k))
        x = x.view(batchsize, n_pts, self.k)

        return x


def feature_transform_regularizer(_trans):
    """feature_transform_regularizer"""
    d = _trans.shape[1]

    eye = ops.Eye()
    I = eye(d, d, mindspore.float32)[None, :, :]
    transpose = ops.Transpose()
    reduce = ops.ReduceMean()
    loss = reduce(ops.norm(np.matmul(_trans.asnumpy(), transpose(_trans, (0, 2, 1).asnumpy()) - I)), dim=(1, 2))
    return loss

if __name__ == '__main__':

    shape1 = (32, 3, 2500)
    uniformreal = ops.UniformReal()
    sim_data = uniformreal(shape1)
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.shape)

    shape2 = (32, 64, 2500)
    sim_data_64d = uniformreal(shape2)
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.shape)

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.shape)

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.shape)

    cls = PointNetCls(k=5)
    out, _, _ = cls(sim_data)
    print('class', out.shape)

    seg = PointNetDenseCls(k=3)
    out, _, _ = seg(sim_data)
    print('seg', out.shape)
    