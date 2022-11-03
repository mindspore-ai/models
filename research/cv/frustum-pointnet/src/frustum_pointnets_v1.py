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

import datetime
import sys
import os

import importlib

import mindspore as ms
import mindspore.ops.functional as mF
from mindspore import ops, nn, context
import mindspore.numpy as np
import numpy
from src.model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, repeat
from src.model_util import point_cloud_masking_v2, parse_output_to_tensors
from src.model_util import FrustumPointNetLoss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'train'))

datautil = importlib.import_module("train.datautil")

context.set_context(mode=context.PYNATIVE_MODE)

NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 8  # one cluster for each type
NUM_OBJECT_POINT = 512

g_type2class = {
    'Car': 0,
    'Van': 1,
    'Truck': 2,
    'Pedestrian': 3,
    'Person_sitting': 4,
    'Cyclist': 5,
    'Tram': 6,
    'Misc': 7
}
g_class2type = {g_type2class[t]: t for t in g_type2class}
g_type2onehotclass = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}

g_type_mean_size = {
    'Car': numpy.array([3.88311640418, 1.62856739989, 1.52563191462]),
    'Van': numpy.array([5.06763659, 1.9007158, 2.20532825]),
    'Truck': numpy.array([10.13586957, 2.58549199, 3.2520595]),
    'Pedestrian': numpy.array([0.84422524, 0.66068622, 1.76255119]),
    'Person_sitting': numpy.array([0.80057803, 0.5983815, 1.27450867]),
    'Cyclist': numpy.array([1.76282397, 0.59706367, 1.73698127]),
    'Tram': numpy.array([16.17150617, 2.53246914, 3.53079012]),
    'Misc': numpy.array([3.64300781, 1.54298177, 1.92320313])
}

g_mean_size_arr = numpy.zeros((NUM_SIZE_CLUSTER, 3))  # size clustrs
for i in range(NUM_SIZE_CLUSTER):
    g_mean_size_arr[i, :] = g_type_mean_size[g_class2type[i]]


class WarpConv1d(nn.Cell):

    def __init__(self,
                 input_chan,
                 output_chan,
                 kernel_size=1,
                 BN=True,
                 use_activity=True,
                 has_bias: bool = True):
        super(WarpConv1d, self).__init__()
        self.conv1d = nn.Conv1d(input_chan,
                                output_chan,
                                pad_mode='valid',
                                kernel_size=kernel_size,
                                has_bias=has_bias)
        self.BN = BN
        self.use_activity = use_activity
        if BN:
            self.BatchNorm2d = nn.BatchNorm2d(num_features=output_chan)
        if use_activity:
            self.relu = nn.ReLU()

    def construct(self, in_x):
        # input:[B,C,N]
        x = self.conv1d(in_x)
        # [B,output_chan,N]
        x = mF.expand_dims(x, 3)
        # [B,output_chan,N,1]
        if self.BN:
            x = self.BatchNorm2d(x)
        if self.use_activity:
            x = self.relu(x)
        x = mF.squeeze(x)  # [B C H 1] -> [B C H]
        return x


class PointNetInstanceSeg(nn.Cell):

    def __init__(self, n_classes=3, n_channel=4):
        '''v1 3D Instance Segmentation PointNet
        @input: (B,C(4),N)
        @return: logits: [bs,n,2]
        :param n_classes:3
        '''
        super(PointNetInstanceSeg, self).__init__()
        self.max_op = ms.ops.ArgMaxWithValue(axis=2, keep_dims=True)
        self.concat = ms.ops.Concat(1)

        self.wconv1 = WarpConv1d(n_channel,
                                 64,
                                 kernel_size=1,
                                 BN=True,
                                 use_activity=True)
        self.wconv2 = WarpConv1d(64,
                                 64,
                                 kernel_size=1,
                                 BN=True,
                                 use_activity=True)
        self.wconv3 = WarpConv1d(64,
                                 64,
                                 kernel_size=1,
                                 BN=True,
                                 use_activity=True)
        self.wconv4 = WarpConv1d(64,
                                 128,
                                 kernel_size=1,
                                 BN=True,
                                 use_activity=True)
        self.wconv5 = WarpConv1d(128,
                                 1024,
                                 kernel_size=1,
                                 BN=True,
                                 use_activity=True)

        self.n_classes = n_classes

        self.dconv1 = WarpConv1d(1088 + n_classes,
                                 512,
                                 kernel_size=1,
                                 BN=True,
                                 use_activity=True)
        self.dconv2 = WarpConv1d(512,
                                 256,
                                 kernel_size=1,
                                 BN=True,
                                 use_activity=True)
        self.dconv3 = WarpConv1d(256,
                                 128,
                                 kernel_size=1,
                                 BN=True,
                                 use_activity=True)
        self.dconv4 = WarpConv1d(128,
                                 128,
                                 kernel_size=1,
                                 BN=True,
                                 use_activity=True)
        self.dropout = nn.Dropout(keep_prob=0.5)
        self.dconv5 = WarpConv1d(128,
                                 2,
                                 kernel_size=1,
                                 BN=False,
                                 use_activity=False)

    def construct(self, pts: ms.Tensor, one_hot_vec: ms.Tensor):  # bs,4,n
        '''
        :param pts: [bs,4,n]: x,y,z,intensity
        :return: logits: [bs,n,2],scores for bkg/clutter and object
        '''
        batch = pts.shape[0]
        n_pts = pts.shape[2]

        out1 = self.wconv1(pts)  # bs,64,n

        out2 = self.wconv2(out1)  # bs,64,n

        out3 = self.wconv3(out2)  # bs,64,n

        out4 = self.wconv4(out3)  # bs,128,n

        out5 = self.wconv5(out4)  # bs,1024,n

        global_feat = self.max_op(out5)[1]  #bs,1024,1

        expand_one_hot_vec = one_hot_vec.view(batch, -1, 1)  #bs,3,1

        l1 = mF.cast(global_feat, ms.float32)
        l2 = mF.cast(expand_one_hot_vec, ms.float32)
        expand_global_feat = self.concat((l1, l2))  #bs,1027,1

        expand_global_feat_repeat = ms.numpy.tile(
            expand_global_feat.view(batch, -1, 1), (1, 1, n_pts))  # bs,1027,n

        concat_feat = self.concat([out2, expand_global_feat_repeat
                                   ])  # bs, (64+1024+3)=1091, n

        x = self.dconv1(concat_feat)  #bs,512,n
        x = self.dconv2(x)  #bs,256,n
        x = self.dconv3(x)  #bs,128,n
        x = self.dconv4(x)  #bs,128,n

        x = self.dropout(x)
        x = self.dconv5(x)  #bs, 2, n

        seg_pred = x.swapaxes(1, 2)
        return seg_pred


class PointNetEstimation(nn.Cell):

    def __init__(self, n_classes=3):
        '''v1 Amodal 3D Box Estimation Pointnet
        @input [bs,3,m=512]
        @return box_pred: [bs,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4]
        :param n_classes:3
        :param one_hot_vec:[bs,n_classes]
        '''
        super(PointNetEstimation, self).__init__()

        self.max_op = ms.ops.ArgMaxWithValue(axis=2, keep_dims=False)
        self.concat = ms.ops.Concat(axis=1)

        self.wconv1 = WarpConv1d(3,
                                 128,
                                 kernel_size=1,
                                 BN=True,
                                 use_activity=True)
        self.wconv2 = WarpConv1d(128,
                                 128,
                                 kernel_size=1,
                                 BN=True,
                                 use_activity=True)
        self.wconv3 = WarpConv1d(128,
                                 256,
                                 kernel_size=1,
                                 BN=True,
                                 use_activity=True)
        self.wconv4 = WarpConv1d(256,
                                 512,
                                 kernel_size=1,
                                 BN=True,
                                 use_activity=True)

        self.n_classes = n_classes

        self.fc1 = nn.Dense(512 + n_classes, 512)
        self.fc2 = nn.Dense(512, 256)
        self.fc3 = nn.Dense(256,
                            3 + NUM_HEADING_BIN * 2 + NUM_SIZE_CLUSTER * 4)

        self.fcbn1 = nn.BatchNorm2d(512)
        self.fcbn2 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def construct(self, pts: ms.Tensor, one_hot_vec: ms.Tensor):  # bs,3,m
        '''
        :param pts: [bs,3,m]: x,y,z after InstanceSeg
        :return: box_pred: [bs,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4]
            including box centers, heading bin class scores and residuals,
            and size cluster scores and residuals
        '''
        bs = pts.shape[0]
        _ = pts.shape[2]

        out1 = self.wconv1(pts)  # bs,128,n
        out2 = self.wconv2(out1)  # bs,128,n
        out3 = self.wconv3(out2)  # bs,256,n
        out4 = self.wconv4(out3)  # bs,512,n

        global_feat = self.max_op(out4)[1]  #bs,512

        expand_one_hot_vec = one_hot_vec.reshape((bs, -1))  #bs,3

        expand_global_feat = self.concat([
            mF.cast(global_feat, ms.float32),
            mF.cast(expand_one_hot_vec, ms.float32)
        ])  #bs,515
        x = self.fc1(expand_global_feat)
        x = mF.expand_dims(x, 2)
        x = mF.expand_dims(x, 3)
        x = self.fcbn1(x)
        x = mF.squeeze(x)
        x = mF.squeeze(x)
        x = self.relu1(x)  #bs,512
        x = self.fc2(x)
        x = mF.expand_dims(x, 2)
        x = mF.expand_dims(x, 3)
        x = self.fcbn2(x)
        x = mF.squeeze(x)
        x = mF.squeeze(x)
        x = self.relu1(x)  # bs,256
        box_pred = self.fc3(x)  # bs,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4
        return box_pred


class STNxyz(nn.Cell):

    def __init__(self, n_classes=3):
        super(STNxyz, self).__init__()

        self.max_op = ms.ops.ArgMaxWithValue(axis=2, keep_dims=False)
        self.concat = ms.ops.Concat(axis=1)

        self.wconv1 = WarpConv1d(3,
                                 128,
                                 kernel_size=1,
                                 BN=True,
                                 use_activity=True)
        self.wconv2 = WarpConv1d(128,
                                 128,
                                 kernel_size=1,
                                 BN=True,
                                 use_activity=True)
        self.wconv3 = WarpConv1d(128,
                                 256,
                                 kernel_size=1,
                                 BN=True,
                                 use_activity=True)

        self.fc1 = nn.Dense(256 + n_classes, 256)
        self.fc2 = nn.Dense(256, 128)
        self.fc3 = nn.Dense(128, 3)

        self.fcbn1 = nn.BatchNorm2d(256)
        self.fcbn2 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()

    def construct(self, pts: ms.Tensor, one_hot_vec: ms.Tensor):
        '''
        :param pts: [bs,3,m]: x,y,z after InstanceSeg
        '''
        # pts:[32,3,512]
        _ = pts.shape[0]
        x = self.wconv1(pts)  # bs,128,n
        x = self.wconv2(x)  # bs,128,n
        x = self.wconv3(x)  # bs,256,n
        x = self.max_op(x)[1]  # bs,256
        expand_one_hot_vec = one_hot_vec  # bs,3
        x = self.concat(
            [mF.cast(x, ms.float32),
             mF.cast(expand_one_hot_vec, ms.float32)])  #bs,259
        x = self.fc1(x)
        x = mF.expand_dims(x, 2)
        x = mF.expand_dims(x, 3)
        x = self.fcbn1(x)
        x = mF.squeeze(x)
        x = mF.squeeze(x)
        x = self.relu(x)  # bs,256
        x = self.fc2(x)
        x = mF.expand_dims(x, 2)
        x = mF.expand_dims(x, 3)
        x = self.fcbn2(x)
        x = mF.squeeze(x)
        x = mF.squeeze(x)
        x = self.relu(x)  # bs,128
        x = self.fc3(x)  # bs,3
        return x


class FrustumPointNetv1(nn.Cell):

    def __init__(self, n_classes=3, n_channel=4):
        super(FrustumPointNetv1, self).__init__()
        self.n_classes = n_classes
        self.InsSeg = PointNetInstanceSeg(n_classes=3, n_channel=n_channel)
        self.STN = STNxyz(n_classes=3)
        self.est = PointNetEstimation(n_classes=3)

    def construct(self, pts: ms.Tensor, one_hot_vec: ms.Tensor):
        # pts:[bs,n,4]

        pts = pts.swapaxes(2, 1)  #bs,4,n
        # 3D Instance Segmentation PointNet
        logits = self.InsSeg(pts, one_hot_vec)  #[bs,n,2]
        # Mask Point Centroid
        object_pts_xyz, mask_xyz_mean, mask = \
                 point_cloud_masking_v2(pts, logits)
        # T-Net
        # assert object_pts_xyz.shape = (B,3,M)
        center_delta = self.STN(object_pts_xyz, one_hot_vec)  #(32,3)
        stage1_center = center_delta + mask_xyz_mean  #(32,3)

        temp = repeat(center_delta.view(center_delta.shape[0], -1, 1),
                      (1, 1, object_pts_xyz.shape[-1]))
        object_pts_xyz_new = object_pts_xyz - temp

        # 3D Box Estimation
        box_pred = self.est(object_pts_xyz_new, one_hot_vec)  #(32, 59)
        center_boxnet, \
        heading_scores, heading_residuals_normalized, heading_residuals, \
        size_scores, size_residuals_normalized, size_residuals = \
                parse_output_to_tensors(box_pred, logits, mask, stage1_center)

        box3d_center = center_boxnet + stage1_center  #bs,3
        return logits, mask, stage1_center, center_boxnet, \
            heading_scores, heading_residuals_normalized, heading_residuals, \
            size_scores, size_residuals_normalized, size_residuals, box3d_center


class FpointWithLoss_old(nn.Cell):

    def __init__(self, net, loss_fn):
        super(FpointWithLoss_old, self).__init__()
        self.net = net
        self.lossfn = loss_fn
        self.scalerecord = ops.ScalarSummary()

    def construct(self, batch_data, batch_label, batch_center, batch_hclass,
                  batch_hres, batch_sclass, batch_sres, batch_rot_angle,
                  batch_one_hot_vec):

        batch_data = batch_data.astype(ms.float32)
        batch_one_hot_vec = batch_one_hot_vec.astype(ms.float32)

        logits, _, stage1_center, _, heading_scores, \
            heading_residuals_normalized, heading_residuals, size_scores, \
                size_residuals_normalized, \
                    size_residuals, center = self.net(batch_data, batch_one_hot_vec)

        batch_label = batch_label.astype(ms.float32)
        batch_center = batch_center.astype(ms.float32)
        batch_hclass = batch_hclass.astype(ms.int32)
        batch_hres = batch_hres.astype(ms.float32)
        batch_sclass = batch_sclass.astype(ms.int32)
        batch_sres = batch_sres.astype(ms.float32)

        total_loss = self.lossfn(logits, batch_label, \
                        center, batch_center, stage1_center, \
                        heading_scores, heading_residuals_normalized, \
                        heading_residuals, \
                        batch_hclass, batch_hres, \
                        size_scores, size_residuals_normalized, \
                        size_residuals, \
                        batch_sclass, batch_sres)

        return total_loss


class FpointWithEval(nn.Cell):

    def __init__(self, net, lossfn=None):
        super(FpointWithEval, self).__init__()
        self.net = net
        self.lossfn = lossfn

    def construct(self, batch_data, batch_label, batch_center, batch_hclass,
                  batch_hres, batch_sclass, batch_sres, batch_rot_angle,
                  batch_one_hot_vec):
        # batch_label = seg
        output = self.net(batch_data, batch_one_hot_vec)

        label = (batch_data, batch_label.astype(ms.float32),
                 batch_center.astype(ms.float32),
                 batch_hclass.astype(ms.int32), batch_hres.astype(ms.float32),
                 batch_sclass.astype(ms.int32), batch_sres.astype(ms.float32),
                 batch_rot_angle.astype(ms.float32))


        batch_data = batch_data.astype(ms.float32)
        batch_one_hot_vec = batch_one_hot_vec.astype(ms.float32)

        return output, label


def demo():
    #python models/pointnet.py
    train_data_set = datautil.get_train_data()
    print(f"{datetime.datetime.now().isoformat()}:construct net ...")
    data = train_data_set.create_dict_iterator(1).__next__()
    points = data['data']
    print("data,", points.shape)
    print("input points")
    label = data['one_hot_vec']

    model = FrustumPointNetv1(n_classes=1, n_channel=4)
    logits, mask, stage1_center, center_boxnet, \
            heading_scores, heading_residuals_normalized, heading_residuals, \
            size_scores, size_residuals_normalized, size_residuals, center \
            = model(points, label)
    print('logits:', logits.shape, logits.dtype)
    print('mask:', mask.shape, mask.dtype)
    print('stage1_center:', stage1_center.shape, stage1_center.dtype)
    print('center_boxnet:', center_boxnet.shape, center_boxnet.dtype)
    print('heading_scores:', heading_scores.shape, heading_scores.dtype)
    print('heading_residuals_normalized:', heading_residuals_normalized.shape, \
          heading_residuals_normalized.dtype)
    print('heading_residuals:', heading_residuals.shape, \
          heading_residuals.dtype)
    print('size_scores:', size_scores.shape, size_scores.dtype)
    print('size_residuals_normalized:', size_residuals_normalized.shape, \
          size_residuals_normalized.dtype)
    print('size_residuals:', size_residuals.shape, size_residuals.dtype)
    print('center:', center.shape, center.dtype)

    loss = FrustumPointNetLoss()
    mask_label = np.zeros((32, 1024))
    center_label = np.zeros((32, 3))
    heading_class_label = np.zeros(32, np.int32)
    heading_residuals_label = np.zeros(32)
    size_class_label = np.zeros(32, np.int32)
    size_residuals_label = np.zeros((32, 3))
    output_loss = loss(logits, mask_label, \
                center, center_label, stage1_center, \
                heading_scores, heading_residuals_normalized, heading_residuals, \
                heading_class_label, heading_residuals_label, \
                size_scores, size_residuals_normalized, size_residuals, \
                size_class_label, size_residuals_label)
    print('output_loss', output_loss)


if __name__ == '__main__':
    demo()
