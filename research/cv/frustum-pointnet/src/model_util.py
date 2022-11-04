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

import sys

import numpy as np

import mindspore as ms
import mindspore.ops.functional as mF
from mindspore.ops import constexpr
from mindspore.ops import Print
from mindspore import ops, nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ms_print = Print()
# -----------------
# Global Constants
# -----------------

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
    'Car': np.array([3.88311640418, 1.62856739989, 1.52563191462]),
    'Van': np.array([5.06763659, 1.9007158, 2.20532825]),
    'Truck': np.array([10.13586957, 2.58549199, 3.2520595]),
    'Pedestrian': np.array([0.84422524, 0.66068622, 1.76255119]),
    'Person_sitting': np.array([0.80057803, 0.5983815, 1.27450867]),
    'Cyclist': np.array([1.76282397, 0.59706367, 1.73698127]),
    'Tram': np.array([16.17150617, 2.53246914, 3.53079012]),
    'Misc': np.array([3.64300781, 1.54298177, 1.92320313])
}
g_mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3))  # size clustrs
for _i in range(NUM_SIZE_CLUSTER):
    g_mean_size_arr[_i, :] = g_type_mean_size[g_class2type[_i]]
g_mean_size_arr_ms = ms.Tensor.from_numpy(g_mean_size_arr)


def checksummary(data, name: str = None):
    # return
    if name:
        print(name)
    print("mean \t var")
    print(f"{np.mean(data)}\t{np.var(data)}")


def repeat(x: ms.Tensor, d) -> ms.Tensor:
    return ms.numpy.tile(x, d)


def parse_output_to_tensors(box_pred, logits, mask, stage1_center):
    '''
    :param box_pred: (bs,59)
    :param logits: (bs,1024,2)
    :param mask: (bs,1024)
    :param stage1_center: (bs,3)
    :return:
        center_boxnet:(bs,3)
        heading_scores:(bs,12)
        heading_residuals_normalized:(bs,12),-1 to 1
        heading_residuals:(bs,12)
        size_scores:(bs,8)
        size_residuals_normalized:(bs,8)
        size_residuals:(bs,8)
    '''

    bs = box_pred.shape[0]
    # center
    center_boxnet = box_pred[:, :3]  # 0:3
    c = 3

    # heading
    heading_scores = box_pred[:, c:c + NUM_HEADING_BIN]  # 3:3+12
    c += NUM_HEADING_BIN
    heading_residuals_normalized = \
        box_pred[:, c:c + NUM_HEADING_BIN]  # 3+12 : 3+2*12
    heading_residuals = \
        heading_residuals_normalized * (ms.numpy.pi / NUM_HEADING_BIN)
    c += NUM_HEADING_BIN

    # size
    size_scores = box_pred[:, c:c + NUM_SIZE_CLUSTER]  # 3+2*12 : 3+2*12+8
    c += NUM_SIZE_CLUSTER
    size_residuals_normalized = \
        box_pred[:, c:c + 3 * NUM_SIZE_CLUSTER]  # [32,24] 3+2*12+8 : 3+2*12+4*8
    size_residuals_normalized = \
        size_residuals_normalized.view(bs, NUM_SIZE_CLUSTER, 3)  # [32,8,3]

    temp = ms.ops.functional.expand_dims(g_mean_size_arr_ms.astype(ms.float32), 0)  # [1,8,3]
    temp = ms.ops.tile(temp, (bs, 1, 1))  # [32,8,3]
    size_residuals = size_residuals_normalized * temp
    return center_boxnet,\
        heading_scores, heading_residuals_normalized, heading_residuals,\
        size_scores, size_residuals_normalized, size_residuals


def point_cloud_masking_v2(point_cloud, logits, xyz_only=True):
    '''
    point_cloud : [B,C,N]
    '''
    bs = point_cloud.shape[0]
    n_pts = point_cloud.shape[2]

    mask = logits[:, :, 0] < logits[:, :, 1]  # (bs, n)
    mask: ms.Tensor = ms.numpy.expand_dims(mask, 1)  # (bs, 1, n)
    temp = mask.astype(ms.float32)
    temp = temp.sum(2, keepdims=True)
    mask_count = repeat(temp, (1, 3, 1))  # (bs, 3, 1)

    pts_xyz = point_cloud[:, :3, :]  # (bs,3,n)

    mask_xyz_mean = (repeat(mask, (1, 3, 1)) * pts_xyz).sum(
        2, keepdims=True)  # (bs, 3, 1)
    mask_xyz_mean: ms.Tensor = mask_xyz_mean / \
        ops.clip_by_value(mask_count, clip_value_min=ms.Tensor(
            1, ms.int32))  # (bs, 3, 1)

    mask = mask.view(bs, -1)  # (bs,n)
    pts_xyz_stage1 = pts_xyz - repeat(mask_xyz_mean, (1, 1, n_pts))  # (bs,3,n)
    if xyz_only:
        pts_stage1 = pts_xyz_stage1
    else:
        pts_features = point_cloud[:, 3:, :]
        pts_stage1 = ms.ops.Concat(-1)([pts_xyz_stage1, pts_features])


    object_pts, _ = ms_gather_object_pc_v10(pts_stage1, mask,
                                            NUM_OBJECT_POINT)  # (32,512,3)
    object_pts: ms.Tensor = object_pts.reshape(bs, NUM_OBJECT_POINT, -1)
    object_pts = object_pts.view(bs, 3, -1)


    return object_pts, ops.Squeeze()(mask_xyz_mean), mask


def mask_to_indices_v2(pts, mask, npoints=NUM_OBJECT_POINT):
    '''
    :param point_cloud: (bs,c(3),1024)
    :param mask: (bs,1024)
    :param n_pts: max number of points of an object
    :return:
        object_pts:(bs,c,n_pts)
        indices:(bs,n_pts)
    '''
    bs = pts.shape[0]
    indices = np.zeros((bs, npoints), dtype=np.int32)
    object_pts = np.zeros((bs, pts.shape[1], npoints))
    for i in range(bs):
        pos_indices = np.where(mask[i, :] > 0.5)[0]
        # skip cases when pos_indices is empty
        if pos_indices.__len__() > 0:
            if len(pos_indices) > npoints:
                choice = np.random.choice(len(pos_indices),
                                          npoints,
                                          replace=False)
            else:
                choice = np.random.choice(len(pos_indices),
                                          npoints - len(pos_indices),
                                          replace=True)
                choice = np.concatenate((np.arange(len(pos_indices)), choice))
            np.random.shuffle(choice)
            indices[i, :] = pos_indices[choice]  # 512,3

            object_pts[i, :, :] = np.swapaxes(pts[i, :, indices[i, :]], 0, 1)
        else:
            pass
    return object_pts, indices


def ms_gather_object_pc_v10(point_cloud, mask, npoints=NUM_OBJECT_POINT):
    '''
    :param point_cloud: (bs,c(3),1024)
    :param mask: (bs,1024)
    :param n_pts: max number of points of an object
    :return:
        object_pts:(bs,c,n_pts)
        indices:(bs,n_pts)
    '''
    assert point_cloud.dtype == ms.float32
    object_pts, indices = mask_to_indices_v2(point_cloud.asnumpy(),
                                             (mask > 0.5).asnumpy(), npoints)
    object_pts = ms.Tensor(object_pts, ms.float32)
    indices = ms.Tensor(indices, ms.int32)

    return object_pts, indices


slice_op = ops.Slice()
concat_1 = ops.Concat(axis=1)
expand_dims_op = ops.ExpandDims()
stack_op = ops.Stack(axis=1)
batchmatmul_op = ops.BatchMatMul()


def get_box3d_corners_helper(centers, headings, sizes) -> ms.Tensor:
    """
    TF layer.
    Input: (N,3), (N,), (N,3)
    Output: (N,8,3)
    """


    N = centers.shape[0]
    l = slice_op(sizes, [0, 0], [sizes.shape[0] - 0, 1])  # (N,1)
    w = slice_op(sizes, [0, 1], [sizes.shape[0] - 0, 1])  # (N,1)
    h = slice_op(sizes, [0, 2], [sizes.shape[0] - 0, 1])  # (N,1)

    x_corners = concat_1(
        [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])  # (N,8)
    y_corners = concat_1(
        [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2])  # (N,8)
    z_corners = concat_1(
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])  # (N,8)

    corners = concat_1([
        expand_dims_op(x_corners, 1),
        expand_dims_op(y_corners, 1),
        expand_dims_op(z_corners, 1)
    ])  # (N,3,8)


    c = mF.cos(headings)

    s = mF.sin(headings)
    ones = ms.numpy.ones([N], dtype=ms.float32)
    zeros = ms.numpy.zeros([N], dtype=ms.dtype.float32)
    row1 = stack_op([c, zeros, s])  # (N,3)
    row2 = stack_op([zeros, ones, zeros])
    row3 = stack_op([-s, zeros, c])
    R = concat_1([
        expand_dims_op(row1, 1),
        expand_dims_op(row2, 1),
        expand_dims_op(row3, 1)
    ])  # (N,3,3)
    # corners_3d  (N,3,8)
    # R : (N,3,3)
    # corners (N,3,8)
    corners_3d = batchmatmul_op(R, corners.astype(ms.float32))  # (N,3,8)
    corners_3d += mF.tile(expand_dims_op(centers, 2), (1, 1, 8))  # (N,3,8)

    corners_3d = mF.transpose(corners_3d, (0, 2, 1))  # (N,8,3)
    return corners_3d




@constexpr()
def gen_np_arange():
    return ms.Tensor(
        np.arange(0.0,
                  2 * np.pi,
                  2 * np.pi / NUM_HEADING_BIN,
                  dtype=np.float32))


def get_box3d_corners(center, heading_residuals, size_residuals):
    """
    Inputs:
        center: (B,3)
        heading_residuals: (B,NH)
        size_residuals: (B,NS,3)
    Outputs:
        box3d_corners: (B,NH,NS,8,3) tensor
    """
    expand_dims = ops.ExpandDims()
    tile_op = ops.Tile()
    batch_size = center.shape[0]
    heading_bin_centers = gen_np_arange()  # (NH,)
    headings = heading_residuals + expand_dims(heading_bin_centers,
                                               0)  # (B,NH)

    mean_sizes = expand_dims(g_mean_size_arr_ms.astype(dtype=ms.dtype.float32),
                             0) + size_residuals  # (B,NS,1)
    sizes = mean_sizes + size_residuals  # (B,NS,3)
    sizes = tile_op(expand_dims(sizes, 1),
                    (1, NUM_HEADING_BIN, 1, 1))  # (B,NH,NS,3)
    headings = tile_op(expand_dims(headings, -1),
                       (1, 1, NUM_SIZE_CLUSTER))  # (B,NH,NS)
    centers = tile_op(expand_dims(expand_dims(center, 1), 1),
                      (1, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 1))  # (B,NH,NS,3)
    N = batch_size * NUM_HEADING_BIN * NUM_SIZE_CLUSTER
    corners_3d = get_box3d_corners_helper(centers.reshape([N, 3]),
                                          headings.reshape([N]),
                                          sizes.reshape([N, 3]))
    return corners_3d.reshape(
        [batch_size, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 8,
         3])  # [32, 12, 8, 8, 3]




def huber_loss(error, delta) -> ms.Tensor:

    abs_error = mF.absolute(error)

    quadratic = mF.minimum(abs_error, delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic**2 + delta * linear

    ans = mF.reduce_mean(losses)
    return ans


class FrustumPointNetLoss(ms.nn.Cell):

    def __init__(self, return_all=False, enable_summery=False):
        super(FrustumPointNetLoss, self).__init__()
        self.return_all = return_all
        self.sparse_softmax_cross_entropy_with_logits_op = ops.SparseSoftmaxCrossEntropyWithLogits(
        )
        self.norm_op = nn.Norm(1)
        self.norm_op_11 = nn.Norm(-1)
        self.min_op = ops.Minimum()
        self.nllloss_op = ops.NLLLoss(reduction="mean")
        self.log_softmax = ops.LogSoftmax(1)
        self.scalarsummary = ops.ScalarSummary()
        self.tensorrecorder = ops.TensorSummary()
        self.enable_summery = enable_summery

    def construct(self,
                  logits: ms.Tensor,
                  mask_label: ms.Tensor,
                  center: ms.Tensor,
                  center_label: ms.Tensor,
                  stage1_center: ms.Tensor,
                  heading_scores: ms.Tensor,
                  heading_residuals_normalized: ms.Tensor,
                  heading_residuals: ms.Tensor,
                  heading_class_label: ms.Tensor,
                  heading_residuals_label: ms.Tensor,
                  size_scores: ms.Tensor,
                  size_residuals_normalized: ms.Tensor,
                  size_residuals: ms.Tensor,
                  size_class_label: ms.Tensor,
                  size_residuals_label: ms.Tensor,
                  corner_loss_weight=30.0,
                  box_loss_weight=1.0,
                  eval_mode=False):
        '''
        1.InsSeg
        logits: torch.Size([32, 1024, 2]) torch.float32
        mask_label: [32, 1024]
        2.Center
        center: torch.Size([32, 3]) torch.float32
        stage1_center: torch.Size([32, 3]) torch.float32
        center_label:[32,3]
        3.Heading
        heading_scores: torch.Size([32, 12]) torch.float32
        heading_residuals_snormalized: torch.Size([32, 12]) torch.float32
        heading_residuals: torch.Size([32, 12]) torch.float32
        heading_class_label:(32,)
        heading_residuals_label:(32)
        4.Size
        size_scores: torch.Size([32, 8]) torch.float32
        size_residuals_normalized: torch.Size([32, 8, 3]) torch.float32
        size_residuals: torch.Size([32, 8, 3]) torch.float32
        size_class_label:(32,)
        size_residuals_label:(32,3)
        5.Corner
        6.Weight
        corner_loss_weight: float scalar
        box_loss_weight: float scalar
        '''
        bs = logits.shape[0]

        logits = self.log_softmax(logits.view(-1, 2))  # [32768, 2]
        mask_label = mask_label.view(-1).astype(ms.int32)
        mask_loss, _ = self.nllloss_op(logits, mask_label, ms.numpy.ones(
            (2,)))

        center_dist = ms.numpy.norm(center - center_label, axis=-1)  # (32,)
        center_loss = huber_loss(center_dist, delta=2.0)
        stage1_center_dist = self.norm_op(center - stage1_center)  # (32,)
        stage1_center_loss = huber_loss(stage1_center_dist, delta=1.0)

        heading_class_label = heading_class_label.astype(ms.int32)
        heading_class_loss, _ = self.nllloss_op(
            self.log_softmax(heading_scores), heading_class_label,
            ms.numpy.ones((12,)))
        hcls_onehot = ms.numpy.eye(NUM_HEADING_BIN)[
            heading_class_label]  # 32,12

        heading_residuals_normalized_label = heading_residuals_label / (
            ms.numpy.pi / NUM_HEADING_BIN)  # 32,
        heading_residuals_normalized_dist = mF.reduce_sum(
            heading_residuals_normalized * hcls_onehot.astype(ms.float32),
            1)  # 32, nan

        # Only compute reg loss on gt label
        heading_residuals_normalized_loss = huber_loss(
            heading_residuals_normalized_dist -
            heading_residuals_normalized_label,
            delta=1.0)
        # Size loss
        size_class_label = size_class_label.astype(ms.int32)
        size_class_loss, _ = self.nllloss_op(self.log_softmax(size_scores),
                                             size_class_label,
                                             ms.numpy.ones((8,)))
        scls_onehot = ms.numpy.eye(NUM_SIZE_CLUSTER)[size_class_label]  # 32,8
        scls_onehot_repeat = scls_onehot.view(-1, NUM_SIZE_CLUSTER, 1)
        scls_onehot_repeat = ms.numpy.tile(scls_onehot_repeat,
                                           (1, 1, 3))  # 32,8,3

        predicted_size_residuals_normalized_dist = mF.reduce_sum(
            size_residuals_normalized * scls_onehot_repeat.astype(ms.float32),
            1)  # 32,3
        mean_size_arr_expand = g_mean_size_arr_ms.view(1, NUM_SIZE_CLUSTER,
                                                       3)  # 1,8,3
        mean_size_label = mF.reduce_sum(scls_onehot_repeat *
                                        mean_size_arr_expand.astype(ms.float32), 1)  # 32,3
        size_residuals_label_normalized = size_residuals_label / mean_size_label
        size_normalized_dist = self.norm_op(
            size_residuals_label_normalized.astype(ms.float32) -
            predicted_size_residuals_normalized_dist.astype(ms.float32))  # 32
        size_residuals_normalized_loss = huber_loss(
            size_normalized_dist,
            delta=1.0)
        # Corner Loss
        corners_3d = get_box3d_corners(
            center, heading_residuals,
            size_residuals)  # (bs,NH,NS,8,3)(32, 12, 8, 8, 3)
        t1 = ms.numpy.tile(hcls_onehot.view(bs, NUM_HEADING_BIN, 1),
                           (1, 1, NUM_SIZE_CLUSTER))
        t2 = ms.numpy.tile(scls_onehot.view(bs, 1, NUM_SIZE_CLUSTER),
                           (1, NUM_HEADING_BIN, 1))
        gt_mask = t1 * t2  # (bs,NH=12,NS=8)

        corners_3d_pred = mF.reduce_sum(
            gt_mask.view(bs, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 1, 1).astype(
                ms.float32) * corners_3d, [1, 2])  # (bs,8,3)

        heading_bin_centers = gen_np_arange()
        heading_label = heading_residuals_label.view(bs, 1) + \
            heading_bin_centers.view(
                1, NUM_HEADING_BIN)  # (bs,1)+(1,NH)=(bs,NH)

        heading_label = mF.reduce_sum(
            hcls_onehot.astype(ms.float32) * heading_label, 1)
        mean_sizes = g_mean_size_arr_ms.astype(ms.float32).view(
            1, NUM_SIZE_CLUSTER, 3)  # (1,NS,3)
        size_label = mean_sizes + size_residuals_label.view(
            bs, 1, 3)  # (1,NS,3)+(bs,1,3)=(bs,NS,3)

        size_label = (
            scls_onehot.view(bs, NUM_SIZE_CLUSTER, 1).astype(ms.float32) *
            size_label).sum(1).astype(ms.float32)  # (B,3)
        corners_3d_gt = get_box3d_corners_helper(center_label, heading_label,
                                                 size_label)  # (B,8,3)
        corners_3d_gt_flip = get_box3d_corners_helper(
            center_label, heading_label + ms.numpy.pi, size_label)  # (B,8,3)
        n1 = ms.numpy.norm(corners_3d_pred - corners_3d_gt, axis=-1)
        n2 = ms.numpy.norm(corners_3d_pred - corners_3d_gt_flip, axis=-1)
        corners_dist = self.min_op(n1, n2)
        corners_loss = huber_loss(corners_dist, delta=1.0)
        # Weighted sum of all losses
        total_loss = mask_loss + box_loss_weight * (
            center_loss + heading_class_loss + size_class_loss +
            heading_residuals_normalized_loss * 20 +
            size_residuals_normalized_loss * 20 + stage1_center_loss +
            corner_loss_weight * corners_loss)

        _ = {
            "total_loss": total_loss,
            "mask_loss": mask_loss,
            "center_loss": center_loss,
            "heading_class_loss": heading_class_loss,
            "size_class_loss": size_class_loss,
            "heading_residuals_normalized_loss":
            heading_residuals_normalized_loss,
            "size_residuals_normalized_loss": size_residuals_normalized_loss,
            "stage1_center_loss": stage1_center_loss,
            "corner_loss": corners_loss,
        }

        if self.enable_summery:
            if not eval_mode:
                self.scalarsummary("total_loss", total_loss)
                self.scalarsummary("mask_loss", mask_loss)
                self.scalarsummary("center_loss", center_loss)
                self.scalarsummary("heading_class_loss", heading_class_loss)
                self.scalarsummary("size_class_loss", size_class_loss)
                self.scalarsummary("heading_residuals_normalized_loss",
                                   heading_residuals_normalized_loss)
                self.scalarsummary("size_residuals_normalized_loss",
                                   size_residuals_normalized_loss)
                self.scalarsummary("stage1_center_loss", stage1_center_loss)
                self.scalarsummary("corner_loss", corners_loss)
            else:
                self.scalarsummary("eval_total_loss", total_loss)
                self.scalarsummary("eval_mask_loss", mask_loss)
                self.scalarsummary("eval_center_loss", center_loss)
                self.scalarsummary("eval_heading_class_loss",
                                   heading_class_loss)
                self.scalarsummary("eval_size_class_loss", size_class_loss)
                self.scalarsummary("eval_heading_residuals_normalized_loss",
                                   heading_residuals_normalized_loss)
                self.scalarsummary("eval_size_residuals_normalized_loss",
                                   size_residuals_normalized_loss)
                self.scalarsummary("eval_stage1_center_loss",
                                   stage1_center_loss)
                self.scalarsummary("eval_corner_loss", corners_loss)

        if self.return_all:
            return total_loss, mask_loss, \
                box_loss_weight * center_loss, \
                box_loss_weight * heading_class_loss, \
                box_loss_weight * size_class_loss, \
                box_loss_weight * heading_residuals_normalized_loss * 20, \
                box_loss_weight * size_residuals_normalized_loss * 20,\
                box_loss_weight * stage1_center_loss, \
                box_loss_weight * corners_loss * corner_loss_weight
        return total_loss
