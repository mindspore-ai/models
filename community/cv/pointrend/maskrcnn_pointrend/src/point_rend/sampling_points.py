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
'''sampling points'''

import mindspore
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P, constexpr
from mindspore.ops import functional as F
import mindspore.numpy as np
from mindspore import ops
import numpy

def point_sample(input_, point_coords, align_corners=False):
    """
    From Detectron2, point_features.py#19

    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """

    add_dim = False
    if point_coords.ndim == 3:
        add_dim = True
        expand_dims = ops.ExpandDims()
        point_coords = expand_dims(point_coords, 2)
    gridSample = GridSampler(align_corners=align_corners)
    output = gridSample(input_, 2.0 * point_coords - 1.0)
    if add_dim:
        squeeze = ops.Squeeze(3)
        output = squeeze(output)
    return output

@constexpr
def get_tensor(data, datatype):
    return Tensor(input_data=data, dtype=datatype)

@constexpr
def get_tensor1(data):
    return Tensor(input_data=data)


@constexpr
def get_int(data):
    return int(data)

def affine_grid_generator(height, width, theta):
    """
    This function returns a sampling grid, which when
    used with the bilinear sampler on the input feature
    map, will create an output feature map that is an
    affine transformation [1] of the input feature map.

    zero = Tensor(np.zeros([]), mindspore.float32)
    Input
    -----
    - height: desired height of grid/output. Used
        to downsample or upsample.

    - width: desired width of grid/output. Used
        to downsample or upsample.

    - theta: affine transform matrices of shape (num_batch, 2, 3).
        For each image in the batch, we have 6 theta parameters of
        the form (2x3) that define the affine transformation T.

    Returns
    -------
    - normalized grid (-1, 1) of shape (num_batch, 2, H, W).
        The 2nd dimension has 2 components: (x, y) which are the
        sampling points of the original image for each point in the
        target image.

    Note
    ----
    [1]: the affine transformation allows cropping, translation,
            and isotropic scaling.
    """
    x = np.linspace(0.0, 1.0, height + 1)
    x = x[:-1] + 1 / (height*2)
    y = np.linspace(0.0, 1.0, width + 1)
    y = y[:-1] + 1 / (width*2)
    x_t, y_t = np.meshgrid(x, y)
    expand_dims = P.ExpandDims()
    x_t = expand_dims(x_t, 0)
    y_t = expand_dims(y_t, 0)
    flatten = P.Flatten()
    x_t_flat = flatten(x_t)
    y_t_flat = flatten(y_t)
    oneslike = P.OnesLike()
    ones = oneslike(x_t_flat)
    concat = P.Concat()
    sampling_grid = concat((x_t_flat, y_t_flat, ones))
    shape = P.Shape()
    num_batch = shape(theta)[0]
    cast = P.Cast()
    theta = cast(theta, mindspore.float32)
    matmul = P.BatchMatMul()
    tile = P.Tile()
    sampling_grid = tile(expand_dims(sampling_grid, 0), (num_batch, 1, 1))
    cast = P.Cast()
    sampling_grid = cast(sampling_grid, mindspore.float32)
    batch_grids = matmul(theta, sampling_grid)
    transpose = P.Transpose()
    batch_grids = transpose(batch_grids, (0, 2, 1))
    reshape = P.Reshape()
    batch_grids = reshape(batch_grids, (num_batch, height, width, 2))
    return batch_grids

def generate_regular_grid_point_coords(R, side_size):
    """
    Generate regular square grid of points in [0, 1] x [0, 1] coordinate space.

    Args:
        R (int): The number of grids to sample, one for each region.
        side_size (int): The side size of the regular grid. 14
        device (torch.device): Desired device of returned tensor.

    Returns:
        (Tensor): A tensor of shape (R, side_size^2, 2) that contains coordinates
            for the regular grids.
    """
    aff = get_tensor([[[1.0, 0, 0], [0, 1.0, 0]]], mindspore.float32)
    r = affine_grid_generator(side_size, side_size, aff)
    r = r.view(1, -1, 2)
    tile = ops.Tile()
    r = tile(r, (R, 1, 1))
    return r


def get_point_coords_wrt_image(boxes_coords, point_coords):
    """
    Convert box-normalized [0, 1] x [0, 1] point cooordinates to image-level coordinates.

    Args:
        boxes_coords (Tensor): A tensor of shape (R, 4) that contains bounding boxes.
            coordinates.
        point_coords (Tensor): A tensor of shape (R, P, 2) that contains
            [0, 1] x [0, 1] box-normalized coordinates of the P sampled points.

    Returns:
        point_coords_wrt_image (Tensor): A tensor of shape (R, P, 2) that contains
            image-normalized coordinates of P sampled points.
    """
    point_coords_wrt_image = point_coords.copy()
    point_coords_wrt_image = F.stop_gradient(point_coords_wrt_image)
    point_coords_wrt_image[:, :, 0] = point_coords_wrt_image[:, :, 0] * (
        boxes_coords[:, None, 2] - boxes_coords[:, None, 0]
    )
    point_coords_wrt_image[:, :, 1] = point_coords_wrt_image[:, :, 1] * (
        boxes_coords[:, None, 3] - boxes_coords[:, None, 1]
    )
    point_coords_wrt_image[:, :, 0] += boxes_coords[:, None, 0]
    point_coords_wrt_image[:, :, 1] += boxes_coords[:, None, 1]
    return point_coords_wrt_image

def point_sample_fine_grained_features2(features_list, feature_scales, boxes, point_coords):
    """
    Get features from feature maps in `features_list` that correspond to specific point coordinates
        inside each bounding box from `boxes`.

    Args:
        features_list (list[Tensor]): A list of feature map tensors to get features from.
        feature_scales (list[float]): A list of scales for tensors in `features_list`.

        # boxes (list[Boxes]): A list of I Boxes  objects that contain R_1 + ... + R_I = R boxes all
        #     together.
        boxes (tensor) : (128, 4)

        point_coords (Tensor): A tensor of shape (R, P, 2) that contains
            [0, 1] x [0, 1] box-normalized coordinates of the P sampled points.

    Returns:
        point_features (Tensor): A tensor of shape (R, C, P) that contains features sampled
            from all features maps in feature_list for P sampled points for all R boxes in `boxes`.
        point_coords_wrt_image (Tensor): A tensor of shape (R, P, 2) that contains image-level
            coordinates of P points.
    """
    catops = ops.Concat(0)
    concat = P.Concat(axis=0)
    num_boxes = len(boxes)
    split = ops.Split(0, num_boxes)

    pos_proposal = concat(boxes)
    point_coords_wrt_image = get_point_coords_wrt_image(pos_proposal, point_coords)  # (128,196,2)
    split_point_coords_wrt_image = split(point_coords_wrt_image)

    point_features = []
    for idx_img, point_coords_wrt_image_per_image in enumerate(split_point_coords_wrt_image):
        point_features_per_image = []
        for idx_feature, feature_map in enumerate(features_list):
            h, w = feature_map.shape[-2:]
            scale = get_tensor1([w, h]) / feature_scales[idx_feature]
            point_coords_scaled = point_coords_wrt_image_per_image / scale
            expand_dims = ops.ExpandDims()
            feature_ = expand_dims(feature_map[idx_img], 0)
            point_coords_ = expand_dims(point_coords_scaled, 0)
            point_feature = point_sample(
                feature_,
                point_coords_,
            )
            squeeze_ = ops.Squeeze(0)
            point_feature = squeeze_(point_feature)
            transpose = ops.Transpose()
            point_feature = transpose(point_feature, (1, 0, 2))
            point_features_per_image.append(
                point_feature
            )
        cat_1 = ops.Concat(1)
        point_features.append(cat_1(point_features_per_image))
    return catops(point_features), point_coords_wrt_image

def point_sample_fine_grained_features(features_list, feature_scales, boxes, point_coords):
    """
    Get features from feature maps in `features_list` that correspond to specific point coordinates
        inside each bounding box from `boxes`.

    Args:
        features_list (list[Tensor]): A list of feature map tensors to get features from.
        feature_scales (list[float]): A list of scales for tensors in `features_list`.

        # boxes (list[Boxes]): A list of I Boxes  objects that contain R_1 + ... + R_I = R boxes all
        #     together.
        boxes (tensor) : (128, 4)

        point_coords (Tensor): A tensor of shape (R, P, 2) that contains
            [0, 1] x [0, 1] box-normalized coordinates of the P sampled points.

    Returns:
        point_features (Tensor): A tensor of shape (R, C, P) that contains features sampled
            from all features maps in feature_list for P sampled points for all R boxes in `boxes`.
        point_coords_wrt_image (Tensor): A tensor of shape (R, P, 2) that contains image-level
            coordinates of P points.
    """
    catops = ops.Concat(0)
    concat = P.Concat(axis=0)
    tmp = [b.shape[0] for b in boxes]
    num_boxes = []
    for i in range(1, len(tmp)):
        tmp[i] += tmp[i - 1]
        num_boxes.append(tmp[i-1])

    pos_proposal = concat(boxes)
    point_coords_wrt_image = get_point_coords_wrt_image(pos_proposal, point_coords)  # (128,196,2)
    split_point_coords_wrt_image = np.split(point_coords_wrt_image, num_boxes)

    point_features = []
    for idx_img, point_coords_wrt_image_per_image in enumerate(split_point_coords_wrt_image):
        point_features_per_image = []
        for idx_feature, feature_map in enumerate(features_list):
            h, w = feature_map.shape[-2:]
            scale = get_tensor1([w, h]) / feature_scales[idx_feature]
            point_coords_scaled = point_coords_wrt_image_per_image / scale
            expand_dims = ops.ExpandDims()
            feature_ = expand_dims(feature_map[idx_img], 0)
            point_coords_ = expand_dims(point_coords_scaled, 0)
            point_feature = point_sample(
                feature_,
                point_coords_,
            )
            squeeze_ = ops.Squeeze(0)
            point_feature = squeeze_(point_feature)
            transpose = ops.Transpose()
            point_feature = transpose(point_feature, (1, 0, 2))
            point_features_per_image.append(
                point_feature
            )
        cat_1 = ops.Concat(1)
        point_features.append(cat_1(point_features_per_image))
    return catops(point_features), point_coords_wrt_image

def safe_index(index, max_):
    '''safe index'''
    return ops.clip_by_value(index, 0, max_-1)

class GridSampler(nn.Cell):
    '''GridSampler'''
    def __init__(self, align_corners=False):
        super(GridSampler, self).__init__()
        self.gather = ops.GatherNd()
        self.stack = ops.Stack(1)
        self.concat = ops.Concat(1)
        self.align_corners = align_corners
        self.logical_and = P.LogicalAnd()
        self.cast = P.Cast()

    def construct(self, x, grid):
        '''construct'''
        n, c, h, w = x.shape
        _, nh, nw, _ = grid.shape

        N = self.cast(np.arange(n).repeat(c * nh * nw), mindspore.int32).reshape((-1, 1))
        C = self.cast(np.arange(c).reshape(1, -1).repeat(nh * nw, 1).repeat(n, 0), mindspore.int32).reshape((-1, 1))

        if self.align_corners:
            grid_y = ((grid[..., 1] + 1) / 2) * (h - 1)
            grid_x = ((grid[..., 0] + 1) / 2) * (w - 1)
            grad_y = grid_y.reshape((n, -1))
            grid_x = grid_x.reshape((n, -1))
            grid_real_y = ops.Tile()(grad_y, (1, c))
            grid_real_x = ops.Tile()(grid_x, (1, c))
            grid_real_y = grid_real_y.reshape((-1, 1))
            grid_real_x = grid_real_x.reshape((-1, 1))
        else:
            grid_y = ((grid[..., 1] + 1) * h - 1) / 2
            grid_x = ((grid[..., 0] + 1) * w - 1) / 2

            grad_y = grid_y.reshape((n, -1))
            grid_x = grid_x.reshape((n, -1))

            grid_real_y = ops.Tile()(grad_y, (1, c))
            grid_real_x = ops.Tile()(grid_x, (1, c))

            grid_real_y = grid_real_y.reshape((-1, 1))
            grid_real_x = grid_real_x.reshape((-1, 1))

        tl_x = ops.floor(grid_real_x)
        tl_y = ops.floor(grid_real_y)
        br_x = tl_x + 1
        br_y = tl_y + 1
        tlx_w = self.logical_and(tl_x >= 0, tl_x < w)
        tly_w = self.logical_and(tl_y >= 0, tl_y < h)
        brx_w = self.logical_and(br_x >= 0, br_x < w)
        bry_w = self.logical_and(br_y >= 0, br_y < h)

        tlx = safe_index(tl_x, w)
        tly = safe_index(tl_y, h)
        brx = safe_index(br_x, w)
        bry = safe_index(br_y, h)

        tlp_index = self.concat((N, C, self.cast(tly, mindspore.int32), self.cast(tlx, mindspore.int32)))
        tlp = self.gather(x, tlp_index)
        tlp = tlp * self.cast(self.logical_and(tly_w, tlx_w)[:, 0], mindspore.float32)

        trp_index = self.concat((N, C, self.cast(tly, mindspore.int32), self.cast(brx, mindspore.int32)))
        trp = self.gather(x, trp_index)
        trp = trp * self.cast(self.logical_and(tly_w, brx_w)[:, 0], mindspore.float32)

        blp_index = self.concat((N, C, self.cast(bry, mindspore.int32), self.cast(tlx, mindspore.int32)))
        blp = self.gather(x, blp_index)
        blp = blp * self.cast(self.logical_and(bry_w, tlx_w)[:, 0], mindspore.float32)

        brp_index = self.concat((N, C, self.cast(bry, mindspore.int32), self.cast(brx, mindspore.int32)))
        brp = self.gather(x, brp_index)
        brp = brp * self.cast(self.logical_and(bry_w, brx_w)[:, 0], mindspore.float32)

        brpw = ops.absolute(((grid_real_x - tl_x) * (grid_real_y - tl_y)).reshape((-1,)))  #
        blpw = ops.absolute(((br_x - grid_real_x) * (grid_real_y - tl_y)).reshape((-1,)))  #
        trpw = ops.absolute(((grid_real_x - tl_x) * (br_y - grid_real_y)).reshape((-1,)))  #
        tlpw = ops.absolute(((br_x - grid_real_x) * (br_y - grid_real_y)).reshape((-1,)))  # se

        y = tlpw * tlp + trpw * trp + blpw * blp + brpw * brp
        return y.reshape((n, c, nh, nw))

def nms_plain(boxes, threshold):
    '''nms plain'''
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]
    order = scores.argsort()[::-1]
    areas = (x2-x1+1)*(y2-y1+1)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = numpy.maximum(x1[i], x1[order[1:]])
        yy1 = numpy.maximum(y1[i], y1[order[1:]])
        xx2 = numpy.minimum(x2[i], x2[order[1:]])
        yy2 = numpy.minimum(y2[i], y2[order[1:]])
        w = numpy.maximum(0, xx2-xx1)
        h = numpy.maximum(0, yy2-yy1)
        overlaps = w*h
        ious = overlaps/(areas[i] + areas[order[1:]] - overlaps)
        inds = numpy.where(ious < threshold)[0]
        order = order[inds+1]
    return keep

def batch_nms(boxes, idxs, scores, iou_threshold):
    '''batch nms'''
    max_coordinate = boxes.max()
    offsets = idxs.astype(numpy.float) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    boxes_scores = numpy.concatenate((boxes_for_nms, scores[:, None]), axis=1)
    keep = nms_plain(boxes_scores, iou_threshold)
    return keep
