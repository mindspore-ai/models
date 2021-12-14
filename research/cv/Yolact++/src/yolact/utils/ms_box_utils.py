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
"""Mindspore version of box_utils"""
import mindspore
import mindspore.ops as P
import mindspore.nn as nn


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    concat = P.Concat(1)
    return concat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                   boxes[:, :2] + boxes[:, 2:]/2))  # xmax, ymax

def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    concat = P.Concat(1)
    return concat(((boxes[:, 2:] + boxes[:, :2])/2,     # cx, cy
                   boxes[:, 2:] - boxes[:, :2]))  # w, h


def jaccard(box_a, box_b, iscrowd: bool = False):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes. If iscrowd=True, put the crowd in box_b.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    use_batch = True
    if box_a.ndim == 2:  # dim
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]
    inter = intersect(box_a, box_b)
    expand_dims = P.ExpandDims()
    area_a = expand_dims((box_a[:, :, 2]-box_a[:, :, 0]) *(box_a[:, :, 3]-box_a[:, :, 1]), 2)
    broadcast_to = P.BroadcastTo(inter.shape)
    area_a = broadcast_to(area_a)

    area_b = expand_dims((box_b[:, :, 2] - box_b[:, :, 0]) * (box_b[:, :, 3] - box_b[:, :, 1]), 1)
    area_b = broadcast_to(area_b)

    union = area_a + area_b - inter

    out = inter / area_a if iscrowd else inter / union

    squeeze = P.Squeeze(0)
    return out if use_batch else squeeze(out)

def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [n,A,4].
      box_b: (tensor) bounding boxes, Shape: [n,B,4].
    Return:
      (tensor) intersection area, Shape: [n,A,B].
    """
    n = box_a.shape[0]
    A = box_a.shape[1]
    B = box_b.shape[1]
    intersect_min = P.Minimum()
    intersect_max = P.Maximum()
    prod = P.ReduceProd()
    expand_dims = P.ExpandDims()
    shape = (n, A, B, 2)
    broadcast_to = P.BroadcastTo(shape)
    max_xy = intersect_min(broadcast_to(expand_dims(box_a[:, :, 2:], 2)),
                           broadcast_to(expand_dims(box_b[:, :, 2:], 1)))
    min_xy = intersect_max(broadcast_to(expand_dims(box_a[:, :, :2], 2)),
                           broadcast_to(expand_dims(box_b[:, :, :2], 1)))
    zeroslike = P.ZerosLike()
    select = P.Select()
    min_tensor = zeroslike(max_xy - min_xy)
    wants = select(min_tensor > max_xy - min_xy, min_tensor, max_xy - min_xy)
    return prod(wants, 3)




def elemwise_box_iou(box_a, box_b):
    """ Does the same as above but instead of pairwise, elementwise along the inner dimension. """
    elemwise_min = P.Minimum()
    elemwise_max = P.Maximum()
    max_xy = elemwise_min(box_a[:, 2:], box_b[:, 2:])
    min_xy = elemwise_max(box_a[:, :2], box_b[:, :2])
    select = P.Select()
    zeroslike = P.ZerosLike()
    min_tensor = zeroslike(max_xy - min_xy)
    inter = select(min_tensor > max_xy - min_xy, min_tensor, max_xy - min_xy)

    inter = inter[:, 0] * inter[:, 1]

    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])

    union = area_a + area_b - inter
    oneslike = P.OnesLike()
    min_tensor = oneslike(union) * 0.1
    union = select(min_tensor > union, min_tensor, union)

    max_tensor = oneslike(inter / union)
    return_x = select(inter / union > max_tensor, max_tensor, inter / union)

    return return_x

def mask_iou(masks_a, masks_b, iscrowd=False):
    """
    Computes the pariwise mask IoU between two sets of masks of size [a, h, w] and [b, h, w].
    The output is of size [a, b].

    Wait I thought this was "box_utils", why am I putting this in here?
    """

    masks_a = masks_a.view(masks_a.size(0), -1)
    masks_b = masks_b.view(masks_b.size(0), -1)

    matmul = nn.MatMul()
    intersection = matmul(masks_a, masks_b.T)
    mask_iou_sum = P.ReduceSum()
    expand_dims = P.ExpandDims()
    area_a = expand_dims(mask_iou_sum(masks_a, 1), 1)
    area_b = expand_dims(mask_iou_sum(masks_b, 1), 0)
    return intersection / (area_a + area_b - intersection) if not iscrowd else intersection / area_a


def change(gt, priors):
    """
    Compute the d_change metric proposed in Box2Pix:
    https://lmb.informatik.uni-freiburg.de/Publications/2018/UB18/paper-box2pix.pdf

    Input should be in point form (xmin, ymin, xmax, ymax).

    Output is of shape [num_gt, num_priors]
    Note this returns -change so it can be a drop in replacement for
    """
    num_priors = priors.size(0)
    num_gt = gt.size(0)

    gt_w = (gt[:, 2] - gt[:, 0])[:, None].expand(num_gt, num_priors)
    gt_h = (gt[:, 3] - gt[:, 1])[:, None].expand(num_gt, num_priors)

    gt_mat = gt[:, None, :].expand(num_gt, num_priors, 4)
    pr_mat = priors[None, :, :].expand(num_gt, num_priors, 4)

    diff = gt_mat - pr_mat
    diff[:, :, 0] /= gt_w
    diff[:, :, 2] /= gt_w
    diff[:, :, 1] /= gt_h
    diff[:, :, 3] /= gt_h

    return -P.Sqrt((diff ** 2).sum(dim=2))


def match(pos_thresh, neg_thresh, truths, priors, labels, loc_t, conf_t, idx_t, idx, loc_data):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        pos_thresh: (float) IoU > pos_thresh ==> positive.
        neg_thresh: (float) IoU < neg_thresh ==> negative.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        crowd_boxes: (tensor) All the crowd box annotations or None if there are none.
        loc_t: (tensor) Tensor to be filled w/ encoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds. Note: -1 means neutral.
        idx_t: (tensor) Tensor to be filled w/ the index of the matched gt box for each prior.
        idx: (int) current batch index.
        loc_data: (tensor) The predicted bbox regression coordinates for this batch.
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    decoded_priors = point_form(priors)


    overlaps = jaccard(truths, decoded_priors)

    # Size [num_priors] best ground truth for each prior
    overlaps = P.Cast()(overlaps, mindspore.float32)
    best_truth_idx, best_truth_overlap = P.ArgMaxWithValue(0)(overlaps)


    _, drop_pad_overlap = P.ArgMaxWithValue(1)(overlaps)
    for i in range(overlaps.shape[0]):
        if drop_pad_overlap[i] == 0:
            overlaps = overlaps[:i, :]
            break

    for _ in range(overlaps.shape[0]):
        # Find j, the gt with the highest overlap with a prior
        # In effect, this will loop through overlaps.size(0) in a "smart" order,
        # always choosing the highest overlap first.
        best_prior_idx, best_prior_overlap = P.ArgMaxWithValue(1)(overlaps)
        # best_prior_overlap, best_prior_idx = max(overlaps, 1)
        # j = best_prior_overlap.max(0)[1]

        cast = P.Cast()
        # best_prior_overlap = cast(best_prior_overlap, mindspore.float32)

        idx_j, _ = P.ArgMaxWithValue(0)(best_prior_overlap)
        # Find i, the highest overlap anchor with this gt

        # j = cast(j, mindspore.int32)
        i = best_prior_idx[idx_j]

        # i = cast(i, mindspore.int32)
        # idx_j = cast(idx_j, mindspore.int32)
        # Set all other overlaps with i to be -1 so that no other gt uses it
        overlaps[:, i] = mindspore.ops.ScalarCast()(-1, mindspore.float32)
        # Set all other overlaps with j to be -1 so that this loop never uses j again
        # overlaps[j, :] = -1
        overlaps[idx_j, :] = mindspore.ops.ScalarCast()(-1, mindspore.float32)

        best_truth_overlap[i] = mindspore.ops.ScalarCast()(2, mindspore.float32)


        best_truth_idx = cast(best_truth_idx, mindspore.float16)
        new_best_truth_idx = mindspore.ops.expand_dims(best_truth_idx, 0)
        new_best_truth_idx[::, i] = idx_j
        best_truth_idx = mindspore.ops.Squeeze()(new_best_truth_idx)

        best_truth_idx = cast(best_truth_idx, mindspore.int32)

    matches = truths[best_truth_idx]  # Shape: [num_priors,4]

    conf = labels[best_truth_idx] + 1  # Shape: [num_priors]

    conf[best_truth_overlap < neg_thresh] = 0  # label as background

    loc = encode(matches, priors)
    loc_t[idx] = loc  # [num_priors,4] encoded offsets to learn

    conf_t[idx] = conf  # [num_priors] top class label for each prior

    best_truth_idx = P.Cast()(best_truth_idx, mindspore.int32)
    idx_t[idx] = best_truth_idx  # [num_priors] indices for lookup

    return 0

class match_test(nn.Cell):
    """Match test"""
    def __init__(self):
        super(match_test, self).__init__()
        self.cast = P.Cast()
        self.argmaxwithV0 = P.ArgMaxWithValue(0)
        self.argmaxwithV1 = P.ArgMaxWithValue(1)
        self.scalarcast = P.ScalarCast()
        self.squeeze = P.Squeeze()

        self.concat = P.Concat(1)
        self.log = P.Log()
        self.squeeze = P.Squeeze(0)
        self.expand_dims = P.ExpandDims()
        shape = (1, 128, 19248)
        self.broadcast_to1 = P.BroadcastTo(shape)
        self.min = P.Minimum()
        self.max = P.Maximum()
        self.prod = P.ReduceProd()
        self.zeroslike = P.ZerosLike()
        self.select = P.Select()
        shape2 = (1, 128, 19248, 2)
        self.broadcast_to2 = P.BroadcastTo(shape2)

    def construct(self, pos_thresh, neg_thresh, truths, priors, labels, loc_t, conf_t, idx_t, idx,
                  loc_data):
        """Forward"""
        decoded_priors = self.point_form(priors)
        overlaps = self.jaccard(truths, decoded_priors)
        overlaps = self.cast(overlaps, mindspore.float32)

        _, drop_pad_overlap = self.argmaxwithV1(overlaps)
        for i in range(overlaps.shape[0]):
            if drop_pad_overlap[i] == 0:
                overlaps = overlaps[:i, :]
                break

        best_truth_idx, best_truth_overlap = self.argmaxwithV0(overlaps)
        for _ in range(overlaps.shape[0]):
            best_prior_idx, best_prior_overlap = self.argmaxwithV1(overlaps)
            idx_j, _ = self.argmaxwithV0(best_prior_overlap)
            i = best_prior_idx[idx_j]
            overlaps[:, i] = self.scalarcast(-1, mindspore.float32)
            overlaps[idx_j, :] = self.scalarcast(-1, mindspore.float32)
            best_truth_overlap[i] = self.scalarcast(2, mindspore.float32)
            best_truth_idx = self.cast(best_truth_idx, mindspore.float16)
            new_best_truth_idx = self.expand_dims(best_truth_idx, 0)
            new_best_truth_idx[::, i] = idx_j
            best_truth_idx = self.squeeze(new_best_truth_idx)
            best_truth_idx = self.cast(best_truth_idx, mindspore.int32)
        matches = truths[best_truth_idx]  # Shape: [num_priors,4]
        conf = labels[best_truth_idx] + 1  # Shape: [num_priors]
        conf[best_truth_overlap < neg_thresh] = 0  # label as background

        loc = self.encode(matches, priors)

        loc_t[idx] = loc  # [num_priors,4] encoded offsets to learn
        conf_t[idx] = conf  # [num_priors] top class label for each prior
        best_truth_idx = self.cast(best_truth_idx, mindspore.int32)
        idx_t[idx] = best_truth_idx  # [num_priors] indices for lookup

        return 0

    def point_form(self, boxes):
        return self.concat((boxes[:, :2] - boxes[:, 2:] / 2,  # xmin, ymin
                            boxes[:, :2] + boxes[:, 2:] / 2))  # xmax, ymax

    def jaccard(self, box_a, box_b, iscrowd: bool = False):
        """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
        is simply the intersection over union of two boxes.  Here we operate on
        ground truth boxes and default boxes. If iscrowd=True, put the crowd in box_b.
        E.g.:
            A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
        Args:
            box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
            box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
        Return:
            jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
        """
        use_batch = True
        if box_a.ndim == 2:
            use_batch = False
            box_a = box_a[None, ...]
            box_b = box_b[None, ...]
        inter = self.intersect(box_a, box_b)  # 1 128 4 ，1 19248，4  ---> 1 128 19248
        area_a = self.expand_dims((box_a[:, :, 2] - box_a[:, :, 0]) * (box_a[:, :, 3] - box_a[:, :, 1]), 2)

        area_a = self.broadcast_to1(area_a)
        area_b = self.expand_dims((box_b[:, :, 2] - box_b[:, :, 0]) * (box_b[:, :, 3] - box_b[:, :, 1]), 1)
        area_b = self.broadcast_to1(area_b)
        union = area_a + area_b - inter

        out = inter / area_a if iscrowd else inter / union
        return out if use_batch else self.squeeze(out)

    def intersect(self, box_a, box_b):
        """ We resize both tensors to [A,B,2] without new malloc:
        [A,2] -> [A,1,2] -> [A,B,2]
        [B,2] -> [1,B,2] -> [A,B,2]
        Then we compute the area of intersect between box_a and box_b.
        Args:
          box_a: (tensor) bounding boxes, Shape: [n,A,4].
          box_b: (tensor) bounding boxes, Shape: [n,B,4].
        Return:
          (tensor) intersection area, Shape: [n,A,B].
        """
        max_xy = self.min(self.broadcast_to2(self.expand_dims(box_a[:, :, 2:], 2)),
                          self.broadcast_to2(self.expand_dims(box_b[:, :, 2:], 1)))
        min_xy = self.max(self.broadcast_to2(self.expand_dims(box_a[:, :, :2], 2)),
                          self.broadcast_to2(self.expand_dims(box_b[:, :, :2], 1)))

        min_tensor = self.zeroslike(max_xy - min_xy)
        wants = self.select(min_tensor > max_xy - min_xy, min_tensor, max_xy - min_xy)
        return self.prod(wants, 3)

    def encode(self, matched, priors):
        """
        Encode bboxes matched with each prior into the format
        produced by the network. See decode for more details on
        this format. Note that encode(decode(x, p), p) = x.

        Args:
            - matched: A tensor of bboxes in point form with shape [num_priors, 4]
            - priors:  The tensor of all priors with shape [num_priors, 4]
        Return: A tensor with encoded relative coordinates in the format
                outputted by the network (see decode). Size: [num_priors, 4]
        """
        # Encode the coordinates corresponding to the gtbox and the coordinates corresponding to the a priori box
        variances = [0.1, 0.2]

        # dist b/t match center and prior's center
        g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
        # encode variance
        g_cxcy /= (variances[0] * priors[:, 2:])
        # match wh / prior wh
        g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
        g_wh = self.log(g_wh) / variances[1]
        # return target for smooth_l1_loss
        loc = self.concat((g_cxcy, g_wh))  # [num_priors,4]

        return loc



def encode(matched, priors):
    """
    Encode bboxes matched with each prior into the format
    produced by the network. See decode for more details on
    this format. Note that encode(decode(x, p), p) = x.

    Args:
        - matched: A tensor of bboxes in point form with shape [num_priors, 4]
        - priors:  The tensor of all priors with shape [num_priors, 4]
    Return: A tensor with encoded relative coordinates in the format
            outputted by the network (see decode). Size: [num_priors, 4]
    """
    # Encode the coordinates corresponding to the gtbox and the coordinates corresponding to the a priori box
    variances = [0.1, 0.2]

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = P.Log()(g_wh) / variances[1]
    # return target for smooth_l1_loss
    loc = P.Concat(1)((g_cxcy, g_wh))  # [num_priors,4]

    return loc

def decode(loc, priors, use_yolo_regressors: bool = False):
    """
    Decode predicted bbox coordinates using the same scheme
    employed by Yolov2: https://arxiv.org/pdf/1612.08242.pdf

        b_x = (sigmoid(pred_x) - .5) / conv_w + prior_x
        b_y = (sigmoid(pred_y) - .5) / conv_h + prior_y
        b_w = prior_w * exp(loc_w)
        b_h = prior_h * exp(loc_h)

    Note that loc is inputted as [(s(x)-.5)/conv_w, (s(y)-.5)/conv_h, w, h]
    while priors are inputted as [x, y, w, h] where each coordinate
    is relative to size of the image (even sigmoid(x)). We do this
    in the network by dividing by the 'cell size', which is just
    the size of the convouts.

    Also note that prior_x and prior_y are center coordinates which
    is why we have to subtract .5 from sigmoid(pred_x and pred_y).

    Args:
        - loc:    The predicted bounding boxes of size [num_priors, 4]
        - priors: The priorbox coords with size [num_priors, 4]

    Returns: A tensor of decoded relative coordinates in point form
             form with size [num_priors, 4]
    """

    if use_yolo_regressors:
        # Decoded boxes in center-size notation
        concat = P.Concat(1)
        exp = P.Exp()
        boxes = concat((
            loc[:, :2] + priors[:, :2],
            priors[:, 2:] * exp(loc[:, 2:])
        ))

        boxes = point_form(boxes)
    else:
        variances = [0.1, 0.2]
        concat = P.Concat(1)
        exp = P.Exp()
        boxes = concat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, 2:] * exp(loc[:, 2:] * variances[1])))
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]

    return boxes


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    log_reduce_sum = P.ReduceSum()
    log = P.Log()
    exp = P.Exp()
    x_max = max(x.data)
    return log(log_reduce_sum(exp(x - x_max), 1)) + x_max



def sanitize_coordinates(_x1, _x2, img_size: int, padding: int = 0, cast: bool = True):
    """
    Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
    Also converts from relative to absolute coordinates and casts the results to long tensors.

    If cast is false, the result won't be cast to longs.
    Warning: this does things in-place behind the scenes so copy if necessary.
    """
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size
    if cast:
        _x1 = _x1.long()
        _x2 = _x2.long()
    coordinates_min = P.Minimum()
    coordinates_max = P.Maximum()
    x1 = coordinates_min(_x1, _x2)
    x2 = coordinates_max(_x1, _x2)

    select = P.Select()
    zeroslike = P.ZerosLike()
    oneslike = P.OnesLike()
    min_tensor = zeroslike(x1 - padding)
    x1 = select(min_tensor > x1 - padding, min_tensor, x1 - padding)

    max_tensor = oneslike(x2 + padding) * img_size
    x2 = select(x2 + padding > max_tensor, max_tensor, x2 + padding)


    return x1, x2


def crop(masks, boxes, padding: int = 1):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """
    h, w, n = masks.shape
    x1, x2 = sanitize_coordinates(boxes[:, 0:1:1], boxes[:, 2:3:1], w, padding, cast=False)
    y1, y2 = sanitize_coordinates(boxes[:, 1:2:1], boxes[:, 3:4:1], h, padding, cast=False)

    cast = P.Cast()
    crop_range = nn.Range(w)
    broadcast_to = P.BroadcastTo((h, w, n))
    row = broadcast_to((crop_range().view(1, -1, 1)))
    rows = cast(row, x1.dtype)
    col = broadcast_to((crop_range().view(-1, 1, 1)))
    cols = cast(col, x2.dtype)


    masks_left = rows >= x1.view(1, 1, -1)
    masks_right = rows < x2.view(1, 1, -1)
    masks_left = P.Cast()(masks_left, mindspore.float16)
    masks_right = P.Cast()(masks_right, mindspore.float16)
    crop_mask = masks_left * masks_right
    masks_up = cols >= y1.view(1, 1, -1)
    masks_up = P.Cast()(masks_up, mindspore.float16)
    crop_mask *= masks_up
    masks_down = cols < y2.view(1, 1, -1)
    masks_down = P.Cast()(masks_down, mindspore.float16)
    crop_mask *= masks_down

    return masks * crop_mask


def index2d(src, idx):
    """
    Indexes a tensor by a 2d index.

    In effect, this does
        out[i, j] = src[i, idx[i, j]]

    Both src and idx should have the same size.
    """
    temp_range = nn.Range(idx.shape[0])
    broadcast_to = P.BroadcastTo(idx.shape)
    offs = broadcast_to(temp_range()[:, None])
    idx = idx + (offs()) * idx.shape[1]

    return src.view(-1)[idx.view(-1)].view(idx.shpe)
