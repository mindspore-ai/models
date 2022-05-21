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
"""Auxiliary utils."""
import os

import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import numpy as msnp
from mindspore import ops
from mindspore.ops import functional as F


def mkdir_if_missing(directory):
    os.makedirs(directory, exist_ok=True)


def xyxy2xywh(x):
    """
    Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h],
    where x, y are coordinates of center, (x1, y1) and (x2, y2)
    are coordinates of bottom left and top right respectively.
    """
    y = np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    """
    Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2],
    where x, y are coordinates of center, (x1, y1) and (x2, y2)
    are coordinates of bottom left and top right respectively.
    """
    y = np.zeros_like(x)
    y[:, 0] = (x[:, 0] - x[:, 2] / 2)  # Bottom left x
    y[:, 1] = (x[:, 1] - x[:, 3] / 2)  # Bottom left y
    y[:, 2] = (x[:, 0] + x[:, 2] / 2)  # Top right x
    y[:, 3] = (x[:, 1] + x[:, 3] / 2)  # Top right y
    return y


def scale_coords(img_size, coords, img0_shape):
    """
    Rescale x1, y1, x2, y2 to image size.
    """
    gain_w = float(img_size[0]) / img0_shape[1]  # gain  = old / new
    gain_h = float(img_size[1]) / img0_shape[0]
    gain = min(gain_w, gain_h)
    pad_x = (img_size[0] - img0_shape[1] * gain) / 2  # width padding
    pad_y = (img_size[1] - img0_shape[0] * gain) / 2  # height padding
    coords[:, [0, 2]] -= pad_x
    coords[:, [1, 3]] -= pad_y
    coords[:, 0:4] /= gain
    cords_max = np.max(coords[:, :4])
    coords[:, :4] = np.clip(coords[:, :4], a_min=0, a_max=cords_max)
    return coords


class SoftmaxCE(nn.Cell):
    """
    Original nn.SoftmaxCrossEntropyWithLogits with modifications:
    1) Set ignore index = -1.
    2) Reshape labels and logits to (n, C).
    3) Calculate mean by mask.
    """
    def __init__(self):
        super().__init__()
        # Set necessary operations and constants
        self.soft_ce = ops.SoftmaxCrossEntropyWithLogits()
        self.expand_dim = ops.ExpandDims()
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()
        self.one_hot = ops.OneHot()
        self.sum = ops.ReduceSum()
        self.one = Tensor(1, mstype.float32)
        self.zero = Tensor(0, mstype.float32)

        # Set eps to escape division by zero
        self.eps = Tensor(1e-16, dtype=mstype.float32)

    def construct(self, logits, labels, ignore_index):
        """
        Calculate softmax loss between logits and labels with ignore mask.
        """
        # Ignore indices which have not exactly recognized iou
        mask = labels != ignore_index
        mask = mask.astype('float32')
        channels = F.shape(logits)[-1]

        # One-hot labels for total identities in dataset
        labels_one_hot = self.one_hot(labels.flatten(), channels, self.one, self.zero)
        raw_loss, _ = self.soft_ce(
            self.reshape(logits, (-1, channels)),
            self.reshape(labels_one_hot, (-1, channels)),
        )

        # Apply mask and take mean of losses
        result = raw_loss * mask.reshape(raw_loss.shape)
        result = self.sum(result) / (self.sum(mask) + self.eps)

        return result


def build_targets_thres(target, anchor_wh, na, ngh, ngw, k_max):
    """
    Build grid of targets confidence mask, bbox delta and id with thresholds.

    Args:
        target (np_array): Targets bbox cords and ids.
        anchor_wh (np_array): Resized anchors for map size.
        na (int): Number of anchors.
        ngh (int): Map height.
        ngw (int): Map width.
        k_max (int): Limitation of max detections per image.

    Returns:
        tconf (np_array): Mask with bg (0), gt (1) and ign (-1) indices. Shape (na, ngh, ngw).
        tbox (np_array): Targets delta bbox values. Shape (na, ngh, ngw, 4).
        tid (np_array): Grid with id for every cell. Shape (na, ngh, ngw).

    """
    id_thresh = 0.5
    fg_thresh = 0.5
    bg_thresh = 0.4

    bg_id = -1  # Background id

    tbox = np.zeros((na, ngh, ngw, 4), dtype=np.float32)  # Fill grid with zeros bbox cords
    tconf = np.zeros((na, ngh, ngw), dtype=np.int32)  # Fill grid with zeros confidence
    tid = np.full((na, ngh, ngw), bg_id, dtype=np.int32)  # Fill grid with background id

    t = target
    t_id = t[:, 1].copy().astype(np.int32)
    t = t[:, [0, 2, 3, 4, 5]]

    # Convert relative cords for map size
    gxy, gwh = t[:, 1:3].copy(), t[:, 3:5].copy()
    gxy[:, 0] = gxy[:, 0] * ngw
    gxy[:, 1] = gxy[:, 1] * ngh
    gwh[:, 0] = gwh[:, 0] * ngw
    gwh[:, 1] = gwh[:, 1] * ngh
    gxy[:, 0] = np.clip(gxy[:, 0], a_min=0, a_max=ngw - 1)
    gxy[:, 1] = np.clip(gxy[:, 1], a_min=0, a_max=ngh - 1)

    gt_boxes = np.concatenate((gxy, gwh), axis=1)  # Shape (num of targets, 4), 4 is (xc, yc, w, h)

    # Apply anchor to each cell of the grid
    anchor_mesh = generate_anchor(ngh, ngw, anchor_wh)  # Shape (na, 4, ngh, ngw)
    anchor_list = anchor_mesh.transpose(0, 2, 3, 1).reshape(-1, 4)  # Shape (na x ngh x ngw, 4)

    # Compute anchor iou with ground truths bboxes
    iou_pdist = bbox_iou(anchor_list, gt_boxes)  # Shape (na x ngh x ngw, Ng)
    max_gt_index = iou_pdist.argmax(axis=1)   # Shape (na x ngh x ngw)
    iou_max = iou_pdist.max(axis=1)   # Shape (na x ngh x ngw)

    iou_map = iou_max.reshape(na, ngh, ngw)
    gt_index_map = max_gt_index.reshape(na, ngh, ngw)

    # Fill tconf by thresholds
    id_index = iou_map > id_thresh
    fg_index = iou_map > fg_thresh
    bg_index = iou_map < bg_thresh
    ign_index = (iou_map < fg_thresh) * (iou_map > bg_thresh)  # Search unclear cells
    tconf[fg_index] = 1
    tconf[bg_index] = 0
    tconf[ign_index] = -1  # Index to ignore unclear cells

    # Take ground truths with mask
    gt_index = gt_index_map[fg_index]
    gt_box_list = gt_boxes[gt_index]
    gt_id_list = t_id[gt_index_map[id_index]]
    if np.sum(fg_index) > 0:
        tid[id_index] = gt_id_list
        fg_anchor_list = anchor_list.reshape((na, ngh, ngw, 4))[fg_index]
        delta_target = encode_delta(gt_box_list, fg_anchor_list)
        tbox[fg_index] = delta_target

    # Indices of cells with detections
    tconf_max = tconf.max(0)
    tid_max = tid.max(0)
    indices = np.where((tconf_max.flatten() > 0) & (tid_max.flatten() >= 0))[0]

    # Fill indices with zeros if k < k_max
    # Where k - is the detections per image
    # k_max - max detections per image
    k = len(indices)
    t_indices = np.zeros(k_max)
    t_indices[..., :min(k_max, k)] = indices[..., :min(k_max, k)]

    return tconf, tbox, tid, t_indices


def bbox_iou(box1, box2, x1y1x2y2=False):
    """
    Returns the IoU of two bounding boxes.
    """
    n, m = len(box1), len(box2)
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # Get the coordinates of the intersection rectangle
    inter_rect_x1 = np.maximum(np.expand_dims(b1_x1, 1), b2_x1)
    inter_rect_y1 = np.maximum(np.expand_dims(b1_y1, 1), b2_y1)
    inter_rect_x2 = np.minimum(np.expand_dims(b1_x2, 1), b2_x2)
    inter_rect_y2 = np.minimum(np.expand_dims(b1_y2, 1), b2_y2)

    # Intersection area
    i_r_x = inter_rect_x2 - inter_rect_x1
    i_r_y = inter_rect_y2 - inter_rect_y1
    inter_area = np.clip(i_r_x, 0, np.max(i_r_x)) * np.clip(i_r_y, 0, np.max(i_r_y))

    # Union Area
    b1_area = np.broadcast_to(((b1_x2 - b1_x1) * (b1_y2 - b1_y1)).reshape(-1, 1), (n, m))
    b2_area = np.broadcast_to(((b2_x2 - b2_x1) * (b2_y2 - b2_y1)).reshape(1, -1), (n, m))

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)


def generate_anchor(ngh, ngw, anchor_wh):
    """
    Generate anchor for every cell in grid.
    """
    na = len(anchor_wh)
    yy, xx = np.meshgrid(np.arange(ngh), np.arange(ngw), indexing='ij')

    mesh = np.stack([xx, yy], axis=0)  # Shape 2, ngh, ngw
    mesh = np.tile(np.expand_dims(mesh, 0), (na, 1, 1, 1)).astype(np.float32)  # Shape na, 2, ngh, ngw
    anchor_offset_mesh = np.tile(np.expand_dims(np.expand_dims(anchor_wh, -1), -1), (1, 1, ngh, ngw))  # Shape na, 2, ngh, ngw
    anchor_mesh = np.concatenate((mesh, anchor_offset_mesh), axis=1)  # Shape na, 4, ngh, ngw
    return anchor_mesh


def encode_delta(gt_box_list, fg_anchor_list):
    """
    Calculate delta for bbox center, width, height.
    """
    px, py, pw, ph = fg_anchor_list[:, 0], fg_anchor_list[:, 1], \
                     fg_anchor_list[:, 2], fg_anchor_list[:, 3]
    gx, gy, gw, gh = gt_box_list[:, 0], gt_box_list[:, 1], \
                     gt_box_list[:, 2], gt_box_list[:, 3]
    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = np.log(gw / pw)
    dh = np.log(gh / ph)

    return np.stack([dx, dy, dw, dh], axis=1)


def create_grids(anchors, img_size, ngw):
    """
    Resize anchor according to image size and feature map size.

    Note:
        Ratio of feature maps dimensions if 1:3 such as anchors.
        Thus, it's enough to calculate stride per one dimension.
    """
    stride = img_size[0] / ngw
    anchor_vec = np.array(anchors) / stride

    return anchor_vec, stride


def build_thresholds(
        labels,
        anchor_vec_s,
        anchor_vec_m,
        anchor_vec_b,
        k_max,
):
    """
    Build thresholds for all feature map sizes.
    """
    s = build_targets_thres(labels, anchor_vec_s, 4, 19, 34, k_max)
    m = build_targets_thres(labels, anchor_vec_m, 4, 38, 68, k_max)
    b = build_targets_thres(labels, anchor_vec_b, 4, 76, 136, k_max)

    return s, m, b


def create_anchors_vec(anchors, img_size=(1088, 608)):
    """
    Create anchor vectors for every feature map size.
    """
    anchors1 = anchors[0:4]
    anchors2 = anchors[4:8]
    anchors3 = anchors[8:12]
    anchor_vec_s, stride_s = create_grids(anchors3, img_size, 34)
    anchor_vec_m, stride_m = create_grids(anchors2, img_size, 68)
    anchor_vec_b, stride_b = create_grids(anchors1, img_size, 136)

    anchors = (anchor_vec_s, anchor_vec_m, anchor_vec_b)
    strides = (stride_s, stride_m, stride_b)

    return anchors, strides


class DecodeDeltaMap(nn.Cell):
    """
    Network predicts delta for base anchors.

    Decodes predictions into relative bbox cords.
    """
    def __init__(self):
        super().__init__()
        self.exp = ops.operations.Exp()
        self.stack0 = ops.Stack(axis=0)
        self.stack1 = ops.Stack(axis=1)
        self.expand_dims = ops.ExpandDims()
        self.reshape = ops.Reshape()
        self.concat = ops.Concat(axis=2)

    def construct(self, delta_map, anchors):
        """
        Decode delta of bbox predictions and summarize it with anchors.
        """
        anchors = anchors.astype('float32')
        nb, na, ngh, ngw, _ = delta_map.shape
        yy, xx = msnp.meshgrid(msnp.arange(ngh), msnp.arange(ngw), indexing='ij')

        mesh = self.stack0([xx, yy]).astype('float32')  # Shape (2, ngh, ngw)
        mesh = msnp.tile(self.expand_dims(mesh, 0), (nb, na, 1, 1, 1))  # Shape (nb, na, 2, ngh, ngw)
        anchors_unsqueezed = self.expand_dims(self.expand_dims(anchors, -1), -1)  # Shape (na, 2, 1, 1)
        anchor_offset_mesh = msnp.tile(anchors_unsqueezed, (nb, 1, 1, ngh, ngw))  # Shape (nb, na, 2, ngh, ngw)
        anchor_mesh = self.concat((mesh, anchor_offset_mesh))  # Shape (nb, na, 4, ngh, ngw)

        anchor_mesh = anchor_mesh.transpose(0, 1, 3, 4, 2)

        delta = delta_map.reshape(-1, 4)
        fg_anchor_list = anchor_mesh.reshape(-1, 4)
        px, py, pw, ph = fg_anchor_list[:, 0], fg_anchor_list[:, 1], \
                         fg_anchor_list[:, 2], fg_anchor_list[:, 3]
        dx, dy, dw, dh = delta[:, 0], delta[:, 1], delta[:, 2], delta[:, 3]
        gx = pw * dx + px
        gy = ph * dy + py
        gw = pw * self.exp(dw)
        gh = ph * self.exp(dh)

        pred_list = self.stack1([gx, gy, gw, gh])

        pred_map = pred_list.reshape(nb, na, ngh, ngw, 4)

        return pred_map


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.

    Args:
        prediction (np.array): All predictions from model output.
        conf_thres (float): Threshold for confidence.
        nms_thres (float): Threshold for iou into nms.

    Returns:
        output (np.array): Predictions with shape (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    output = [None for _ in range(len(prediction))]
    for image_i, pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        # Get score and class with highest confidence

        v = pred[:, 4] > conf_thres
        v = np.squeeze(v.nonzero())
        if v.ndim == 0:
            v = np.expand_dims(v, 0)

        pred = pred[v]

        # If none are remaining => process next image
        npred = pred.shape[0]
        if not npred:
            continue
        # From (center x, center y, width, height) to (x1, y1, x2, y2)
        pred[:, :4] = xywh2xyxy(pred[:, :4])

        # Non-maximum suppression
        bboxes = np.concatenate((pred[:, :4], np.expand_dims(pred[:, 4], -1)), axis=1)
        nms_indices = nms(bboxes, nms_thres)
        det_max = pred[nms_indices]

        if det_max.size > 0:
            # Add max detections to outputs
            output[image_i] = det_max if output[image_i] is None else np.concatenate((output[image_i], det_max))

    return output


def nms(dets, thresh):
    """
    Non-maximum suppression with threshold.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def ap_per_class(tp, conf, pred_cls, target_cls):
    """
    Computes the average precision, given the recall and precision curves.
    Method originally from https://github.com/rafaelpadilla/Object-Detection-Metrics.

    Args:
        tp (list): True positives.
        conf (list): Objectness value from 0-1.
        pred_cls (np.array): Predicted object classes.
        target_cls (np.array): True object classes.

    Returns:
        ap (np.array): The average precision as computed in py-faster-rcnn.
        unique classes (np.array): Classes of predictions.
        r (np.array): Recall.
        p (np.array): Precision.
    """

    # lists/pytorch to numpy
    tp, conf, pred_cls, target_cls = np.array(tp), np.array(conf), np.array(pred_cls), np.array(target_cls)

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(np.concatenate((pred_cls, target_cls), 0))

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = sum(target_cls == c)  # Number of ground truth objects
        n_p = sum(i)  # Number of predicted objects

        if (n_p == 0) and (n_gt == 0):
            continue

        if (n_p == 0) or (n_gt == 0):
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = np.cumsum(1 - tp[i])
            tpc = np.cumsum(tp[i])

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(tpc[-1] / (n_gt + 1e-16))

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(tpc[-1] / (tpc[-1] + fpc[-1]))

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    return np.array(ap), unique_classes.astype('int32'), np.array(r), np.array(p)


def compute_ap(recall, precision):
    """
    Computes the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    Args:
        recall (list): The recall curve.
        precision (list): The precision curve.

    Returns:
        ap (np.array): The average precision as computed in py-faster-rcnn.
    """

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
