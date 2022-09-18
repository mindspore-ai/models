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

# This file was copied from project [ZhaoWeicheng][Pyramidbox.pytorch]

import numpy as np


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: center-size default boxes from priorbox layers.
    Return:
        boxes: Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return np.concatenate((boxes[:, :2] - boxes[:, 2:] / 2,
                           boxes[:, :2] + boxes[:, 2:] / 2), 1)

def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: point_form boxes
    Return:
        boxes: Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return np.concatenate([(boxes[:, 2:] + boxes[:, :2]) / 2,
                           boxes[:, 2:] - boxes[:, :2]], 1)

def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: bounding boxes, Shape: [A,4].
      box_b: bounding boxes, Shape: [B,4].
    Return:
      intersection area, Shape: [A,B].
    """
    A = box_a.shape[0]
    B = box_b.shape[0]

    max_xy = np.minimum(np.broadcast_to(np.expand_dims(box_a[:, 2:], 1), (A, B, 2)),
                        np.broadcast_to(np.expand_dims(box_b[:, 2:], 0), (A, B, 2)))
    min_xy = np.maximum(np.broadcast_to(np.expand_dims(box_a[:, :2], 1), (A, B, 2)),
                        np.broadcast_to(np.expand_dims(box_b[:, :2], 0), (A, B, 2)))
    inter = np.clip((max_xy - min_xy), 0, np.inf)
    return inter[:, :, 0] * inter[:, :, 1]

def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))
    area_a = np.expand_dims(area_a, 1)
    area_a = np.broadcast_to(area_a, inter.shape)

    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1]))
    area_b = np.expand_dims(area_b, 0)
    area_b = np.broadcast_to(area_b, inter.shape)

    union = area_a + area_b - inter

    return inter / union

def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when matching boxes.
        truths: Ground truth boxes, Shape: [num_obj, num_priors].
        priors: Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: All the class labels for the image, Shape: [num_obj].
        loc_t: Tensor to be filled w/ encoded location targets.
        conf_t: Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    overlaps = jaccard(truths, point_form(priors))

    # best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    best_prior_overlap = np.max(overlaps, 1, keepdims=True)
    best_prior_idx = np.argmax(overlaps, 1)

    best_truth_overlap = np.max(overlaps, 0, keepdims=True)
    best_truth_idx = np.argmax(overlaps, 0)

    best_truth_idx = np.squeeze(best_truth_idx, 0)
    best_truth_overlap = np.squeeze(best_truth_overlap, 0)

    best_prior_idx = np.squeeze(best_prior_idx, 1)
    best_prior_overlap = np.squeeze(best_prior_overlap, 1)

    for i in best_prior_idx:
        best_truth_overlap[i, :] = 2

    for j in range(best_prior_idx.shape[0]):
        best_truth_idx[best_prior_idx[j]] = j

    _th1, _th2, _th3 = threshold

    N = (np.sum(best_prior_overlap >= _th2) +
         np.sum(best_prior_overlap >= _th3)) // 2

    matches = truths[best_truth_idx]
    conf = labels[best_truth_idx]
    conf[best_truth_overlap < _th2] = 0

    best_truth_overlap_clone = best_truth_overlap.copy()
    idx_1 = np.greater(best_truth_overlap_clone, _th1)
    idx_2 = np.less(best_truth_overlap_clone, _th2)
    add_idx = np.equal(idx_1, idx_2)

    best_truth_overlap_clone[1 - add_idx] = 0
    stage2_overlap = np.sort(best_truth_overlap_clone)[:, ::-1]
    stage2_idx = np.argsort(best_truth_overlap_clone)[:, ::-1]

    stage2_overlap = np.greater(stage2_overlap, _th1)

    if N > 0:
        N = np.sum(stage2_overlap[:N]) if np.sum(stage2_overlap[:N]) < N else N
        conf[stage2_idx[:N]] += 1

    loc = encode(matches, priors, variances)
    loc_t[idx] = loc
    conf_t[idx] = conf


def match_ssd(threshold, truths, priors, variances, labels):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when matching boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ encoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    overlaps = jaccard(truths, point_form(priors))

    # best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    best_prior_overlap = np.max(overlaps, 1, keepdims=True)
    best_prior_idx = np.argmax(overlaps, 1)

    best_truth_overlap = np.max(overlaps, 0, keepdims=True)
    best_truth_idx = np.argmax(overlaps, 0)

    best_truth_overlap = np.squeeze(best_truth_overlap, 0)
    best_prior_overlap = np.squeeze(best_prior_overlap, 1)

    for i in best_prior_idx:
        best_truth_overlap[i] = 2

    for j in range(best_prior_idx.shape[0]):
        best_truth_idx[best_prior_idx[j]] = j

    matches = truths[best_truth_idx]
    conf = labels[best_truth_idx]
    conf[best_truth_overlap < threshold] = 0
    loc = encode(matches, priors, variances)

    return loc, conf

def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = np.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return np.concatenate([g_cxcy, g_wh], 1)


def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc: location predictions for loc layers,
            Shape: [num_priors,4]
        priors: Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    if priors.shape[0] == 1:
        priors = priors[0, :, :]
    boxes = np.concatenate((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                            priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
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
    x_max = x.max()
    return np.log(np.sum(np.exp(x - x_max), 1, keepdim=True)) + x_max


def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: The location preds for the img, Shape: [num_priors,4].
        scores: The class predscores for the img, Shape:[num_priors].
        overlap: The overlap thresh for suppressing unnecessary boxes.
        top_k: The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    keep = np.zeros_like(scores).astype(np.int32)
    if boxes.size == 0:
        return keep, 0
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = np.multiply(x2 - x1, y2 - y1)
    idx = np.argsort(scores, axis=0)

    idx = idx[-top_k:]

    count = 0
    while idx.size > 0:
        i = idx[-1]
        keep[count] = i
        count += 1
        if idx.shape[0] == 1:
            break
        idx = idx[:-1]
        xx1 = x1[idx]
        yy1 = y1[idx]
        xx2 = x2[idx]
        yy2 = y2[idx]

        xx1 = np.clip(xx1, x1[i], np.inf)
        yy1 = np.clip(yy1, y1[i], np.inf)
        xx2 = np.clip(xx2, -np.inf, x2[i])
        yy2 = np.clip(yy2, -np.inf, y2[i])

        w = xx2 - xx1
        h = yy2 - yy1

        w = np.clip(w, 0, np.inf)
        h = np.clip(h, 0, np.inf)
        inter = w * h

        rem_areas = area[idx]
        union = (rem_areas - inter) + area[i]
        IoU = inter / union

        idx = idx[np.less(IoU, overlap)]

    return keep, count
