# Copyright 2023 Huawei Technologies Co., Ltd
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
# =======================================================================================

import numpy as np


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    box_corner = np.zeros_like(prediction)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]

    for i, image_pred in enumerate(prediction):
        if not image_pred.shape[0]:
            continue
        # Get score and class with highest confidence
        class_conf = np.max(image_pred[:, 5:5 + num_classes], axis=-1)
        class_pred = np.argmax(image_pred[:, 5:5 + num_classes], axis=-1)
        conf_mask = image_pred[:, 4] * class_conf >= conf_thre

        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = np.concatenate((image_pred[:, :5], class_conf[:, None], class_pred[:, None]), axis=1)
        detections = detections[conf_mask]

        if not detections.shape[0]:
            continue
        if class_agnostic:
            nms_out_index = nms(detections[:, :4],
                                detections[:, 4] * detections[:, 5],
                                nms_thre)
        else:
            nms_out_index = batch_nms(detections[:, :4],
                                      detections[:, 4] * detections[:, 5],
                                      detections[:, 6],
                                      nms_thre)
        detections = detections[nms_out_index]

        if output[i] is None:
            output[i] = detections
        else:
            output[i] = np.concatenate((output[i], detections))

    return output


def nms(xyxys, scores, threshold):
    x1 = xyxys[:, 0]
    y1 = xyxys[:, 1]
    x2 = xyxys[:, 2]
    y2 = xyxys[:, 3]
    scores = scores
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    reserved_boxes = []
    while order.size > 0:
        i = order[0]
        reserved_boxes.append(i)
        max_x1 = np.maximum(x1[i], x1[order[1:]])
        max_y1 = np.maximum(y1[i], y1[order[1:]])
        min_x2 = np.minimum(x2[i], x2[order[1:]])
        min_y2 = np.minimum(y2[i], y2[order[1:]])

        intersect_w = np.maximum(0.0, min_x2 - max_x1 + 1)
        intersect_h = np.maximum(0.0, min_y2 - max_y1 + 1)
        intersect_area = intersect_w * intersect_h

        ovr = intersect_area / (areas[i] + areas[order[1:]] - intersect_area)
        indexes = np.where(ovr <= threshold)[0]
        order = order[indexes + 1]
    return reserved_boxes


def batch_nms(xyxys, scores, idxs, threshold, use_offset=True):
    if use_offset:
        max_coordinate = xyxys.max()
        offsets = idxs * (max_coordinate + np.array([1]))
        boxes_for_nms = xyxys + offsets[:, None]
        keep = nms(boxes_for_nms, scores, threshold)
        return keep

    keep_mask = np.zeros_like(scores, dtype=np.bool_)
    for class_id in np.unique(idxs):
        curr_indices = np.where(idxs == class_id)[0]
        curr_keep_indices = nms(xyxys[curr_indices], scores[curr_indices], threshold)
        keep_mask[curr_indices[curr_keep_indices]] = True
    keep_indices = np.where(keep_mask)[0]
    return keep_indices[np.argsort(-scores[keep_indices])]
