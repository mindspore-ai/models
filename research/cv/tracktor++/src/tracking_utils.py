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
"""Tracking utils"""

import mindspore as ms
from mindspore import nn
import motmetrics as mm
import numpy as np

from src.model_utils.config import config


def get_center(pos):
    """Get center of bounding box."""
    x1 = pos[0, 0]
    y1 = pos[0, 1]
    x2 = pos[0, 2]
    y2 = pos[0, 3]
    return np.asarray([(x2 + x1) / 2, (y2 + y1) / 2])


def get_width(pos):
    """Get width of bounding box"""
    return pos[0, 2] - pos[0, 0]


def get_height(pos):
    """Get height of bounding box"""
    return pos[0, 3] - pos[0, 1]


def make_pos(cx, cy, width, height):
    """Convert bbox coords of bounding boxes [cx, cy, w, h] -> [x1, y1, x2, y2]"""
    return np.asarray([[
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2
    ]])


def warp_pos(pos, warp_matrix):
    """Warp bbox coordinates using warp matrix."""
    p1 = np.reshape(np.asarray([pos[0, 0], pos[0, 1], 1]), (3, 1))
    p2 = np.reshape(np.asarray([pos[0, 2], pos[0, 3], 1]), (3, 1))
    p1_n = np.reshape(np.matmul(warp_matrix, p1), (1, 2))
    p2_n = np.reshape(np.matmul(warp_matrix, p2), (1, 2))
    return np.reshape(np.concatenate((p1_n, p2_n), 1), (1, -1))


def get_mot_accum(results, seq_loader):
    """Accumulate tracking results for MOT metrics evaluation."""
    mot_accum = mm.MOTAccumulator(auto_id=True)

    for i, data in enumerate(seq_loader):
        gt = data['gt']
        gt_ids = []
        if gt:
            gt_boxes = []
            for gt_id, box in gt.items():
                gt_ids.append(gt_id)
                gt_boxes.append(box[0])

            gt_boxes = np.stack(gt_boxes, axis=0)
            # x1, y1, x2, y2 --> x1, y1, width, height
            gt_boxes = np.stack((gt_boxes[:, 0],
                                 gt_boxes[:, 1],
                                 gt_boxes[:, 2] - gt_boxes[:, 0],
                                 gt_boxes[:, 3] - gt_boxes[:, 1]),
                                axis=1)
        else:
            gt_boxes = np.array([])

        track_ids = []
        track_boxes = []
        for track_id, frames in results.items():
            if i in frames:
                track_ids.append(track_id)
                # frames = x1, y1, x2, y2, score
                track_boxes.append(frames[i][:4])

        if track_ids:
            track_boxes = np.stack(track_boxes, axis=0)
            # x1, y1, x2, y2 --> x1, y1, width, height
            track_boxes = np.stack(
                (
                    track_boxes[:, 0],
                    track_boxes[:, 1],
                    track_boxes[:, 2] - track_boxes[:, 0],
                    track_boxes[:, 3] - track_boxes[:, 1],
                ),
                axis=1,
            )
        else:
            track_boxes = np.array([])

        distance = mm.distances.iou_matrix(gt_boxes, track_boxes, max_iou=0.5)

        mot_accum.update(
            gt_ids,
            track_ids,
            distance)

    return mot_accum


def evaluate_mot_accums(accums, names, generate_overall=False):
    """Evaluate accumulated MOT results"""
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accums,
        metrics=mm.metrics.motchallenge_metrics,
        names=names,
        generate_overall=generate_overall,)

    str_summary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names,)
    print(str_summary)


def clip_boxes_to_image(boxes, size):
    """
    Clip boxes so that they lie inside an image of size `size`.

    Args:
        boxes (Tensor[N, 4]): boxes in ``(x1, y1, x2, y2)`` format
            with ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        size (Tuple[height, width]): size of the image

    Returns:
        Tensor[N, 4]: clipped boxes
    """
    dim = boxes.ndim
    boxes_x = boxes[..., 0::2]
    boxes_y = boxes[..., 1::2]
    height, width = size

    boxes_x = np.clip(boxes_x, 0, width)
    boxes_y = np.clip(boxes_y, 0, height)

    clipped_boxes = np.stack((boxes_x, boxes_y), axis=dim)
    return np.reshape(clipped_boxes, boxes.shape)


def nms(dets, scores, thresh):
    """Numpy based NMS"""
    keep = []
    if dets.size == 0:
        return keep

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1] # get boxes with more ious first

    while order.size > 0:
        i = order[0]  # pick maximum iou box
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)  # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1)  # maximum height
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return np.asarray(keep)


class SingleModelFasterRCNN(nn.Cell):
    """
    FasterRcnn Network wrapper. Using numpy arrays and suitable for tracking module.

    Args:
        feature_extractor (nn.Cell): Feature Extractor of Faster RCNN
        inference_head (nn.Cell): RCNN Head of Faster RCNN
        preprocessing_function (Callable): Preprocessing function for images.
    """

    def __init__(self, feature_extractor, inference_head, preprocessing_function):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.preprocessing_function = preprocessing_function
        self.inference_head = inference_head

    def preprocess_bbox(self, boxes, img_metas):
        """Preprocess boxes for detector."""
        if boxes.size == 0:
            return (
                ms.Tensor(np.zeros((config.rpn_max_num, 5), dtype=np.float32)),
                ms.Tensor(np.zeros(config.rpn_max_num, dtype=np.bool_)),
                boxes,
            )
        # Boxes shape is [N, 4]
        # Proposal shape Tuple([1, 1000, 5])
        boxes = boxes.astype(np.float32)
        # Rescale and clip bboxes coords
        boxes[:, ::2] *= img_metas[3]
        boxes[:, 1::2] *= img_metas[2]
        boxes[:, ::2] = np.clip(boxes[:, ::2], a_min=0, a_max=img_metas[1] * img_metas[3])
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], a_min=0, a_max=img_metas[0] * img_metas[2])

        # Prepare proposal and mask for FasterRCNN head
        proposal = np.zeros((config.rpn_max_num, 5))
        proposal[:boxes.shape[0]] = np.concatenate([boxes, np.ones((boxes.shape[0], 1))], axis=-1)
        proposal_mask = np.zeros(config.rpn_max_num, dtype=np.bool_)
        proposal_mask[:boxes.shape[0]] = True
        proposal = ms.Tensor(proposal, dtype=ms.float32)
        proposal_mask = ms.Tensor(proposal_mask)
        return proposal, proposal_mask, boxes

    def postprocess_bbox(self, output, boxes):
        """Postprocess results from detector."""
        if boxes.size == 0:
            return np.zeros(0), np.zeros(0)
        output_boxes, output_labels, output_masks = output
        output_boxes, output_masks = output_boxes.asnumpy(), output_masks.asnumpy().astype(np.bool)
        output_labels = output_labels.asnumpy()[output_masks][:, 1]
        result_boxes = output_boxes[output_masks]
        result_padded_boxes = np.zeros(boxes.shape)
        result_padded_scores = np.zeros((boxes.shape[0],))
        result_padded_boxes[:result_boxes.shape[0]] = result_boxes
        result_padded_scores[:result_boxes.shape[0]] = output_labels

        return result_padded_boxes, result_padded_scores

    def process_images_and_boxes(self, images, boxes_1, boxes_2):
        """Process image and generate boxes from two lists of proposals"""
        input_tensor, img_metas = self.preprocessing_function(images)

        proposal_1, proposal_mask_1, boxes_1 = self.preprocess_bbox(boxes_1, img_metas)
        proposal_2, proposal_mask_2, boxes_2 = self.preprocess_bbox(boxes_2, img_metas)

        img_metas = ms.Tensor(img_metas)
        output_1, output_2 = self(
            input_tensor,
            proposal_1,
            proposal_mask_1,
            proposal_2,
            proposal_mask_2,
            img_metas,
        )

        result_padded_boxes_1, results_padded_scores_1 = self.postprocess_bbox(output_1, boxes_1)
        result_padded_boxes_2, results_padded_scores_2 = self.postprocess_bbox(output_2, boxes_2)

        return result_padded_boxes_1, results_padded_scores_1, result_padded_boxes_2, results_padded_scores_2

    def construct(self, input_tensor, proposal_1, proposal_mask_1, proposal_2, proposal_mask_2, img_metas):
        """construct"""
        features = self.feature_extractor(input_tensor)
        output_1 = self.inference_head(features, (proposal_1,), (proposal_mask_1,), img_metas)
        output_2 = self.inference_head(features, (proposal_2,), (proposal_mask_2,), img_metas)
        return output_1, output_2
