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
""" servable entrance module """
import cv2
import numpy as np
import mindspore.dataset.vision.c_transforms as VC
from mindspore_serving.server import register


def preprocess(image, input_size=(640, 640)):
    decode = VC.Decode()
    image = decode(image)

    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    scale = min(input_size[0] / image.shape[0], input_size[1] / image.shape[1])
    resized_img = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)),
                             interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    padded_img[: int(image.shape[0] * scale), : int(image.shape[1] * scale)] = resized_img

    padded_img = padded_img.transpose((2, 0, 1))
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, scale, image


def postprocess(prediction, scale, image_origin, nms_thre, conf_thre, num_classes):
    """ nms """
    if len(prediction.shape) < 3:
        prediction = prediction[None]

    box_corner = np.zeros_like(prediction)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]
    all_bboxes = [None for _ in range(len(prediction))]
    all_scores = all_bboxes.copy()
    all_cls = all_bboxes.copy()

    if isinstance(nms_thre, np.ndarray):
        nms_thre = nms_thre.item()

    if isinstance(conf_thre, np.ndarray):
        conf_thre = conf_thre.item()

    if isinstance(num_classes, np.ndarray):
        num_classes = num_classes.item()

    for i, image_pred in enumerate(prediction):
        if not image_pred.shape[0]:
            continue
        # Get score and class with highest confidence
        class_conf = np.max(image_pred[:, 5:5 + num_classes], axis=-1)
        class_pred = np.argmax(image_pred[:, 5:5 + num_classes], axis=-1)
        conf_mask = (image_pred[:, 4] * class_conf >= conf_thre).squeeze()
        class_conf = np.expand_dims(class_conf, axis=-1)
        class_pred = np.expand_dims(class_pred, axis=-1).astype(np.float16)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = np.concatenate((image_pred[:, :5], class_conf, class_pred), axis=1)
        detections = detections[conf_mask]
        if not detections.shape[0]:
            continue

        nms_out_index = batch_nms(detections[:, :4], detections[:, 4] * detections[:, 5], detections[:, 6], nms_thre)
        detections = detections[nms_out_index]

        h_origin, w_origin, _ = image_origin.shape
        bboxes = detections[:, 0:4]
        bboxes = bboxes / scale
        bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, w_origin)
        bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, h_origin)
        cls = detections[:, 6]
        scores = detections[:, 4] * detections[:, 5]

        all_bboxes[i] = bboxes
        all_scores[i] = scores
        all_cls[i] = cls

    return all_bboxes, all_scores, all_cls


def nms(xyxys, scores, threshold):
    """Calculate NMS"""
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
    """Calculate Nms based on class info,Each index value correspond to a category,
    and NMS will not be applied between elements of different categories."""
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


yolox_model = register.declare_model(model_file='yolofpn.mindir', model_format='MindIR')


@register.register_method(output_names=['bboxes', 'scores', 'cls'])
def inference(image, input_size, nms_thre, conf_thre, num_classes):
    image, scale, image_origin = register.add_stage(preprocess, image, input_size, outputs_count=3)
    prediction = register.add_stage(yolox_model, image, outputs_count=1)
    all_bboxes, all_scores, all_cls = register.add_stage(postprocess, prediction, scale, image_origin, nms_thre,
                                                         conf_thre, num_classes, outputs_count=3)
    return all_bboxes, all_scores, all_cls
