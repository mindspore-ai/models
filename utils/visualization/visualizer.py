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
'''
This module contains image visualizing functions for detection and segmentation tasks based on OpenCV.
'''
import numpy as np
import cv2


def get_n_hsv_colors(num_color):
    """return a specific number of hsv colors."""
    h = [i * 360 / num_color for i in range(num_color)]
    s = 0.6
    v = 0.6
    hsv_color = []
    for hi in h:
        hsv_color.append([hi, s, v])
    return hsv_color


def draw_bbox(
        image, boxes, class_name,
        bbox_mode=0):
    """draw bounding boxes for predictions.

    :param image: Image to visualize.
    :type image: nd.array with shape H*W*3
    :param boxes: List of predition boxes
    :type boxes: nd.array in the format: [[x1, y1, w, h, cls, conf],...] or [[x1, y1, x2, y2, cls, conf],...]
    :param class_name: List of names of the classes
    :type class_name: [str]
    :param bbox_mode: Box mode with xywh or 'xyxy', default to 'xywh'
    :type bbox_mode: 0 for 'xywh' or 1 for 'xyxy'
    """

    num_class = len(class_name)
    hsv_color = get_n_hsv_colors(num_class)
    if bbox_mode == 1:
        for box in boxes:
            x1, y1, x2, y2, label, score = box[0], box[1], box[2], box[3], box[4], box[5]
            cv2.rectangle(image, (x1, y1), (x2, y2), hsv_color[label], 1, 8)
            cv2.putText(image, class_name[label] + '(' + str(score) + ')', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                        1, hsv_color[label], 2, cv2.LINE_AA)
    else:
        for box in boxes:
            x1, y1, w, h, label, score = box[0], box[1], box[2], box[3], box[4], box[5]
            cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), hsv_color[label], 1, 8)
            cv2.putText(image, class_name[label] + '(' + str(score) + ')', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                        1, hsv_color[label], 2, cv2.LINE_AA)
    cv2.imshow('draw_bboxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_mask(output, num_class):
    """draw masks for predictions.

    :param output: Output mask to visualize.
    :type output: nd.array with shape H*W
    :param num_class: The number of output types for the segmentation task
    :type num_class: int
    """
    m, n = output.shape
    mask = np.zeros((m, n, 3))
    for i in range(m):
        for j in range(n):
            mask[i, j] = [0, 0, (output[i, j] / num_class)]
    cv2.imshow('draw_masks', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
