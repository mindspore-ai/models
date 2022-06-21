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
"""data_augment"""
import math
import random
import sys
import numpy as np
import mindspore.dataset.vision.py_transforms as transforms

class Transforms():
    def __init__(self):
        pass
    def __call__(self, img, boxes):
        if random.random() < 0.3:
            img, boxes = colorJitter(img, boxes)
        if random.random() < 0.5:
            img, boxes = random_rotation(img, boxes)
        return img, np.array(boxes)

def colorJitter(img, boxes, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
    img = transforms.RandomColorAdjust(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)(img)
    return img, boxes

def random_rotation(img, boxes, degree=10):
    d = random.uniform(-degree, degree)
    w, h = img.size
    rx0, ry0 = w / 2.0, h / 2.0
    img = img.rotate(d)
    a = -d / 180.0 * math.pi
    new_boxes = np.zeros_like(boxes)
    new_boxes[:, 0] = boxes[:, 1]
    new_boxes[:, 1] = boxes[:, 0]
    new_boxes[:, 2] = boxes[:, 3]
    new_boxes[:, 3] = boxes[:, 2]
    for i in range(boxes.shape[0]):
        ymin, xmin, ymax, xmax = new_boxes[i, :]
        xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
        x0, y0 = xmin, ymin
        x1, y1 = xmin, ymax
        x2, y2 = xmax, ymin
        x3, y3 = xmax, ymax
        z = np.array([[y0, x0], [y1, x1], [y2, x2], [y3, x3]], dtype=np.float32)
        tp = np.zeros_like(z)
        tp[:, 1] = (z[:, 1] - rx0) * math.cos(a) - (z[:, 0] - ry0) * math.sin(a) + rx0
        tp[:, 0] = (z[:, 1] - rx0) * math.sin(a) + (z[:, 0] - ry0) * math.cos(a) + ry0
        ymax, xmax = np.max(tp, axis=0)
        ymin, xmin = np.min(tp, axis=0)
        new_boxes[i] = np.stack([ymin, xmin, ymax, xmax])
    new_boxes[:, 1::2] = np.clip(new_boxes[:, 1::2], 0, w - 1)
    new_boxes[:, 0::2] = np.clip(new_boxes[:, 0::2], 0, h - 1)
    boxes[:, 0] = new_boxes[:, 1]
    boxes[:, 1] = new_boxes[:, 0]
    boxes[:, 2] = new_boxes[:, 3]
    boxes[:, 3] = new_boxes[:, 2]
    return img, boxes

def _box_inter(box1, box2):
    tl = np.maximum(box1[:, None, :2], box2[:, :2])  # [n,m,2]
    br = np.minimum(box1[:, None, 2:], box2[:, 2:])  # [n,m,2]
    inter_tensor = np.array((br-tl), dtype=np.float32)
    hw = np.clip(inter_tensor, 0, sys.maxsize)  # [n,m,2]
    inter = hw[:, :, 0] * hw[:, :, 1]  # [n,m]
    return inter
