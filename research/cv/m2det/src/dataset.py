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

import math
import random

import cv2
import numpy as np
from mindspore import Tensor
from mindspore import dataset as de
from mindspore import dtype as mstype
from mindspore import set_seed
from mindspore.communication import get_group_size
from mindspore.communication import get_rank

from src.box_utils import match
from src.box_utils import matrix_iou
from src.coco_utils import COCODetection


def _crop(image, boxes, labels):
    height, width, _ = image.shape

    if boxes.size == 0:
        return image, boxes, labels

    while True:
        mode = random.choice((
            None,
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            (None, None),
        ))

        if mode is None:
            return image, boxes, labels

        min_iou, max_iou = mode
        if min_iou is None:
            min_iou = float('-inf')
        if max_iou is None:
            max_iou = float('inf')

        for _ in range(50):
            scale = random.uniform(0.3, 1.)
            min_ratio = max(0.5, scale * scale)
            max_ratio = min(2, 1. / scale / scale)
            ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
            w = int(scale * ratio * width)
            h = int((scale / ratio) * height)

            left = random.randrange(width - w)
            top = random.randrange(height - h)
            roi = np.array((left, top, left + w, top + h))

            iou = matrix_iou(boxes, roi[np.newaxis])

            if not (min_iou <= iou.min() and iou.max() <= max_iou):
                continue

            image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            mask = np.logical_and(roi[:2] < centers, centers < roi[2:]) \
                .all(axis=1)
            boxes_t = boxes[mask].copy()
            labels_t = labels[mask].copy()
            if boxes_t.size == 0:
                continue

            boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
            boxes_t[:, :2] -= roi[:2]
            boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
            boxes_t[:, 2:] -= roi[:2]

            return image_t, boxes_t, labels_t


def _distort(image):
    def _convert(image_, alpha=1, beta=0):
        tmp_ = image_.astype(float) * alpha + beta
        tmp_[tmp_ < 0] = 0
        tmp_[tmp_ > 255] = 255
        image_[:] = tmp_

    image = image.copy()

    if random.randrange(2):
        _convert(image, beta=random.uniform(-32, 32))

    if random.randrange(2):
        _convert(image, alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if random.randrange(2):
        tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        image[:, :, 0] = tmp

    if random.randrange(2):
        _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def _expand(image, boxes, fill, p):
    if random.random() > p:
        return image, boxes

    height, width, depth = image.shape
    for _ in range(50):
        scale = random.uniform(1, 4)

        min_ratio = max(0.5, 1. / scale / scale)
        max_ratio = min(2, scale*scale)
        ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
        ws = scale * ratio
        hs = scale / ratio
        if ws < 1 or hs < 1:
            continue
        w = int(ws * width)
        h = int(hs * height)

        left = random.randint(0, w - width)
        top = random.randint(0, h - height)

        boxes_t = boxes.copy()
        boxes_t[:, :2] += (left, top)
        boxes_t[:, 2:] += (left, top)

        expand_image = np.empty(
            (h, w, depth),
            dtype=image.dtype)
        expand_image[:, :] = fill
        expand_image[top:top + height, left:left + width] = image
        image = expand_image

        return image, boxes_t


def _mirror(image, boxes):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


def preproc_for_test(image, insize, mean):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    image = cv2.resize(image, (insize, insize), interpolation=interp_method)
    image = image.astype(np.float32)
    image -= mean
    return image.transpose(2, 0, 1)


class preproc:

    def __init__(self, resize, rgb_means, p):
        self.means = rgb_means
        self.resize = resize
        self.p = p

    def __call__(self, image, targets):
        boxes = targets[:, :-1].copy()
        labels = targets[:, -1].copy()
        if boxes.size == 0:
            targets = np.zeros((1, 5))
            image = preproc_for_test(image, self.resize, self.means)
            return image, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :-1]
        labels_o = targets_o[:, -1]
        boxes_o[:, 0::2] /= width_o
        boxes_o[:, 1::2] /= height_o
        labels_o = np.expand_dims(labels_o, 1)
        targets_o = np.hstack((boxes_o, labels_o))

        image_t, boxes, labels = _crop(image, boxes, labels)
        image_t = _distort(image_t)
        image_t, boxes = _expand(image_t, boxes, self.means, self.p)
        image_t, boxes = _mirror(image_t, boxes)

        height, width, _ = image_t.shape
        image_t = preproc_for_test(image_t, self.resize, self.means)
        boxes = boxes.copy()
        boxes[:, 0::2] /= width
        boxes[:, 1::2] /= height
        b_w = (boxes[:, 2] - boxes[:, 0])*1.
        b_h = (boxes[:, 3] - boxes[:, 1])*1.
        mask_b = np.minimum(b_w, b_h) > 0.01
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b].copy()

        if boxes_t.size == 0:
            image = preproc_for_test(image_o, self.resize, self.means)
            return image, targets_o

        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((boxes_t, labels_t))

        return image_t, targets_t


class BaseTransform:
    """Defines the transformations that should be applied to test PIL image
        for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels
    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, resize, rgb_means, swap=(2, 0, 1)):
        self.means = rgb_means
        self.resize = resize
        self.swap = swap

    # assume input is cv2 img for now
    def __call__(self, img):

        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = interp_methods[0]
        img = cv2.resize(np.array(img), (self.resize,
                                         self.resize), interpolation=interp_method).astype(np.float32)
        img -= self.means
        img = img.transpose(self.swap)
        return Tensor(img, dtype=mstype.float32)


def target_preprocess(img, annotation, cfg, priors):
    loc, conf = match(cfg.loss['overlap_thresh'],
                      annotation[:, :-1],
                      priors,
                      [0.1, 0.2],
                      annotation[:, -1])
    return img, loc, conf.astype(np.int32)


def get_dataset(cfg, dataset, priors, setname='train_sets', random_seed=None, distributed=False):
    _preproc = preproc(cfg.model['input_size'], cfg.model['rgb_means'], cfg.model['p'])
    Dataloader_function = {'COCO': COCODetection}
    _Dataloader_function = Dataloader_function[dataset]
    shuffle = False
    if random_seed is not None:
        set_seed(random_seed)
        shuffle = True

    if setname == 'train_sets':
        generator = _Dataloader_function(cfg.COCOroot if dataset == 'COCO' else cfg.VOCroot,
                                         cfg.dataset[dataset][setname], _preproc)
    else:
        generator = _Dataloader_function(cfg.COCOroot if dataset == 'COCO' else cfg.VOCroot,
                                         cfg.dataset[dataset][setname], None)
    if distributed:
        rank_id = get_rank()
        rank_size = get_group_size()
        ds = de.GeneratorDataset(source=generator,
                                 column_names=['img', 'annotation'],
                                 num_parallel_workers=cfg.train_cfg['num_workers'],
                                 shuffle=shuffle,
                                 num_shards=rank_size,
                                 shard_id=rank_id)
    else:
        ds = de.GeneratorDataset(source=generator,
                                 column_names=['img', 'annotation'],
                                 num_parallel_workers=cfg.train_cfg['num_workers'],
                                 shuffle=shuffle)
    target_preprocess_function = (lambda img, annotation: target_preprocess(img, annotation, cfg, priors))
    ds = ds.map(operations=target_preprocess_function, input_columns=['img', 'annotation'],
                output_columns=['img', 'loc', 'conf'], column_order=['img', 'loc', 'conf'])
    ds = ds.batch(cfg.train_cfg['per_batch_size'], drop_remainder=True)

    return ds, generator
