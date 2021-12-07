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
# ===========================================================================
"""Prepare Cityscapes dataset"""
import random

import cv2
import numpy as np

import mindspore.dataset as ds


def normalize(image, label):
    """normalize"""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    image = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
    label = cv2.imdecode(np.frombuffer(label, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    _img1 = image / 255.0
    _img2 = _img1 - mean
    _img3 = _img2 / std
    out_img = _img3.transpose((2, 0, 1))
    return out_img.astype(np.float32), label.astype(np.int32)


def train_preprocess(image, label, crop_size=None, ignore_label=255):
    """train_preprocess"""
    min_scale, max_scale = 0.5, 2.0

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    image = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
    label = cv2.imdecode(np.frombuffer(label, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

    # flip images
    if random.random() < 0.5:
        flipped_img = image[:, ::-1, :]
        flipped_lbl = label[:, ::-1]
    else:
        flipped_img = image
        flipped_lbl = label

    # scale images
    h, w = flipped_img.shape[0], flipped_img.shape[1]
    random_scale = random.uniform(min_scale, max_scale)
    new_size = (int(round(w * random_scale)), int(round(h * random_scale)))
    scaled_img = cv2.resize(flipped_img, (new_size[0], new_size[1]), interpolation=cv2.INTER_CUBIC)
    scaled_lbl = cv2.resize(flipped_lbl, (new_size[0], new_size[1]), interpolation=cv2.INTER_NEAREST)

    new_h, new_w = scaled_img.shape[0], scaled_img.shape[1]
    pad_h, pad_w = max(0, crop_size[0] - new_h), max(0, crop_size[1] - new_w)
    if pad_h > 0 or pad_w > 0:
        pad_img = cv2.copyMakeBorder(scaled_img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        pad_lbl = cv2.copyMakeBorder(scaled_lbl, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=ignore_label)
    else:
        pad_img = scaled_img
        pad_lbl = scaled_lbl

    # crop images
    h, w = pad_img.shape[0], pad_img.shape[1]
    i = random.randint(0, h - crop_size[0])
    j = random.randint(0, w - crop_size[1])
    cropped_img = pad_img[i:i + crop_size[0], j:j + crop_size[1], :]
    cropped_lbl = pad_lbl[i:i + crop_size[0], j:j + crop_size[1]]

    # normalization
    _img0 = cropped_img / 255.0
    _img1 = _img0 - mean
    _img2 = _img1 / std
    out_img = _img2.transpose((2, 0, 1))
    return out_img.astype(np.float32), cropped_lbl.astype(np.uint8)


def CityScapesDataset(mindrecord_file, process_option='train', ignore_label=255, crop_size=(769, 769),
                      num_shards=1, shard_id=None, shuffle=True):
    """CityScapesDataset"""
    if process_option == 'train':
        dataset = ds.MindDataset(mindrecord_file, columns_list=['image', 'label'], num_parallel_workers=2,
                                 num_shards=num_shards, shard_id=shard_id, shuffle=shuffle)
        preprocess = lambda _img, _lbl: train_preprocess(_img, _lbl, crop_size=crop_size, ignore_label=ignore_label)
    elif process_option == 'eval':
        dataset = ds.MindDataset(mindrecord_file, columns_list=['image', 'label'],
                                 num_parallel_workers=2, shuffle=shuffle)
        preprocess = normalize
    else:
        raise ValueError("Unknown option")

    dataset = dataset.map(operations=preprocess, input_columns=['image', 'label'],
                          output_columns=["image", "label"], num_parallel_workers=4)

    return dataset
